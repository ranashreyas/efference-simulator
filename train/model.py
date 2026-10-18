"""The forward model: predicts the next frame from the last few frames and the motor history.

    frames (B, 3, 96, 128) in [0, 1] ─┐
                                      ├─► encoder ─► bottleneck ─► decoder ─► 3 heads
    motor history (B, 32) ─► MLP ─► FiLM (scale/shift of encoder features at 12x16 and 6x8)

    heads   flow      (B, 2, H, W)  pixels each pixel moved since the last frame
            residual  (B, 1, H, W)  change that motion can't explain
            logvar    (B, 1, H, W)  log of the expected squared error per pixel

    prediction = warp(last frame, flow) + residual

Flow and residual heads start at zero, so an untrained model predicts "next frame = last frame".
With a fixed camera the motor history is all zeros; the input is wired for the moving eye.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FRAMES = 3
ACTION_DIM = 32          # 8 time steps x (commanded pan, tilt, estimated pan, tilt)


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(conv(cin, cout, 2), conv(cout, cout))

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    """Upsample x2, concatenate the encoder skip, convolve."""

    def __init__(self, cin, cskip, cout, double=True):
        super().__init__()
        layers = [conv(cin + cskip, cout)] + ([conv(cout, cout)] if double else [])
        self.body = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat([x, skip], 1))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv(c, c), nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.body(x))


class FiLM(nn.Module):
    """Per-channel scale and shift from the motor embedding. Starts as the identity."""

    def __init__(self, emb, c):
        super().__init__()
        self.lin = nn.Linear(emb, 2 * c)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, e):
        gamma, beta = self.lin(e).chunk(2, dim=1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


def warp(img, flow):
    """Move each pixel of img by flow (in pixels): backward warp with bilinear sampling."""
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device, dtype=img.dtype),
                            torch.arange(w, device=img.device, dtype=img.dtype), indexing="ij")
    x = (xs - flow[:, 0]) / (w - 1) * 2 - 1
    y = (ys - flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([x, y], -1), mode="bilinear", padding_mode="border",
                         align_corners=True)


class ForwardModel(nn.Module):
    def __init__(self, widths=(16, 32, 64, 96), emb=64):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.enc1, self.enc2 = Down(N_FRAMES, w1), Down(w1, w2)     # 48x64, 24x32
        self.enc3, self.enc4 = Down(w2, w3), Down(w3, w4)           # 12x16, 6x8
        self.mid = ResBlock(w4)
        self.motor = nn.Sequential(nn.Linear(ACTION_DIM, emb), nn.ReLU(inplace=True),
                                   nn.Linear(emb, emb), nn.ReLU(inplace=True))
        self.film3, self.film4 = FiLM(emb, w3), FiLM(emb, w4)
        self.up3 = Up(w4, w3, w3)                                   # 12x16
        self.up2 = Up(w3, w2, w2)                                   # 24x32
        self.up1 = Up(w2, w1, 24, double=False)                     # 48x64: flow lives here
        self.up0 = Up(24, N_FRAMES, 16, double=False)               # 96x128
        self.flow_head = nn.Conv2d(24, 2, 3, 1, 1)
        self.res_head = nn.Conv2d(16, 1, 3, 1, 1)
        self.logvar_head = nn.Conv2d(16, 1, 3, 1, 1)
        for head in (self.flow_head, self.res_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -6.0)              # ~ (5% intensity)^2 to start

    def forward(self, frames, motor=None):
        if motor is None:
            motor = frames.new_zeros(frames.shape[0], ACTION_DIM)
        e = self.motor(motor)
        s1 = self.enc1(frames)
        s2 = self.enc2(s1)
        s3 = self.film3(self.enc3(s2), e)
        x = self.mid(self.film4(self.enc4(s3), e))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        flow = F.interpolate(self.flow_head(x), size=frames.shape[-2:], mode="bilinear",
                             align_corners=False) * 2.0            # half-res pixels -> full-res
        x = self.up0(x, frames)
        residual = self.res_head(x)
        logvar = self.logvar_head(x).clamp(-12.0, 2.0)
        pred = warp(frames[:, -1:], flow) + residual
        return pred, logvar, flow


# Expected error is never allowed below ~2 gray levels of camera noise. Without this floor the
# model claims sub-noise precision on still background and its surprise map fills with speckle.
LOGVAR_FLOOR = math.log((2 / 255) ** 2)


def surprise_map(target, pred, logvar):
    """Precision-weighted squared error: how much worse the prediction was than expected."""
    return (target - pred) ** 2 * torch.exp(-logvar.clamp(min=LOGVAR_FLOOR))
