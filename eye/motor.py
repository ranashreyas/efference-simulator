"""Motor history encoding: the efference copy the forward model receives. Shared by the live
loop and training so both always use the same format.

Input: the last STEPS eye states, one per camera frame (oldest first, ending at the frame being
predicted), each (commanded pan, commanded tilt, actual pan, actual tilt) in degrees. The
commanded angle is the command as sent (it leads the motion); the actual angle gets
ACTUAL_NOISE_DEG of noise, like a real position sensor (e.g. a servo's potentiometer), so the
model can't lean on a perfect reading real hardware won't give it.

Output: ACTION_DIM = STEPS * 4 + 3 numbers:
    relative   every angle minus the latest commanded angle, in units of REL_SCALE_DEG. What
               moves the image is how the angles *change*; as absolute angles those changes
               would be ~1% of the input's size, here they're ~1.
    absolute   the latest commanded pan and tilt / SCALE_DEG (how close the eye is to an edge)
    available  1 when the motor history is real, 0 when it's blanked. A blanked input (all
               zeros) then can't be confused with a camera that's known to be still.
At 30 fps the history covers ~0.27 s, longer than the servo's reaction delay.
"""
import numpy as np

STEPS = 8
SCALE_DEG = 20.0
REL_SCALE_DEG = 0.5
REL_CLIP = 20.0
ACTUAL_NOISE_DEG = 0.2
ACTION_DIM = STEPS * 4 + 3
# entries whose sign flips when the image is mirrored left-right: every relative pan + absolute pan
PAN_INDEX = [k * 4 + c for k in range(STEPS) for c in (0, 2)] + [STEPS * 4]


def encode(states, actual_noise_deg=0.0, rng=None):
    """states: MotorState list, oldest first. Shorter histories are padded with the oldest."""
    states = list(states)[-STEPS:]
    if not states:
        return blank()
    states = [states[0]] * (STEPS - len(states)) + states
    rows = np.asarray([(s.pan_target, s.tilt_target, s.pan, s.tilt) for s in states], np.float32)
    return encode_rows(rows, actual_noise_deg, rng)


def encode_rows(rows, actual_noise_deg=0.0, rng=None):
    """rows: (..., STEPS, 4) degrees as (cmd pan, cmd tilt, actual pan, actual tilt), oldest first."""
    rows = np.array(rows, np.float32)
    if actual_noise_deg:
        rows[..., 2:] += (rng or np.random.default_rng()).normal(0, actual_noise_deg, rows[..., 2:].shape)
    ref = rows[..., -1:, :2]                                         # latest commanded pan, tilt
    rel = np.clip((rows - np.concatenate([ref, ref], -1)) / REL_SCALE_DEG, -REL_CLIP, REL_CLIP)
    lead = rows.shape[:-2]
    absolute = ref.reshape(*lead, 2) / SCALE_DEG
    available = np.ones((*lead, 1), np.float32)
    return np.concatenate([rel.reshape(*lead, STEPS * 4), absolute, available], -1).astype(np.float32)


def blank(n=None):
    """Motor input marked unavailable (all zeros)."""
    return np.zeros(ACTION_DIM if n is None else (n, ACTION_DIM), np.float32)


def still(n=None):
    """Motor input for a camera that's known not to move (what the static eye sends)."""
    return encode_rows(np.zeros((STEPS, 4) if n is None else (n, STEPS, 4)))
