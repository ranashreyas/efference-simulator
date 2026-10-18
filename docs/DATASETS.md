# Datasets

We train the forward model on **normal video only** and treat badly predicted frames as
surprising. Evaluation asks: does surprise separate labeled anomalous frames from normal
ones (frame-level AUROC)?

Three sources, all following the same convention: **train = normal only, test = labeled anomalies.**

| | ShanghaiTech Campus | NWPU Campus | Your room |
|---|---|---|---|
| Scenes | 13 | 43 | 1 for now (`--scene`) |
| Train / test videos | 330 / 107 | 305 / 242 | your episodes |
| Frames (train / test) | 274,515 / 42,883 | 1,082,014 / 384,059 | ~108k per hour |
| Resolution | 856×480 | mixed: 704×576 up to 2048×1536 | 1280×720 (logged at 320×180) |
| Frame rate | ~24 fps (not officially stated) | 25 fps | 30 fps |
| Download | ~6.5 GB | 76.6 GB | — |
| Labels | per-frame + pixel masks | per-frame only | per-frame (`labels.npy`) |
| License | none stated; cite the paper | academic, non-commercial, no redistribution or derived datasets | yours |

Items marked **[unverified]** could not be confirmed from a primary source during research (Oct 2026).

---

## ShanghaiTech Campus

- Homepage: https://svip-lab.github.io/dataset/campus_dataset.html
- Paper (Luo et al., ICCV 2017): https://openaccess.thecvf.com/content_iccv_2017/html/Luo_A_Revisit_of_ICCV_2017_paper.html
- Download (OneDrive, no login): https://1drv.ms/u/s!AjjUqiJZsj8whLt-1ABerTT-9eH9Ag?e=eJbY6Y
  - A folder "splits" with `shanghaitech.tar.gz.aa` … `.ag` (six 1 GB parts + 525 MB) and a `readme.txt`.
  - Join, then extract: `cat shanghaitech.tar.gz.* > shanghaitech.tar.gz && tar xzf shanghaitech.tar.gz`
  - The "Google Drive" button on the homepage is a dead link.

**Layout after extraction** (as read by anomalib's loader):
```
training/videos/XX_YYY.avi            # normal training VIDEOS; XX = scene id, e.g. 01_001.avi
testing/frames/XX_YYYY/*.jpg          # test set is extracted FRAMES; e.g. 01_0014/
testing/test_frame_mask/XX_YYYY.npy   # per-frame 0/1 labels  <- what we evaluate on
testing/test_pixel_mask/XX_YYYY.npy   # T×H×W pixel masks (optional)
```

**Gotchas**
- Training is videos and test is JPEG frames, so the loader needs two code paths.
- The original `.avi` files decode badly in some libraries (anomalib re-encodes them to XVID). Decode with OpenCV and check the frame counts.
- **Scene 13 has training videos only, no test videos.** The test set covers scenes 01–12, and scene 01 alone is 34 of the 107 test videos.
- There's no validation set. Many papers effectively tune on the test set. Hold out some training videos for model selection instead.
- **Don't use the "weakly supervised" re-split** (Zhong et al., CVPR 2019: 238 train / 199 test), which puts anomalies *in training*. Repos like https://github.com/tianyu0207/RTFM ship that split. Use the original split from the download above.
- Don't compare against **HR-ShanghaiTech**, a human-only subset used by pose-based methods; its numbers aren't comparable.

**Published bar:** next-frame prediction (FFP, Liu et al. CVPR 2018) reaches **72.8% frame AUROC**.
Paper https://arxiv.org/abs/1712.09867, code https://github.com/StevenLiuWen/ano_pred_cvpr2018.
Their scores are normalized per video before AUROC [unverified from the primary PDF].

## NWPU Campus

- Homepage: https://campusvad.github.io/ (extended version: https://campusvaa.github.io/)
- Paper (Cao et al., CVPR 2023): https://arxiv.org/abs/2305.13611
- Download (Google Drive, public): https://drive.google.com/drive/folders/1_EztmkNpTPyVb4lM0m4rLTXgXo_LzgF1?usp=share_link
  - `videos/NWPUCampusDataset.7z.001` … `.080` (80 parts, 76.6 GB total) + `md5.txt`
  - `groundtruth/NWPU_Campus_gt.npz` (3 MB) + `read_gt_file.py`
  - Mirror: BaiduYun https://pan.baidu.com/s/1mhhxdYYKYZsWBO36NX655g (code `il54`); slow outside China.
- Official code and AUROC implementation: https://github.com/zugexiaodui/campus_vad_code (`main/metrics.py`, `cal_micro_auc`)

**Labels:** `NWPU_Campus_gt.npz` holds 242 keys like `D001_01` (scene `D001`, video `01`), each a 1-D
array of per-frame 0/1 labels. There are no pixel masks. All 43 scenes appear in the test set;
118 test videos are entirely normal and 124 contain anomalies.
What's inside the 7z (likely `.avi` files) and the training file names are **[unverified]**.

**Gotchas**
- **Scene-dependent anomalies:** 4 of the 28 classes (cycling on a footpath, wrong turn, trucks,
  photographing in a restricted area) are normal in some scenes and anomalous in others. A model
  that pools all scenes without knowing which scene it's in will miss these. That's expected,
  so report them separately, or give the model a scene input later.
- **Mixed resolutions and aspect ratios:** we center-crop to 4:3, then resize to 128×96.
- **License:** academic and non-commercial only, no redistribution, no derived datasets. Fine for
  training and for reporting numbers. Don't publish preprocessed copies, and only show frames
  in academic-style write-ups.

**Published bars (frame AUROC):** prediction methods land around 62–65% (MNAD 62.5, MPN 64.4,
AMMC-Net 64.5, HF2-VAD 63.7); the authors' scene-aware method reaches 68.2. FFP wasn't run on NWPU.

## Your room

Record with the Pi (episodes land in `runs/<scene>_<split>_<timestamp>/` on the Pi):

```bash
# training data: normal activity only, nothing staged
python -m eye.run --log --scene bedroom --split train

# test data: stage anomalies and mark them with SPACE in the viewer
python -m eye.run --log --scene bedroom --split test
```

- Press **SPACE** in the viewer when an anomaly starts and again when it ends. The window shows a
  red border while an interval is open, and the border only appears once the Pi has received the label.
- Each episode stores `labels.npy` (per-frame 0/1, the same format as ShanghaiTech's
  `test_frame_mask`) and `labels.json` (the raw intervals).
- Live labels are about 0.2–0.5 s late (reaction time). Start an interval a moment *before* the
  event if you can see it coming, or widen intervals slightly at evaluation time.
- Ideas for test clips: someone walks in, something is dropped or thrown, a light switches on,
  a door slams, the camera gets bumped. Also record **predictable motion with no anomaly**
  (fan on, screen playing video, curtains moving); those frames should score low.
- Vary time of day and lighting across episodes so the model doesn't learn one lighting setup.

---

## How we combine them

**Preprocess everything to one format first:** grayscale, center-crop to 4:3, resize to 128×96,
resample to a common rate by timestamp (start with 15 fps; 24, 25 and 30 don't share a
clean divisor, so take the nearest frame to each 1/15 s tick), and store per-video frame
stacks plus per-frame labels. After that, training and
evaluation code never needs to know which dataset a clip came from.

**Training:** ShanghaiTech train (13 scenes) + NWPU train (43 scenes) + your room's `train` episodes.
Hold out a few training videos per source for validation. Never tune on test.

**Evaluation**, reported per source and never pooled into one number:
1. **ShanghaiTech test** for comparison with published work (72.8% FFP).
2. **NWPU test**, both overall and with the scene-dependent classes split out.
3. **Your room test**, which is the deployment domain (C922 noise, your lighting, 30 fps).
4. **Generalization:** retrain with some scenes left out entirely and evaluate on those. This is
   the direct test of "does it overfit to the scenes it saw?"

Metric: frame-level micro AUROC (all test frames of a source concatenated), as the official
NWPU `cal_micro_auc` does. Report raw scores and smoothed/normalized scores separately.
Smoothing and per-video normalization noticeably inflate AUROC, and papers aren't consistent about using them.

**Disk:** NWPU needs about 77 GB for the archive plus about as much again to extract. Your Mac has
plenty of space, but preprocess scene by scene and delete the raw videos as you go. At 128×96
grayscale the whole NWPU set fits in roughly 18 GB uncompressed, and much less compressed.
Start with ShanghaiTech (6.5 GB) to get the pipeline working, then add NWPU.
