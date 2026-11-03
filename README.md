# not-a-deepfake: Deepfake Detection in Video Calls

We build a classifier that decides whether the face of a video-call participant is real or a deepfake, using the visual stream only.
## Research questions

1. Which pretrained backbone (CNN or Transformer) gives the best frame-level
   detection, especially on manipulation methods it has not seen in training?
2. Does learned temporal aggregation (LSTM or Transformer) improve on simple
   averaging of frame scores?

## Approach

| Stage | Options compared |
|---|---|
| Frame classifier | Xception, EfficientNet-B4 (with and without Self-Blended Images), ViT-B/16, CLIP ViT-B/16 |
| Temporal aggregation | Mean / top-k pooling, LSTM, Transformer encoder (windows of ~32 frames) |
| Conditions | Clean footage and simulated call quality (compression, low resolution, blur, lighting) |

**Evaluation protocol:** train on FaceForensics++ (c23) and Celeb-DF v2, then
test on an unseen dataset: a DFDC subset. Each dataset's own test split is
kept for in-domain results (Celeb-DF: the official 518-video test list).

**Metrics:** frame- and video-level AUC, equal error rate (EER: the error rate
at the threshold where real faces flagged and fakes missed are equally common),
accuracy, F1, TPR at low FPR, and inference time.


## Setup

### GPU machine

The GPU machine runs Ronin's Deep Learning GPU (PyTorch) image, which already
has PyTorch installed in `/opt/pytorch`, matched to the GPU driver. Install the
other packages into that environment instead of making a new `.venv`. A new
venv would download its own `torch`, which may not match the driver.

```
git clone https://github.com/rndllchw/not-a-deepfake.git
cd not-a-deepfake
source /opt/pytorch/bin/activate
pip install -r requirements.txt     # skips torch/torchvision, which are already installed
```

So on this machine, the line to run at the start of every session is
`source /opt/pytorch/bin/activate`, not `source .venv/bin/activate`.

If `git clone` asks for a password, log in with `gh auth login` or a GitHub
personal access token typed at the prompt. Never save a token in the repo.


## Data

Datasets are **not** stored in this repository. The FaceForensics++ terms of use
forbid redistribution, and the files are too large for git.

All data lives on the project's Ronin GPU machine, on a separate 100 GB data
drive: `/mnt/sdd/data`, linked to `~/data`. The drive mounts by itself when the
machine starts (it is in `/etc/fstab`); if `~/data` is missing, run
`sudo mount -a`. `~/data/README.md` says what is where.

### Raw data layout

```
~/data/raw/
├── ffpp/                                   FaceForensics++, as downloaded
│   ├── original_sequences/youtube/c23/videos/
│   │   └── 000.mp4 … 999.mp4               1,000 real videos, one person each
│   └── manipulated_sequences/
│       ├── Deepfakes/c23/videos/           1,000 fakes per method
│       ├── Face2Face/c23/videos/
│       ├── FaceSwap/c23/videos/
│       └── NeuralTextures/c23/videos/
│           └── 000_003.mp4 …               fake made from real videos 000 and 003
├── celebdf/                                Celeb-DF v2
│   ├── Celeb-real/
│   │   └── id0_0000.mp4 …                  590 real clips of 59 celebrities (person 0, clip 0000)
│   ├── YouTube-real/
│   │   └── 00000.mp4 …                     300 extra real clips of other people from YouTube
│   ├── Celeb-synthesis/
│   │   └── id0_id16_0000.mp4 …             5,639 fakes, faces swapped between celebrities 0 and 16
│   └── List_of_testing_videos.txt          official 518-video test list ("1" = real, "0" = fake)
└── dfdc/                                   DFDC, any folder layout below this
    └── …/abcdefghij.mp4                    labelled by a metadata.json or labels.csv found anywhere under dfdc/
```

`c23` is FF++'s high-quality compression level; FF++ also comes as `raw`
(uncompressed) and `c40` (heavily compressed). The YouTube-real clips have no
person ids and are never used to make fakes, so each counts as its own person.

### Splits

Videos are split **by person**, so nobody seen in training appears in
validation or test. Otherwise a model can score well by recognising faces and
backgrounds instead of detecting manipulation.

- **FaceForensics++:** the official train/val/test lists in `splits/ffpp/`
  (3,600 / 700 / 700 videos). Each list entry is a pair of ids, e.g.
  `["000", "003"]`, and puts `000.mp4`, `003.mp4`, `000_003.mp4` and
  `003_000.mp4` (in all four methods) into the same split.
- **Celeb-DF:** the official 518-video list is the test split. The remaining
  videos go to train or val: 20% of the celebrities and, chosen separately, 20%
  of the YouTube clips go to val (`--celebdf-val-frac`, seed 0), everything else
  to train. Celeb-DF test
  videos share people with train, so treat Celeb-DF test results as in-domain.
- **DFDC:** every video is **test**. Models train and choose their best epoch
  on FF++ and Celeb-DF only, so DFDC is the unseen dataset. Labels come from
  `metadata.json` files (`"label": "FAKE"`/`"REAL"`, and `"original"`: the
  real video a fake was made from, kept in `source_ids`) or a `labels.csv`
  (`filename,label`, 1 = fake). Videos with no label are left out, with a warning.

**Audio-only DFDC fakes.** Some DFDC fakes change only the audio; their frames
are their original's, so a model that sees frames cannot detect them. After
`build_index`, run on the GPU machine (CPU only, a few minutes):

```
python -m src.preprocessing.dfdc_audio_only --data-root ~/data
```

It compares each fake with its `original` at full resolution and flags it when
no patch of the frame changes beyond re-encoding noise (largest changed patch
under 0.0001 of the frame). Flags are written to `processed/dfdc_audio_only.csv`.
Only fakes whose original is in the index can be checked: in the Kaggle sample,
58 of 323. 5 of them are flagged, all made from one original (`atvmxvwyns`), with
a clear gap to the rest; the other 265 fakes are kept unchecked. Flagged fakes are
still scored, but every metric leaves them out (`src/exclusions.py`), and each
table also has a `dfdc_unfiltered` row with them. After the list changes,
`python -m src.frames.rescore` recomputes the frame runs' metrics from the saved
scores (no GPU); rerun `src.temporal.evaluate` and `src.temporal.decision` for
the temporal ones.

| | train | val | test |
|---|---|---|---|
| FaceForensics++ | ✓ | ✓ | ✓ in-domain |
| Celeb-DF | ✓ | ✓ | ✓ in-domain |
| DFDC | | | ✓ unseen dataset |

**Dropped videos.** A Celeb-DF fake shows two people. If one of them is in
train and the other in val, the fake would put a person on both sides whichever
split it went to, so it is dropped. With 20% of people in val (seed 0) this
removes 1,824 of the 5,299 non-test fakes (34%), leaving 3,178 for train and
297 for val. Real videos are never dropped. `build_index` prints the counts.

Both datasets have far more fake than real videos (FF++ 4:1, Celeb-DF about
6:1), so training should use class weights or balanced sampling.

### Video index

Every later stage reads its list of videos from one index file, with one row
per video. Build it on Ronin from the repo root:

```
python -m src.preprocessing.build_index --data-root ~/data   # writes ~/data/processed/index.csv
```

| Column | Type | Example values | Meaning |
|---|---|---|---|
| `video_id` | string | `ffpp/real/000`, `ffpp/Deepfakes/000_003`, `celebdf/YouTube-real/00170` | Unique key, `dataset/method/name` |
| `dataset` | string | `ffpp`, `celebdf` | Source dataset |
| `method` | string | `real`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `Celeb-real`, `YouTube-real`, `Celeb-synthesis` | How the video was made (Celeb-DF: folder name) |
| `label` | int | `0`, `1` | 0 = real, 1 = fake |
| `split` | string | `train`, `val`, `test` | Celeb-DF: official list is `test`, the rest is split by person into `train`/`val`   |
| `source_ids` | string | `000`, `000_003`, `id0_id16`, `yt00170` | People in the video, used to check that no identity crosses splits |
| `path` | string | `/home/<user>/data/raw/ffpp/.../000.mp4` | Full path to the video file (`~` expanded) |

Read it with `pd.read_csv(path, dtype={"source_ids": str})` so IDs such as
`000` keep their leading zeros.

**The index we use is trimmed.** To save GPU time on face extraction, the
Celeb-DF training fakes were cut to a random 1,000 of the 3,178; nothing else is
reduced (training balances real and fake anyway). On Ronin:

| File | Videos |
|---|---|
| `processed/index.csv` | 7,927 (5,166 train / 1,143 val / 1,618 test). The list every stage reads; it matches the extracted faces |
| `processed/index_variants/index_full.csv` | 10,105: the untrimmed list, as `build_index` writes it. Not used |
| `processed/index_variants/index_{train,val,test}.csv` | Parts of `index.csv`, for extracting faces one split at a time |

`build_index` always writes the untrimmed list, so on Ronin run it only with
`--out` pointing somewhere else: written over `processed/index.csv`, it no
longer matches `faces/`. The code that drew the 1,000 is not in the repo; the
trimmed `index.csv` itself is the record of which videos were used.

### Face crops

Deepfakes only change the face, and the backbones take 224–380 px inputs, so
every model trains on face crops rather than whole frames. Extract them once on
the GPU machine after building the index (in tmux):

```
python -m src.preprocessing.extract_faces --data-root ~/data --limit 20   # quick test, prints s/video
python -m src.preprocessing.extract_faces --data-root ~/data              # everything
```

Ours was extracted one split at a time, with 2 windows per video for train and
6 for val and test, then merged:

```
P=~/data/processed/index_variants
python -m src.preprocessing.extract_faces --data-root ~/data --workers 2 --index $P/index_train.csv --windows 2
python -m src.preprocessing.extract_faces --data-root ~/data --workers 2 --index $P/index_val.csv --windows 6
python -m src.preprocessing.extract_faces --data-root ~/data --workers 2 --index $P/index_test.csv --windows 6
python -m src.preprocessing.extract_faces --data-root ~/data --workers 2   # merges all into processed/faces.csv
```

- **Frames:** `--windows` runs of 32 consecutive frames (`--window-len`),
  spread evenly: one in the middle of each of `--windows` equal parts of the
  video (default 2). The temporal models use the windows; the frame models use
  the same crops. More windows give steadier video-level results; fewer save
  extraction time on the large train split. Videos too short for every window
  get fewer.
- **Faces:** detected with RetinaFace (ResNet-50, via `batch-face`), a whole
  window at a time, on frames shrunk to 960 px; crops come from the full frame.
  Each window starts with the largest face and follows it frame to frame, so a
  window stays on one person; different windows may pick different people.
- **Landmarks:** RetinaFace's 5 points (eyes, nose, mouth corners) go in the
  CSV. The standard 68 points (FAN, via `face-alignment`; needed for
  Self-Blended Images masks) go in `landmarks68.npy` next to the crops.
- **Crops:** square, 1.3× the face box (`--margin 0.3`), resized to 380 px
  (`--size`), JPEG quality 95. Parts outside the frame are black. Resize per
  backbone when loading (Xception 299, EfficientNet-B4 380, ViT/CLIP 224).
- **Resumable:** each video's `faces.csv` is written last; videos that have one
  are skipped, so the job can be stopped with Ctrl+C and rerun. Videos that
  fail are reported and retried on the next run.
- **GPU:** `--workers` processes (default 4) decode videos on the CPU, and each
  loads its own copy of the models on the GPU. Lower it if the GPU runs out of
  memory; raise it if the GPU sits idle and there are spare CPUs. On the
  project's GPU machine use `--workers 2`: 4 ran out of GPU memory.
  `--device cpu` works for testing but is slow (~40 s/video).

Output in `~/data/faces/<video_id>/`: one `<frame>.jpg` per frame with a face,
`faces.csv`, and `landmarks68.npy`. At the end of every run, the rows are also
merged into `~/data/processed/faces.csv`, one row per extracted frame. That file
is **rewritten from the videos in the index the run was given**, so after a run
on one split it holds only that split. Rerun with the full `index.csv` to merge
everything: finished videos are skipped, so nothing is extracted again.

| Column | Meaning |
|---|---|
| `video_id`, `dataset`, `method`, `label`, `split` | As in the index |
| `window`, `frame` | Window number in the video (0–1 in train, 0–5 in val and test) and frame number in the video |
| `crop` | Crop path relative to `~/data/faces/`; empty if no face was found in that frame |
| `score` | RetinaFace face confidence |
| `crop_x`, `crop_y`, `crop_side` | Where the crop sits in the original frame (pixels) |
| `right_eye_x` … `mouth_left_y` | 5 face landmarks in crop pixels (the person's right eye first) |
| `lm68_score` | Mean confidence of the 68 landmarks (0–1) |

`landmarks68.npy` has shape `(rows, 68, 2)`: the 68 points (iBUG order: jaw
0–16, brows 17–26, nose 27–35, eyes 36–47, mouth 48–67) in crop pixels, one
entry per row of that video's `faces.csv`, in the same order; NaN where no face
was found.


## Frame models

Code in `src/frames/`. Each backbone is fine-tuned on face crops to output a
fake probability per crop and a feature vector for the temporal models.

| `--backbone` | timm weights | Input | Features |
|---|---|---|---|
| `xception` | `legacy_xception.tf_in1k` | 299 | 2048 |
| `efficientnet_b4` | `tf_efficientnet_b4.ap_in1k` | 380 | 1792 |
| `vit_b16` | `vit_base_patch16_224.augreg_in21k` | 224 | 768 |
| `clip_vit_b16` | `vit_base_patch16_clip_224.openai` | 224 | 768 |

- **Quality filter:** crops with a RetinaFace score below 0.9 or a crop side
  under 64 px in the original frame are left out (`--min-score`, `--min-side`).
- **Balanced sampling:** each epoch draws real and fake equally often, and
  every video equally often within its class.
- **Call quality:** half the training crops (`--degrade-prob`) get random
  lighting, noise, blur, downscaling and strong JPEG compression
  (`src/frames/degrade.py`). `predict --call-quality` applies the same to the
  test crops, identically for every model, with the settings drawn once per
  window so they hold steady over its 32 frames as in a real call; only the
  sensor noise changes from frame to frame.
- **Self-Blended Images** (`--sbi only` or `--sbi mix`): real crops blended with
  an altered copy of themselves inside a soft face mask, labelled fake
  (`src/frames/sbi.py`). `only` trains on real crops and SBI fakes alone, as in
  the paper.
- **Which weights train** (`--train-mode`): `full`, `linear` (head only), or
  `norm` (head and normalisation layers; the light CLIP fine-tune).
- **Unseen methods:** `--exclude-methods NeuralTextures` leaves a method out of
  train and val; `predict` still scores it.

### Input setting: mask vs no-mask

Every model is trained in one of two input settings, chosen with `--mask`:

| Setting | Flag | The model sees |
|---|---|---|
| `no-mask` | (none) | the plain face crop, with hair, neck and background |
| `mask` | `--mask` | only the face: outside the 68-landmark outline (extended over the forehead, plus a small border) is black |

All masking code is in `src/frames/mask.py`, applied in one marked step of
`FaceFrames.__getitem__` (`src/frames/dataset.py`). The setting is part of the
run name and folder, and `predict` reads it from the run, so a model is always
tested the way it was trained. Preview the mask on a few crops:
`python -m src.frames.mask --data-root ~/data` (writes `~/data/previews/mask/`).

### Running

Baselines (the four backbones, each with and without the mask; 8 runs), in
tmux on the GPU machine:

```
bash scripts/run_frame_baselines.sh --epochs 1 --samples-per-epoch 256 --val-frames 1  # quick test
bash scripts/run_frame_baselines.sh
```

Each run trains, then scores the test split clean and at call quality. The
script is safe to rerun after the machine stops: finished runs are skipped and
unfinished ones continue. It ends by printing the comparison table
(`python -m src.frames.compare`, also saved as `~/data/runs/frames/comparison.csv`).

Single runs:

```
python -m src.frames.train --backbone efficientnet_b4            # no-mask
python -m src.frames.train --backbone efficientnet_b4 --mask     # mask
python -m src.frames.train --backbone efficientnet_b4 --sbi only
```

Runs are saved by input setting:

```
~/data/runs/frames/
├── no-mask/<run name>/      best.pt, last.pt, metrics.csv, predictions/
├── mask/<run name>/
└── comparison.csv
```

If the machine stops mid-run, rerun the same command with `--resume`. To score
again or score other splits, use `python -m src.frames.predict --run <run folder>`.
To recompute the metrics from the saved scores only (no GPU, e.g. on a laptop),
use `python -m src.frames.rescore`.
Features for the temporal models are saved with `scripts/export_features.sh`
(see [Temporal models](#temporal-models)).


## Temporal models

Code in `src/temporal/`. A temporal model turns the 32 frames of a window into
one window score. It is built on one trained frame run, which stays fixed: the
frame model scores every crop and exports its feature vector once, and the
aggregators work from those.

| Aggregator | Code | Window score |
|---|---|---|
| `mean` | `src.temporal.pooling --how mean` | average of the window's frame probabilities |
| `topk-5` | `src.temporal.pooling --how topk --k 5` | average of its 5 highest frame probabilities |
| `lstm` | `src.temporal.train --model lstm` | bidirectional LSTM (1 layer, 256 units) over the frame features |
| `transformer` | `src.temporal.train --model transformer` | Transformer encoder (2 layers, width 256, 4 heads) with a classification token |

Windows with fewer than 8 usable crops (after the frame model's quality filter)
are left out for every aggregator (`--min-frames`).

### Running

In tmux on the GPU machine, from the repo root:

```
bash scripts/export_features.sh ~/data/runs/frames/no-mask/<run name>   # once per frame run
bash scripts/run_temporal.sh ~/data/runs/frames/no-mask/<run name>      # pooling, LSTM, Transformer, evaluation, decision rule, comparison
bash scripts/run_temporal.sh <frame run> --epochs 1 --samples-per-epoch 256   # quick test
```

`export_features.sh` runs `src.frames.predict --features` on val, test and
train, each clean and at call quality, writing `<name>.csv` and
`<name>_features.npy` (e.g. `train_call_features.npy`) into the frame run's
`predictions/`.

- **Training (LSTM, Transformer):** real and fake windows are drawn equally
  often, and every video equally often within its class. Each window is either
  all clean or all call quality, at random (`--train-versions`). Up to 30
  epochs of 20,000 windows, AdamW at 3e-4 with warm-up and cosine decay. Keeps
  the epoch with the best mean val video AUC (clean and call quality) and stops
  after 6 epochs without improvement (`--patience`). Takes about a minute per
  model.
- **Evaluation** (`src.temporal.evaluate`): a video's score is the mean of its
  window scores (`--video-how max`: the highest). Window and video metrics per
  dataset, and video AUC per fake method.
- **Decision rule** (`src.temporal.decision`): flag a video once N consecutive
  windows score at or above a threshold. Chosen on val: for each N up to 4, the
  lowest threshold that keeps the false-positive rate at or below 1%
  (`--max-fpr`) on clean, call quality and both pooled; then the N that catches
  the most fakes. Test is scored once with that rule. The windows are spread
  through the video with gaps, so "consecutive" means consecutive windows, not
  continuous time.
- **Comparison** (`src.temporal.compare`): every aggregator's test results in
  `runs/temporal/comparison.csv`. These are for reporting; choose the
  aggregator and the rule on val.

Output, next to the frame run it is built on:

```
~/data/runs/temporal/
├── comparison.csv
└── no-mask/<frame run name>/
    ├── mean/, topk-5/                      predictions/, decision.json, decision_test.csv
    └── lstm/, transformer/                 also args.json, metrics.csv, best.pt, last.pt, timing.json
```

| File | Contents |
|---|---|
| `predictions/<name>_windows.csv` | One row per window: `video_id`, `window`, `dataset`, `method`, `label`, `score` (fake probability); `<name>` is `val`, `val_call`, `test` or `test_call` |
| `predictions/<name>_metrics.json` | Window and video metrics per dataset, video AUC per fake method |
| `decision.json` | The rule chosen on val (N, threshold) and its val FPR and TPR |
| `decision_test.csv` | The rule on test: FPR, TPR and median windows until a fake is flagged, per dataset |
| `metrics.csv` | Train loss and val video AUC per epoch |
| `timing.json` | Milliseconds per window for the temporal model alone |

**Rerunning with new inputs.** Both scripts skip work they think is finished.
`export_features.sh` skips an export whose `.csv` is newer than its `.npy`, and
`run_temporal.sh` resumes training (`--resume`), so a run that already stopped
early stops again at once and only rescores. To redo them after the frame
model's inputs change (for example the call-quality degradation), first delete
the old exports in the frame run's `predictions/` (`*_call.csv`,
`*_call_features.npy`, and the clean ones if those changed too) and the frame
run's folder under `runs/temporal/`.


## Analysis

Training curves and error analysis for the frame models, in `src/analysis/frames.py`.
It only reads each run's `metrics.csv` and `predictions/`, so it needs no GPU:

```
python -m src.analysis.frames                     # on the GPU machine; writes ~/data/analysis/frames/
python -m src.analysis.frames --data-root <folder> --out <folder>/analysis   # elsewhere
```

To run it on a laptop, copy `~/data/runs/frames/` into `<folder>/runs/frames/`
without the weights and features (`best.pt`, `last.pt`, `*.npy`).

| Output | Shows |
|---|---|
| `1_training_curves.png` | Train loss and val video AUC per epoch; the kept checkpoint is marked |
| `2_vit_score_distributions.png`, `3_dfdc_score_distributions_all_models.png` | Video scores of real and fake videos, per test dataset |
| `thresholds.csv` | Accuracy, TPR and FPR at 0.5, at the threshold chosen on val as in the decision rule (lowest with val FPR <= 1%; only runs scored with `predict --split val`), and at each test set's own best threshold |
| `4_auc_per_method.png`, `auc_per_method.csv` | Video AUC per fake method (clean test) |
| `5_clean_vs_call_quality.png`, `clean_vs_call.csv` | Video AUC on the clean and call-quality test sets |
| `dfdc_hard_videos.csv` | Each DFDC video's score per model, and how many models get it wrong at 0.5 |

The runs it reads are listed in `RUNS` at the top of the script; add new runs there.

### Backbone gaps and seeds

Whether the differences between the backbones are larger than chance, in
`src/analysis/backbones.py`. No GPU needed:

```
python -m src.analysis.backbones     # writes ~/data/analysis/backbones/
```

For each test set (FF++, Celeb-DF, DFDC, NeuralTextures seen and held out),
clean and at call quality, it gives each backbone's video AUC with a 95%
bootstrap interval, and the gap between every pair with a 95% paired bootstrap
interval (videos resampled, real and fake separately; a gap whose interval
excludes 0 is larger than the choice of test videos explains). It reads the
held-out runs (`<run>_no-NeuralTextures`) and any extra seeds (`<run>_seed<n>`)
next to each run in `RUNS`; with seeds, a backbone's AUC is the mean over them
and the spread between seeds is reported too.

Extra seeds, in tmux on the GPU machine (safe to rerun, like the baselines):

```
bash scripts/run_seeds.sh                                    # seeds 1 and 2, all four backbones
BACKBONES="vit_b16 clip_vit_b16" bash scripts/run_seeds.sh   # only these
HELD_OUT=1 bash scripts/run_seeds.sh                         # the NeuralTextures held-out runs
```

| Output | Contents |
|---|---|
| `backbone_auc.csv` | Video AUC per backbone and test set: per seed, mean, seed sd, 95% interval |
| `backbone_pairs.csv` | AUC difference of every pair of backbones, 95% paired interval, `excludes_0` |

### Grad-CAM

Heatmaps of which parts of a face crop drove each frame model's decision, in
`src/analysis/gradcam.py`. It needs the runs' `best.pt` and the face crops, so
it runs on the GPU machine:

```
python -m src.analysis.gradcam --per-dataset 4   # quick test
python -m src.analysis.gradcam                   # writes ~/data/analysis/gradcam/
```

Example crops come from the test predictions, one frame per video: for each FF++
method and Celeb-DF, a fake and a real that every model gets right; the DFDC
fakes every model misses; and DFDC videos every model gets right. Each crop is
preprocessed exactly as in training (masked for mask runs).

| Output | Shows |
|---|---|
| `1_in_domain.png`, `2_dfdc_missed.png`, `3_dfdc_correct.png` | Each crop, then each model's map |
| `4_sanity_random_weights.png` | Trained models next to randomly initialised ones; maps that look alike would mean Grad-CAM shows edges, not what the model learned |
| `face_share.csv` | Share of each map inside the face outline, next to the share of the crop the face covers |
| `examples.csv` | The chosen crops, why they were chosen, and every model's scores |
