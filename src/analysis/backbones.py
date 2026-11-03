"""
Are the gaps between the frame backbones real? Paired bootstrap over the test videos. No GPU needed.

Reads predictions/test.csv and test_call.csv of each backbone's run in RUNS, its NeuralTextures
held-out run (<run>_no-NeuralTextures) and any extra seeds (<run>_seed<n>,
<run>_no-NeuralTextures_seed<n>) from {data_root}/runs/frames/. From the repo root:
    python -m src.analysis.backbones
On a laptop, after copying runs/frames/ (without best.pt, last.pt, *.npy) into <folder>/runs/frames/:
    python -m src.analysis.backbones --data-root <folder> --out <folder>/analysis/backbones
Extra seeds are the original training command plus --seed and a --run-name ending in _seed<n>:
    python -m src.frames.train --backbone vit_b16 --seed 1 --run-name vit_b16_full_no-mask_ffpp+celebdf_sbi-none_seed1

Writes to {out} (default {data_root}/analysis/backbones), for each test set (FF++, Celeb-DF, DFDC,
NeuralTextures seen in training and held out), clean and at call quality:
    backbone_auc.csv    each backbone's video AUC per seed, their mean, the spread between seeds (sd)
                        and a 95% bootstrap interval of the mean
    backbone_pairs.csv  the AUC difference of every pair of backbones with a 95% paired bootstrap interval;
                        excludes_0 = the gap is larger than the choice of test videos explains
Videos are resampled with real and fake separately, so the class sizes stay fixed, and every run is
scored on the same resampled videos, so most of the noise cancels in the differences. The intervals
cover which videos happen to be in the test set, not retraining; the seed spread shows that, once
there are extra seeds. NeuralTextures rows are the FF++ test reals against the NeuralTextures fakes.
A video's score is the mean of its frame probabilities, as in src.metrics. Videos in src.exclusions
(DFDC audio-only fakes) are left out.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.exclusions import drop_excluded, excluded_videos
from src.metrics import video_scores

# display name -> seed-0 run folder under runs/frames, trained on every fake method
RUNS = {
    "ViT-B/16": "no-mask/vit_b16_full_no-mask_ffpp+celebdf_sbi-none",
    "CLIP ViT-B/16": "no-mask/clip_vit_b16_norm_no-mask_ffpp+celebdf_sbi-none",
    "EfficientNet-B4": "no-mask/efficientnet_b4_full_no-mask_ffpp+celebdf_sbi-none",
    "Xception": "no-mask/xception_full_no-mask_ffpp+celebdf_sbi-none",
}
HELD_OUT = "_no-NeuralTextures"  # suffix train.py gives a run with --exclude-methods NeuralTextures
DATASETS = {"ffpp": "FF++", "celebdf": "Celeb-DF", "dfdc": "DFDC"}
CONDITIONS = {"test": "clean", "test_call": "call quality"}


def seed_runs(runs_dir: Path, run: str) -> list[Path]:
    """The run and its extra seeds (<run>_seed<n>), those that exist."""
    base = runs_dir / run
    return [p for p in [base, *sorted(base.parent.glob(base.name + "_seed*"))] if p.is_dir()]


def load_videos(run_dir: Path, name: str, excluded: set[str]) -> pd.DataFrame | None:
    """One row per video (video_id, label, dataset, method, prob), or None if the run has no such file."""
    path = run_dir / "predictions" / f"{name}.csv"
    if not path.exists():
        return None
    videos = video_scores(pd.read_csv(path, dtype={"video_id": str})).reset_index()
    return drop_excluded(videos, excluded)


def test_sets(videos: pd.DataFrame, held_out: bool) -> dict[str, pd.DataFrame]:
    """{test set: rows}. Held-out runs count only on NeuralTextures, the method they never saw."""
    ffpp = videos[videos.dataset == "ffpp"]
    nt = ffpp[(ffpp.label == 0) | (ffpp.method == "NeuralTextures")]
    if held_out:
        return {"NeuralTextures (held out)": nt}
    return {**{label: videos[videos.dataset == d] for d, label in DATASETS.items()}, "NeuralTextures (seen)": nt}


def batch_auc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """AUC of each row of scores s against the labels y in the same row, both (rows, videos),
    from ranks (ties count half, as in roc_auc_score)."""
    ranks = rankdata(s, axis=1)
    pos = y == 1
    n_fake = pos.sum(axis=1)
    n_real = y.shape[1] - n_fake
    return ((ranks * pos).sum(axis=1) - n_fake * (n_fake + 1) / 2) / (n_fake * n_real)


def compare(scores: dict[str, np.ndarray], y: np.ndarray, boot: int, rng: np.random.Generator) -> tuple[list, list]:
    """Rows for backbone_auc and backbone_pairs from {backbone: (seeds, videos) scores} and labels y.
    A backbone's AUC is the mean over its seeds, on the full test set and on each resample."""
    real, fake = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    idx = np.concatenate([rng.choice(real, (boot, len(real))), rng.choice(fake, (boot, len(fake)))], axis=1)
    y_boot = y[idx]
    seed_auc = {b: batch_auc(np.broadcast_to(y, s.shape), s) for b, s in scores.items()}
    boot_auc = {b: np.mean([batch_auc(y_boot, seed[idx]) for seed in s], axis=0) for b, s in scores.items()}
    order = sorted(scores, key=lambda b: -seed_auc[b].mean())

    aucs = []
    for b in order:
        lo, hi = np.percentile(boot_auc[b], [2.5, 97.5])
        aucs.append(dict(backbone=b, seeds=len(seed_auc[b]), auc=seed_auc[b].mean(),
                         seed_sd=seed_auc[b].std(ddof=1) if len(seed_auc[b]) > 1 else np.nan,
                         lo=lo, hi=hi, per_seed=" ".join(f"{a:.3f}" for a in seed_auc[b])))
    pairs = []
    for a, b in combinations(order, 2):  # a has the higher AUC, so diff >= 0
        lo, hi = np.percentile(boot_auc[a] - boot_auc[b], [2.5, 97.5])
        pairs.append(dict(better=a, worse=b, diff=seed_auc[a].mean() - seed_auc[b].mean(), lo=lo, hi=hi,
                          excludes_0=bool(lo > 0 or hi < 0)))
    return aucs, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--out", type=Path, default=None, help="default: {data_root}/analysis/backbones")
    parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    runs_dir = data_root / "runs" / "frames"
    out = (args.out or data_root / "analysis" / "backbones").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    excluded = excluded_videos(data_root)
    rng = np.random.default_rng(args.seed)

    auc_rows, pair_rows = [], []
    for held_out in (False, True):
        for name, condition in CONDITIONS.items():
            # {backbone: [video scores of each seed]}
            loaded = {}
            for backbone, run in RUNS.items():
                runs = seed_runs(runs_dir, run + (HELD_OUT if held_out else ""))
                seeds = [v for v in (load_videos(r, name, excluded) for r in runs) if v is not None]
                if seeds:
                    loaded[backbone] = seeds
                else:
                    print(f"note: no {name}.csv for {run}{HELD_OUT if held_out else ''}, left out")
            if len(loaded) < 2:
                continue
            for test_set in test_sets(next(iter(loaded.values()))[0], held_out):
                parts = {b: [test_sets(v, held_out)[test_set].set_index("video_id") for v in seeds]
                         for b, seeds in loaded.items()}
                # the videos every run scored (the quality filter is the same for all, so normally all of them)
                ids = sorted(set.intersection(*(set(p.index) for ps in parts.values() for p in ps)))
                first = next(iter(parts.values()))[0]
                y = first.loc[ids, "label"].to_numpy()
                if len(ids) == 0 or y.min() == y.max():
                    continue
                scores = {b: np.stack([p.loc[ids, "prob"].to_numpy() for p in ps]) for b, ps in parts.items()}
                aucs, pairs = compare(scores, y, args.boot, rng)
                cell = dict(test_set=test_set, condition=condition, real=int((y == 0).sum()), fake=int(y.sum()))
                auc_rows += [{**cell, **r} for r in aucs]
                pair_rows += [{**cell, **r} for r in pairs]

    if not auc_rows:
        raise SystemExit(f"no test predictions found under {runs_dir}")
    auc_table, pair_table = pd.DataFrame(auc_rows), pd.DataFrame(pair_rows)
    auc_table.to_csv(out / "backbone_auc.csv", index=False)
    pair_table.to_csv(out / "backbone_pairs.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nVideo AUC (mean over seeds) with 95% bootstrap interval")
    print(auc_table.assign(auc=auc_table.apply(lambda r: f"{r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1))
          .pivot(index=["test_set", "condition", "real", "fake"], columns="backbone", values="auc")
          [[b for b in RUNS if b in set(auc_table.backbone)]].to_string())
    print("\nAUC(better) - AUC(worse), 95% paired bootstrap interval; * = excludes 0")
    print(pair_table.assign(gap=pair_table.apply(
        lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" + (" *" if r.excludes_0 else ""), axis=1))
          [["test_set", "condition", "better", "worse", "gap"]].to_string(index=False))
    if auc_table.seeds.max() > 1:
        print("\nSeed spread (sd of AUC between seeds)")
        print(auc_table[auc_table.seeds > 1][["test_set", "condition", "backbone", "seeds", "per_seed", "seed_sd"]]
              .to_string(index=False))
    print(f"\nwrote {out}/backbone_auc.csv and backbone_pairs.csv")


if __name__ == "__main__":
    main()
