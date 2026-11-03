#!/usr/bin/env bash
# Retrains no-mask baselines with other seeds, to measure how much a result moves
# between training runs. Each run is the original command plus --seed and a run
# name ending in _seed<n>, which src.analysis.backbones picks up.
#
# Run on the GPU machine from the repo root, inside tmux:
#     bash scripts/run_seeds.sh                                  # seeds 1 and 2 of all four backbones
#     BACKBONES="vit_b16 clip_vit_b16" bash scripts/run_seeds.sh # only these
#     HELD_OUT=1 bash scripts/run_seeds.sh                       # the NeuralTextures held-out runs instead
# Extra options are passed to every run, e.g. a quick test:
#     SEEDS=1 bash scripts/run_seeds.sh --data-root ~/test-data --epochs 1 --samples-per-epoch 64
#
# Safe to rerun after the machine stops: finished runs are skipped, and an
# unfinished one continues from its last epoch (--resume).
set -euo pipefail

SEEDS=${SEEDS:-"1 2"}
BACKBONES=${BACKBONES:-"vit_b16 clip_vit_b16 efficientnet_b4 xception"}
declare -A MODE=([xception]=full [efficientnet_b4]=full [vit_b16]=full [clip_vit_b16]=norm)

extra=()
suffix=""
if [[ "${HELD_OUT:-0}" == 1 ]]; then
    extra=(--exclude-methods NeuralTextures)
    suffix="_no-NeuralTextures"
fi

for seed in $SEEDS; do
    for backbone in $BACKBONES; do
        mode=${MODE[$backbone]}
        name="${backbone}_${mode}_no-mask_ffpp+celebdf_sbi-none${suffix}_seed${seed}"
        echo "=== $name ==="
        python -m src.frames.train --backbone "$backbone" --train-mode "$mode" --seed "$seed" \
            --run-name "$name" "${extra[@]}" --resume "$@"
    done
done
