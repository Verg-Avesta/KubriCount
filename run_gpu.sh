#!/bin/bash
# run_l1_gpu.sh
# GPU rendering script supporting Levels 1-5 + objects_split(train/testA/testB).
#
# Levels:
#   1: single category, single asset (single-type)
#   2: same asset, different size/color (dual-type)
#   3: two categories, one asset each, variable size (dual-type)
#   4: same category, two different assets, variable size (dual-type)
#   5: two categories within same super category, multi-asset (dual-type)
#
# Splits:
#   train : train categories, excluding per-category testA heldout assets
#   testA : novel asset OOD within train categories (heldout 1/11, deterministic)
#   testB : novel category OOD (SUPER_CATEGORIES[*]['test'])

OUTPUT_BASE="$(pwd)/KubriCount"
NUM_SCENES=${1:-1}
GPU_IDS=${2:-"all"}
LEVEL=${3:-1}
OBJ_SPLIT=${4:-"train"}     # train / testA / testB
LEVEL2_MODE=${5:-"random"}  # size / color / random
CONFIG_FILE=${6:-""}

DATE_TAG=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${OBJ_SPLIT}/level${LEVEL}/${DATE_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Counting scene generation (GPU)"
echo "=========================================="
echo "Number of scenes: $NUM_SCENES"
echo "GPU: $GPU_IDS"
echo "Level: $LEVEL"
echo "Objects Split: $OBJ_SPLIT"
echo "Level2 Mode: $LEVEL2_MODE"
echo "Config File: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

if [ "$GPU_IDS" = "all" ]; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG="--gpus \"device=$GPU_IDS\""
fi

# Config file argument
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    CONFIG_ARG="--config_file=$CONFIG_FILE"
else
    CONFIG_ARG=""
fi

# Background split:
# Force testA/testB to use the test background split.
# The editing stage removes the background, so this is fixed to test here.
if [ "$OBJ_SPLIT" = "train" ]; then
    BG_SPLIT="train"
else
    BG_SPLIT="test"
fi

FAILED_SCENES=()
SLOW_SCENES=()
SUCCESS_COUNT=0

for i in $(seq 1 $NUM_SCENES); do
    SCENE_DIR="$OUTPUT_DIR/scene_$(printf '%04d' $i)"
    mkdir -p "$SCENE_DIR"

    echo ""
    echo "--- Generating scene $i/$NUM_SCENES (Level $LEVEL, objects_split=$OBJ_SPLIT, bg_split=$BG_SPLIT) ---"

    eval docker run --rm --interactive \
        $GPU_FLAG \
        --user $(id -u):$(id -g) \
        --workdir /kubric \
        --volume "$(pwd):/kubric" \
        --volume "$SCENE_DIR:/output" \
        kubricdockerhub/kubruntu-gpu /usr/bin/python3 render_level.py \
            --job-dir=/output \
            --level=$LEVEL \
            --level2_mode=$LEVEL2_MODE \
            --objects_split=$OBJ_SPLIT \
            --backgrounds_split=$BG_SPLIT \
            --testA_fraction=0.0909090909 \
            --testA_split_seed=42 \
            --low_count_threshold=150 \
            --low_count_weight=0.5 \
            --level4_min_count_for_train=51 \
            --min_objects_per_group_hard=10 \
            --level5_min_assets_per_category=2 \
            --level5_max_assets_per_category=4 \
            --level5_max_total_objects=250 \
            --level5_max_size_ratio=3.0 \
            --density_factor=1.15 \
            --min_distance_ratio=0.85 \
            --placement_attempts=50 \
            --placement_speed_check_count=20 \
            --placement_speed_threshold=0.1 \
            --camera_offset_ratio=0.4 \
            --timeout_minutes=30 \
            --kubasic_assets="assets/KuBasic.json" \
            --hdri_assets="assets/HDRI_haven.json" \
            --hdri_t2l_assets="assets/HDRI_t2l.json" \
            --shapenet_assets="assets/ShapeNetCore.v2.json" \
            --trellis_assets="assets/trellis/metadata.json" \
            --resolution=1024 \
            $CONFIG_ARG

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Scene $i generated successfully."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        if [ -f "$SCENE_DIR/metadata.json" ]; then
            python3 -c "
import json
try:
    d = json.load(open('$SCENE_DIR/metadata.json'))
    level = d.get('level')
    split = d.get('split_info', {}).get('objects_split', 'unknown')
    g = d.get('groups', [])
    cinfo = d.get('counting_info', {})
    total = cinfo.get('total_visible', None)
    if level == 1:
        print(f'  L1/{split}: {g[0][\"category\"]} total={total}')
    elif level == 2:
        mode = d.get('level_specific_info', {}).get('mode', 'unknown')
        print(f'  L2/{split} ({mode}): {g[0][\"category\"]} total={total}')
        if mode == 'color':
            print(f'    group1_color={d.get(\"level_specific_info\", {}).get(\"group1_color\")}')
            print(f'    group2_color={d.get(\"level_specific_info\", {}).get(\"group2_color\")}')
    elif level == 3:
        print(f'  L3/{split}: {g[0][\"category\"]} vs {g[1][\"category\"]} total={total}')
    elif level == 4:
        print(f'  L4/{split}: {g[0][\"category\"]} (two assets) total={total}')
    else:
        sc = d.get('level_specific_info', {}).get('super_category', 'unknown')
        r0 = d.get('level_specific_info', {}).get('size_ratio_before', None)
        r1 = d.get('level_specific_info', {}).get('size_ratio_after', None)
        print(f'  L5/{split} ({sc}): {g[0][\"category\"]} vs {g[1][\"category\"]} total={total} ratio={r0}->{r1}')
except Exception as e:
    print(f'  Failed to read metadata: {e}')
" 2>/dev/null || true
        fi
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "Scene $i timed out and was skipped."
        FAILED_SCENES+=("$i:timeout")
        rm -rf "$SCENE_DIR"
    elif [ $EXIT_CODE -eq 125 ]; then
        echo "Scene $i placement was too slow and was skipped."
        SLOW_SCENES+=("$i:slow_placement")
        rm -rf "$SCENE_DIR"
    else
        echo "Scene $i generation failed. (exit code: $EXIT_CODE)"
        FAILED_SCENES+=("$i:error")
    fi
done

echo ""
echo "=========================================="
echo "Done. Success: $SUCCESS_COUNT/$NUM_SCENES"
echo "Level: $LEVEL, Objects Split: $OBJ_SPLIT, BG Split: $BG_SPLIT"
if [ ${#SLOW_SCENES[@]} -gt 0 ]; then
    echo "Slow placement: ${SLOW_SCENES[*]}"
fi
if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
    echo "Other failures: ${FAILED_SCENES[*]}"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Usage examples:
# bash run_l1_gpu.sh 5 0 1 train random config_dense.json
# bash run_l1_gpu.sh 2 0 1 train random config_gpt.json
# bash run_l1_gpu.sh 2 1 2 train color config_gpt.json

# bash run_l1_gpu.sh 100 0 1 train random config_gpt.json
# bash run_l1_gpu.sh 100 0 2 train random config_gpt.json
# bash run_l1_gpu.sh 100 0 3 train random config_gpt.json
# bash run_l1_gpu.sh 100 0 4 train random config_gpt.json
# bash run_l1_gpu.sh 100 1 5 train random config_gpt.json

# bash run_l1_gpu.sh 50 1 1 testA random config_gpt.json
# bash run_l1_gpu.sh 50 1 2 testA random config_gpt.json
# bash run_l1_gpu.sh 50 1 3 testA random config_gpt.json
# bash run_l1_gpu.sh 50 2 4 testA random config_gpt.json
# bash run_l1_gpu.sh 50 2 5 testA random config_gpt.json

# bash run_l1_gpu.sh 50 2 1 testB random config_gpt.json
# bash run_l1_gpu.sh 50 2 2 testB random config_gpt.json
# bash run_l1_gpu.sh 50 3 3 testB random config_gpt.json
# bash run_l1_gpu.sh 50 3 4 testB random config_gpt.json
# bash run_l1_gpu.sh 50 3 5 testB random config_gpt.json

# ps aux | grep "run_l1_gpu.sh"
# pkill -f "run_l1_gpu.sh"

# tmux a -t 31
# tmux a -t 32
# tmux a -t 33
# tmux a -t 34

# docker load -i docker-image/kubruntu-gpu.tar
