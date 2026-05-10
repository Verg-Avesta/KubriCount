#!/bin/bash
# run_l1.sh
# CPU rendering script supporting Levels 1-4.
# Level 1: single type
# Level 2-4: dual type
# Level 4 update: each category uses its own hyperparameter configuration.

OUTPUT_DIR="$(pwd)/KubriCount/train/level_1/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

NUM_SCENES=${1:-1}
LEVEL=${2:-1}
SPLIT=${3:-"train"}
LEVEL2_MODE=${4:-"random"}
CONFIG_FILE=${5:-""}

echo "=========================================="
echo "Counting scene generation (CPU)"
echo "=========================================="
echo "Number of scenes: $NUM_SCENES"
echo "Level: $LEVEL"
echo "Split: $SPLIT"
echo "Level2 Mode: $LEVEL2_MODE"
echo "Config File: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo "Level description:"
echo "  Level 1: one category, one asset, identical instances"
echo "  Level 2: same asset, different size or color (dual type)"
echo "  Level 3: same category, different assets (dual type)"
echo "  Level 4: different categories within the same super category"
echo "          each category uses its own size/count configuration"
echo "          camera parameters are randomly selected from the two categories"
echo "=========================================="

# Config file argument
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    CONFIG_ARG="--config_file=$CONFIG_FILE"
else
    CONFIG_ARG=""
fi

FAILED_SCENES=()
SLOW_SCENES=()
SUCCESS_COUNT=0

for i in $(seq 1 $NUM_SCENES); do
    SCENE_DIR="$OUTPUT_DIR/scene_$(printf '%04d' $i)"
    mkdir -p "$SCENE_DIR"

    echo ""
    echo "--- Generating scene $i/$NUM_SCENES (Level $LEVEL, Split $SPLIT) ---"

    docker run --rm --interactive \
        --user $(id -u):$(id -g) \
        --workdir /kubric \
        --volume "$(pwd):/kubric" \
        --volume "$SCENE_DIR:/output" \
        kubruntu /usr/bin/python3 render_level.py \
            --job-dir=/output \
            --level=$LEVEL \
            --level2_mode=$LEVEL2_MODE \
            --objects_split=$SPLIT \
            --backgrounds_split=$SPLIT \
            --min_objects_per_group=10 \
            --max_total_objects=250 \
            --object_size_min=0.35 \
            --object_size_max=0.85 \
            --size_variation_min=0.8 \
            --size_variation_max=1.2 \
            --level4_min_assets_per_category=3 \
            --level4_max_assets_per_category=10 \
            --level2_small_ratio_min=0.5 \
            --level2_small_ratio_max=0.9 \
            --level2_large_ratio_min=1.1 \
            --level2_large_ratio_max=1.5 \
            --level2_min_hue_diff=0.3 \
            --density_factor=1.15 \
            --min_distance_ratio=0.85 \
            --placement_attempts=50 \
            --placement_speed_check_count=20 \
            --placement_speed_threshold=0.2 \
            --camera_offset_ratio=0.4 \
            --camera_distance_min=2.5 \
            --camera_distance_max=18.0 \
            --camera_height_min=1.0 \
            --camera_height_max=15.0 \
            --camera_angle_min=10.0 \
            --camera_angle_max=75.0 \
            --focal_length_min=20.0 \
            --focal_length_max=65.0 \
            --coverage_min=0.2 \
            --coverage_max=0.75 \
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
    level = d['level']
    is_single = d.get('is_single_type', False)
    g1 = d['groups'][0]
    g2 = d['groups'][1] if len(d['groups']) > 1 else None
    c1, c2 = d['counting_info']['group1_visible'], d['counting_info']['group2_visible']
    total = d['counting_info']['total_visible']
    
    if level == 1:
        print(f'  Level 1 (Single): {g1[\"category\"]}({c1}), total={total}')
    elif level == 2:
        mode = d.get('level_specific_info', {}).get('mode', 'unknown')
        print(f'  Level 2 ({mode}): group1({c1}) vs group2({c2}), total={total}')
    elif level == 3:
        print(f'  Level 3: {g1[\"category\"]} - asset1({c1}) vs asset2({c2}), total={total}')
    else:  # Level 4
        super_cat = d.get('level_specific_info', {}).get('super_category', 'unknown')
        cat1 = g1['category']
        cat2 = g2['category'] if g2 else 'N/A'
        size1 = d.get('level_specific_info', {}).get('category_1_size', 0)
        size2 = d.get('level_specific_info', {}).get('category_2_size', 0)
        cam_src = d.get('level_specific_info', {}).get('camera_config_source', 'unknown')
        print(f'  Level 4 ({super_cat}):')
        print(f'    {cat1}(size={size1:.2f}, count={c1}) vs {cat2}(size={size2:.2f}, count={c2})')
        print(f'    Camera config from: {cam_src}, total={total}')
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
echo "Level: $LEVEL, Split: $SPLIT"
if [ ${#SLOW_SCENES[@]} -gt 0 ]; then
    echo "Slow placement: ${SLOW_SCENES[*]}"
fi
if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
    echo "Other failures: ${FAILED_SCENES[*]}"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Usage examples:
# bash run_l1.sh 10 1 train                          # Level 1
# bash run_l1.sh 10 2 train size                     # Level 2 (size)
# bash run_l1.sh 10 2 train color                    # Level 2 (color)
# bash run_l1.sh 10 3 train                          # Level 3
# bash run_l1.sh 10 4 train                          # Level 4 (per-category configuration)

# Use a custom config file:
# bash run_l1.sh 10 4 train random config.json
