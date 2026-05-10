#!/bin/bash

# ==============================================================================
# Trellis Assets Parallel Processing Script for Kubric
# Dual-source version: trellis1 + trellis2
# - Merges categories into the same OUTPUT_DIR/ARCHIVE_DIR
# - Adds provenance into each asset's data.json: metadata["source"]="trellis1"/"trellis2"
# bash trellis.sh
# ==============================================================================

set -o pipefail

# ============================================================================
# Directory configuration
# ============================================================================
SOURCE_DIR_1="trellis_assets"
SOURCE_DIR_2="trellis2_assets"

BASE_DIR="."
OUTPUT_DIR="${BASE_DIR}/trellis_output"
ARCHIVE_DIR="${BASE_DIR}/assets/trellis"
SCRIPT_DIR="${BASE_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"

# ============================================================================
# Processing parameters
# ============================================================================
NUM_JOBS=${NUM_JOBS:-12}                # Parallel jobs; override via environment variable.
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-600} # Timeout for each task, in seconds.
PROGRESS_INTERVAL=20                    # Progress reporting interval, in seconds.
DOCKER_IMAGE="kubricdockerhub/shapenet:latest"

# ============================================================================
# Optional: manually specify categories to process.
# Leave empty to process the union of all categories from both sources.
# ============================================================================
CATEGORIES_TO_PROCESS=(
    # Leave empty to auto-scan all categories from both source directories.
)

# ============================================================================
# Initialization
# ============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ARCHIVE_DIR}"
mkdir -p "${LOG_DIR}"

PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
FAILED_FILE="${OUTPUT_DIR}/failed.txt"
TIMEOUT_FILE="${OUTPUT_DIR}/timeout.txt"
STATS_LOG="${OUTPUT_DIR}/processing_stats_$(date +%Y%m%d_%H%M%S).log"

LOCK_DIR="${OUTPUT_DIR}/locks"
mkdir -p "${LOCK_DIR}"

CONTAINER_PREFIX="trellis_$$"

> "${PROGRESS_FILE}"
> "${FAILED_FILE}"
> "${TIMEOUT_FILE}"

# ============================================================================
# Utility functions
# ============================================================================

log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    echo "$message" >> "${STATS_LOG}"
}

cleanup_containers() {
    docker ps -a --filter "name=trellis_" --filter "status=exited" -q 2>/dev/null | xargs -r docker rm 2>/dev/null
    docker ps -a --filter "name=trellis_" --filter "status=dead" -q 2>/dev/null | xargs -r docker rm 2>/dev/null
}

check_docker() {
    if ! docker info &>/dev/null; then
        log "ERROR: Docker daemon is not responding!"
        exit 1
    fi
    log "Docker is healthy"
}

graceful_exit() {
    log ""
    log "Received interrupt signal, cleaning up..."

    touch "${OUTPUT_DIR}/.stop_monitor" 2>/dev/null

    docker ps --filter "name=trellis_" -q 2>/dev/null | xargs -r docker stop -t 5 2>/dev/null
    docker ps --filter "name=trellis_" -q 2>/dev/null | xargs -r docker rm -f 2>/dev/null

    rm -rf "${LOCK_DIR}"/* 2>/dev/null
    log "Cleanup complete. Exiting."
    exit 1
}

trap graceful_exit SIGINT SIGTERM

# ============================================================================
# Core processing function
# ============================================================================

process_glb() {
    local source_dir=$1
    local source_name=$2
    local glb_file=$3

    local ASSET_ID=$(basename "$glb_file" .glb)
    local CATEGORY=$(basename "$(dirname "$glb_file")")

    local LOCK_FILE="${LOCK_DIR}/${source_name}_${CATEGORY}_${ASSET_ID}.lock"

    local WORK_CATEGORY_DIR="${OUTPUT_DIR}/${CATEGORY}"
    local WORK_ASSET_DIR="${WORK_CATEGORY_DIR}/${ASSET_ID}"
    local ARCHIVE_CATEGORY_DIR="${ARCHIVE_DIR}/${CATEGORY}"
    local ARCHIVE_FILE="${ARCHIVE_CATEGORY_DIR}/${ASSET_ID}.tar.gz"
    local LOG_FILE="${LOG_DIR}/${source_name}_${CATEGORY}_${ASSET_ID}.log"

    local TIMESTAMP=$(date +%s%N | md5sum | head -c 8)
    local CONTAINER_NAME="trellis_${source_name}_${CATEGORY}_${ASSET_ID}_${TIMESTAMP}"

    # Skip already processed assets.
    if [ -f "${ARCHIVE_FILE}" ]; then
        return 0
    fi

    # Acquire lock.
    if ! mkdir "${LOCK_FILE}" 2>/dev/null; then
        if [ -d "${LOCK_FILE}" ]; then
            local lock_age=$(( $(date +%s) - $(stat -c %Y "${LOCK_FILE}" 2>/dev/null || echo 0) ))
            if [ $lock_age -gt 600 ]; then
                rmdir "${LOCK_FILE}" 2>/dev/null
                if ! mkdir "${LOCK_FILE}" 2>/dev/null; then
                    return 0
                fi
            else
                return 0
            fi
        else
            return 0
        fi
    fi

    cleanup_on_exit() {
        rmdir "${LOCK_FILE}" 2>/dev/null
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null
    }
    trap cleanup_on_exit EXIT

    echo "[$(date '+%H:%M:%S')] [${source_name}/${CATEGORY}/${ASSET_ID}] Starting..."

    mkdir -p "${WORK_CATEGORY_DIR}"
    mkdir -p "${ARCHIVE_CATEGORY_DIR}"

    local start_time=$(date +%s)

    timeout --signal=TERM --kill-after=60s ${TIMEOUT_SECONDS}s \
      docker run --rm \
      --name "${CONTAINER_NAME}" \
      --user $(id -u):$(id -g) \
      --volume "${source_dir}:${source_dir}:ro" \
      --volume "${OUTPUT_DIR}:${OUTPUT_DIR}" \
      --volume "${ARCHIVE_DIR}:${ARCHIVE_DIR}" \
      --volume "${SCRIPT_DIR}:${SCRIPT_DIR}:ro" \
      ${DOCKER_IMAGE} \
      python "${SCRIPT_DIR}/urdf_trellis.py" \
        --glb_file="${glb_file}" \
        --asset_id="${ASSET_ID}" \
        --category="${CATEGORY}" \
        --source_name="${source_name}" \
        --output_dir="${OUTPUT_DIR}" \
        --work_dir="${WORK_CATEGORY_DIR}" \
        --archive_dir="${ARCHIVE_CATEGORY_DIR}" \
        --stages 0 1 2 3 4 5 6 \
      > "${LOG_FILE}" 2>&1

    local result=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    trap - EXIT

    rmdir "${LOCK_FILE}" 2>/dev/null
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null

    if [ $result -eq 124 ] || [ $result -eq 137 ]; then
        echo "[$(date '+%H:%M:%S')] [${source_name}/${CATEGORY}/${ASSET_ID}] ✗ TIMEOUT (${duration}s)"
        echo "${source_name}/${CATEGORY}/${ASSET_ID}" >> "${TIMEOUT_FILE}"
        rm -rf "${WORK_ASSET_DIR}" 2>/dev/null
        return 1
    elif [ $result -eq 0 ] && [ -f "${ARCHIVE_FILE}" ]; then
        echo "[$(date '+%H:%M:%S')] [${source_name}/${CATEGORY}/${ASSET_ID}] ✓ Success (${duration}s)"
        echo "${source_name}/${CATEGORY}/${ASSET_ID}" >> "${PROGRESS_FILE}"
        return 0
    else
        echo "[$(date '+%H:%M:%S')] [${source_name}/${CATEGORY}/${ASSET_ID}] ✗ Failed (code: $result, ${duration}s)"
        echo "${source_name}/${CATEGORY}/${ASSET_ID}" >> "${FAILED_FILE}"
        return 1
    fi
}

monitor_progress() {
    local label=$1
    local total=$2
    local archive_dir=$3

    while true; do
        sleep ${PROGRESS_INTERVAL}

        if [ -f "${OUTPUT_DIR}/.stop_monitor" ]; then
            rm -f "${OUTPUT_DIR}/.stop_monitor"
            break
        fi

        local done_count=$(find "${archive_dir}" -name "*.tar.gz" 2>/dev/null | wc -l)
        local running=$(docker ps --filter "name=trellis_" -q 2>/dev/null | wc -l)
        local pct=0
        if [ $total -gt 0 ]; then
            pct=$((done_count * 100 / total))
        fi

        log "[${label}] Progress: ${done_count}/${total} (${pct}%) | Running: ${running}"
    done
}

export -f process_glb log
export OUTPUT_DIR ARCHIVE_DIR SCRIPT_DIR LOG_DIR
export PROGRESS_FILE FAILED_FILE TIMEOUT_FILE LOCK_DIR
export TIMEOUT_SECONDS DOCKER_IMAGE

# ============================================================================
# Main processing logic
# ============================================================================

START_TIME=$(date +%s)

log "=========================================="
log "Trellis Assets Parallel Processing (trellis1 + trellis2)"
log "=========================================="
log "Source1: ${SOURCE_DIR_1}"
log "Source2: ${SOURCE_DIR_2}"
log "Output: ${OUTPUT_DIR}"
log "Archive: ${ARCHIVE_DIR}"
log "Parallel jobs: ${NUM_JOBS}"
log "Timeout: ${TIMEOUT_SECONDS}s"
log "=========================================="

check_docker
cleanup_containers

# Sources
SOURCE_DIRS=("${SOURCE_DIR_1}" "${SOURCE_DIR_2}")
SOURCE_NAMES=("trellis1" "trellis2")

# Auto-scan categories from the union of both source directories.
if [ ${#CATEGORIES_TO_PROCESS[@]} -eq 0 ]; then
    tmpfile="$(mktemp)"
    > "$tmpfile"
    for idx in 0 1; do
        src="${SOURCE_DIRS[$idx]}"
        if [ -d "$src" ]; then
            find "$src" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; >> "$tmpfile"
        fi
    done
    mapfile -t CATEGORIES_TO_PROCESS < <(sort -u "$tmpfile")
    rm -f "$tmpfile"
fi

log "Categories to process: ${CATEGORIES_TO_PROCESS[*]}"
log ""

# Process each category by combining GLB files from trellis1 and trellis2.
for CATEGORY in "${CATEGORIES_TO_PROCESS[@]}"; do
    log "=========================================="
    log "Processing category: ${CATEGORY}"
    log "=========================================="

    mkdir -p "${OUTPUT_DIR}/${CATEGORY}"
    mkdir -p "${ARCHIVE_DIR}/${CATEGORY}"

    ALL_GLB_FILES=()

    for idx in 0 1; do
        src="${SOURCE_DIRS[$idx]}"
        sname="${SOURCE_NAMES[$idx]}"

        CATEGORY_PATH="${src}/${CATEGORY}"
        if [ ! -d "$CATEGORY_PATH" ]; then
            continue
        fi

        mapfile -t GLB_FILES < <(find "$CATEGORY_PATH" -name "*.glb" -type f | sort)
        if [ ${#GLB_FILES[@]} -gt 0 ]; then
            for f in "${GLB_FILES[@]}"; do
                # encode source name + path so xargs can pass both
                ALL_GLB_FILES+=("${sname}:::${src}:::${f}")
            done
        fi
    done

    TOTAL=${#ALL_GLB_FILES[@]}
    if [ ${TOTAL} -eq 0 ]; then
        log "No GLB files found for ${CATEGORY} in both sources, skipping"
        continue
    fi

    ALREADY_DONE=$(find "${ARCHIVE_DIR}/${CATEGORY}" -name "*.tar.gz" 2>/dev/null | wc -l)
    log "Total input glb: ${TOTAL}, Already done tar.gz in archive: ${ALREADY_DONE}"

    CATEGORY_START=$(date +%s)

    monitor_progress "${CATEGORY}" "${TOTAL}" "${ARCHIVE_DIR}/${CATEGORY}" &
    MONITOR_PID=$!

    # run parallel: split "source:::srcdir:::file"
    printf '%s\n' "${ALL_GLB_FILES[@]}" | \
      xargs -P ${NUM_JOBS} -I {} bash -c '
        item="$1"
        sname="${item%%:::*}"
        rest="${item#*:::}"
        srcdir="${rest%%:::*}"
        glb="${rest#*:::}"
        process_glb "$srcdir" "$sname" "$glb"
      ' _ {}

    touch "${OUTPUT_DIR}/.stop_monitor"
    wait $MONITOR_PID 2>/dev/null
    rm -f "${OUTPUT_DIR}/.stop_monitor"

    wait
    cleanup_containers

    CATEGORY_END=$(date +%s)
    CATEGORY_DURATION=$((CATEGORY_END - CATEGORY_START))

    SUCCESS=$(find "${ARCHIVE_DIR}/${CATEGORY}" -name "*.tar.gz" 2>/dev/null | wc -l)

    log ""
    log "Category ${CATEGORY} completed:"
    log "  Duration: $(($CATEGORY_DURATION / 60))m $(($CATEGORY_DURATION % 60))s"
    log "  Total tar.gz in archive now: ${SUCCESS}"
    log ""
done

# ============================================================================
# Generate metadata
# ============================================================================

log "=========================================="
log "Generating metadata..."
log "=========================================="

python "${SCRIPT_DIR}/trellis_metadata.py" \
    --work_dir="${OUTPUT_DIR}" \
    --archive_dir="${ARCHIVE_DIR}" \
    --output="${ARCHIVE_DIR}/metadata.json" \
    --combined

if [ $? -eq 0 ]; then
    log "Metadata generated successfully: ${ARCHIVE_DIR}/metadata.json"
else
    log "WARNING: Metadata generation failed"
fi

# ============================================================================
# Final statistics
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log ""
log "=========================================="
log "Final Summary"
log "=========================================="
log "Total duration: $(($DURATION / 60))m $(($DURATION % 60))s"

TOTAL_SUCCESS=$(wc -l < "${PROGRESS_FILE}" 2>/dev/null || echo 0)
TOTAL_FAILED=$(wc -l < "${FAILED_FILE}" 2>/dev/null || echo 0)
TOTAL_TIMEOUT=$(wc -l < "${TIMEOUT_FILE}" 2>/dev/null || echo 0)

log "Successful: ${TOTAL_SUCCESS}"
log "Failed: ${TOTAL_FAILED}"
log "Timeout: ${TOTAL_TIMEOUT}"

if [ ${TOTAL_TIMEOUT} -gt 0 ]; then
    log ""
    log "Timeout files (first 10):"
    head -10 "${TIMEOUT_FILE}" 2>/dev/null | while read line; do log "  - $line"; done
fi

if [ ${TOTAL_FAILED} -gt 0 ]; then
    log ""
    log "Failed files (first 10):"
    head -10 "${FAILED_FILE}" 2>/dev/null | while read line; do log "  - $line"; done
fi

cleanup_containers
rm -rf "${LOCK_DIR}"

log ""
log "Output files:"
log "  - Metadata: ${ARCHIVE_DIR}/metadata.json"
log "  - Statistics: ${ARCHIVE_DIR}/statistics.json"
log "  - Category index: ${ARCHIVE_DIR}/category_index.json"
log "  - Prompt index: ${ARCHIVE_DIR}/prompt_index.json"
log "  - Processing log: ${STATS_LOG}"
log ""
log "Done!"
