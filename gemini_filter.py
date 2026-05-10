# gemini_filter.py
import requests
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
import argparse

# ================= Config =================
BASE_URL = "http://<API_HOST>:<PORT>"
MODEL_NAME = "gemini-3-pro-preview"
API_ENDPOINT = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent"
API_KEY = "sk-<YOUR_API_KEY>"

MAX_WORKERS = 20
FLUSH_EVERY = 2000  # write partial results every N samples (set to 1000 or 2000)
RETRY_TIMES = 3     # retry API call up to N times on failure
TIMEOUT_SECONDS = 180

# ================= VLM Filter Prompt =================
VLM_FILTER_PROMPT = """You are a strict visual quality inspector for a synthetic counting dataset.

You will be given:
- Image A: original RGB render
- Image B: segmentation mask(s) (one mask, or multiple masks for different groups/categories)
- Image C: edited RGB result

Your task:
Decide whether Image C is a valid edit of Image A under the constraints implied by the mask(s).
Output ONLY one token: PASS or FAIL. Do NOT output any explanation.

Acceptance note:
- Small mask-boundary/silhouette deviations are acceptable.
- Object positions, object counts, and object categories MUST remain unchanged.

PASS conditions (ALL must hold):

1) Position/layout preservation (Strict):
   - Each instance remains at the same image location as in Image A (no shifting/repositioning).
   - Global layout is unchanged.
   - No camera/viewpoint change: perspective, scale, and vanishing points remain consistent.

2) Count consistency (Strict):
   - The number of instances indicated by the mask(s) remains EXACTLY the same.
   - No instance is removed, duplicated, merged, or split.

3) Category consistency (Strict):
   - Each masked instance remains the same category as in Image A / implied by the mask(s).
   - For multi-mask (two-group/two-category) input: no swapping between masks; categories remain distinguishable.

4) Missing-instance check with edge/corner focus (Strict, with exception):
   - Focus especially on instances near image borders and corners (these are most likely to be accidentally dropped).
   - Compare Image A vs Image C at those edge/corner locations: if an instance that is present in Image A is missing in Image C, FAIL.
   - Exception: if an instance is already partially out-of-frame in Image A (cropped by the image boundary), you may ignore small visibility differences; do NOT fail solely because that already-cropped instance becomes slightly less visible. But do fail if it disappears entirely.

5) No new target-category instances, focusing on background-only regions (Strict):
   - Focus especially on regions that are background in Image A and outside the mask(s).
   - It is STRICTLY FORBIDDEN to introduce any new instances of the target categories in Image C within these originally background regions.

6) Editing locality (Moderate):
   - Object edits mainly inside masks; background edits mainly outside masks.
   - Minor boundary leakage is acceptable ONLY if it does not change position, count, or category.

7) Image integrity (Strict):
   - No severe artifacts that invalidate the sample: missing regions, duplicated edges creating extra instances, heavy blur making instances uncountable, or obvious geometric distortions.

FAIL rule:
If ANY strict check (1)-(5) or (7) fails, output FAIL.
If uncertain about any strict check, output FAIL.

Output format:
PASS
or
FAIL"""

# ================= Thread-safe state =================
log_lock = threading.Lock()
print_lock = threading.Lock()

# ================= Helpers =================
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def safe_read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_path, path)

def normalize_result_text(text):
    t = (text or "").strip().upper()
    if t == "PASS":
        return "PASS"
    if t == "FAIL":
        return "FAIL"
    if t.startswith("PASS"):
        return "PASS"
    if t.startswith("FAIL"):
        return "FAIL"
    return "FAIL"

def build_request_parts(rgb_b64, mask_b64, edited_b64):
    return [
        {"text": VLM_FILTER_PROMPT},
        {"inline_data": {"mime_type": "image/png", "data": rgb_b64}},
        {"inline_data": {"mime_type": "image/png", "data": mask_b64}},
        {"inline_data": {"mime_type": "image/png", "data": edited_b64}},
    ]

def _call_vlm_filter_once(rgb_path, mask_path, edited_path):
    rgb_b64 = encode_image_to_base64(rgb_path)
    mask_b64 = encode_image_to_base64(mask_path)
    edited_b64 = encode_image_to_base64(edited_path)

    payload = {
        "contents": [{"parts": build_request_parts(rgb_b64, mask_b64, edited_b64)}],
        "generationConfig": {
            "temperature": 0.0,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 10240
        }
    }

    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}

    resp = requests.post(
        API_ENDPOINT,
        headers=headers,
        params=params,
        data=json.dumps(payload),
        timeout=TIMEOUT_SECONDS
    )

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:500]}"

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        prompt_feedback = data.get("promptFeedback", {}) or {}
        block_reason = prompt_feedback.get("blockReason", "Unknown")
        return None, f"No candidates. blockReason={block_reason}"

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        finish_reason = candidates[0].get("finishReason", "")
        return None, f"No parts. finishReason={finish_reason}"

    texts = []
    for p in parts:
        if "text" in p:
            texts.append(p["text"])
    text = "\n".join(texts).strip() if texts else ""
    if not text:
        finish_reason = candidates[0].get("finishReason", "")
        return None, f"No text output. finish_reason={finish_reason}"

    return normalize_result_text(text), None

def call_vlm_filter(rgb_path, mask_path, edited_path, retry_times=RETRY_TIMES):
    # retry_times=3 means up to 3 attempts total
    last_err = None
    attempts = max(1, int(retry_times))
    for i in range(attempts):
        verdict, err = _call_vlm_filter_once(rgb_path, mask_path, edited_path)
        if err is None and verdict in ("PASS", "FAIL"):
            return verdict, None
        last_err = err or "Unknown error"
    return None, f"Retry exhausted ({attempts} attempts). Last error: {last_err}"

def scene_key_from_root(root_path, scene_folder_path):
    rel = os.path.relpath(scene_folder_path, root_path)
    return rel.replace("\\", "/")

def iter_scene_folders(root_path, limit_per_date=None):
    if not os.path.exists(root_path):
        return

    level_dir_names = {
        "level1", "level2", "level3", "level4", "level5",
        "level_1", "level_2", "level_3", "level_4", "level_5",
    }
    entries = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    level_dirs = [d for d in sorted(entries) if d in level_dir_names]
    date_roots = [os.path.join(root_path, lvl) for lvl in level_dirs] if level_dirs else [root_path]

    for date_root in date_roots:
        for date_folder in sorted(os.listdir(date_root)):
            date_folder_path = os.path.join(date_root, date_folder)
            if not os.path.isdir(date_folder_path):
                continue

            scenes = [s for s in sorted(os.listdir(date_folder_path)) if os.path.isdir(os.path.join(date_folder_path, s))]
            if limit_per_date is not None:
                scenes = scenes[: int(limit_per_date)]

            for scene in scenes:
                yield os.path.join(date_folder_path, scene)

def gather_tasks(root_path, limit_per_date, existing_annotations):
    tasks = []
    missing = []
    skipped = 0

    for scene_path in iter_scene_folders(root_path, limit_per_date=limit_per_date):
        key = scene_key_from_root(root_path, scene_path)
        if key in existing_annotations:
            skipped += 1
            continue

        rgb_path = os.path.join(scene_path, "rgba_00000.png")
        seg_path = os.path.join(scene_path, "segmentation_00000.png")
        edited_path = os.path.join(scene_path, "edited_00000.png")

        if not (os.path.exists(rgb_path) and os.path.exists(seg_path) and os.path.exists(edited_path)):
            missing_files = []
            if not os.path.exists(rgb_path):
                missing_files.append("rgba_00000.png")
            if not os.path.exists(seg_path):
                missing_files.append("segmentation_00000.png")
            if not os.path.exists(edited_path):
                missing_files.append("edited_00000.png")
            missing.append({"path": key, "missing_files": missing_files})
            continue

        tasks.append({
            "key": key,
            "scene_path": scene_path,
            "rgb_path": rgb_path,
            "seg_path": seg_path,
            "edited_path": edited_path,
        })

    return tasks, missing, skipped

def process_one_task(task, retry_times):
    key = task["key"]
    verdict, err = call_vlm_filter(task["rgb_path"], task["seg_path"], task["edited_path"], retry_times=retry_times)
    if err is not None:
        return {"key": key, "verdict": "FAIL", "error": err}
    return {"key": key, "verdict": verdict, "error": None}

def update_stats(annotations):
    stats = {"PASS": 0, "FAIL": 0}
    for _, v in annotations.items():
        if v == "PASS":
            stats["PASS"] += 1
        else:
            stats["FAIL"] += 1
    return stats

def write_checkpoint(out_json, root_path, model_name, start_time, workers, limit_per_date,
                     existing_annotations, new_annotations, missing, errors,
                     processed_this_run, skipped_existing, flush_count, retry_times):
    merged_annotations = dict(existing_annotations)
    merged_annotations.update(new_annotations)

    stats = update_stats(merged_annotations)
    out = {
        "model": model_name,
        "root_path": root_path,
        "updated_at": datetime.now().isoformat(),
        "run": {
            "started_at": start_time,
            "last_checkpoint_at": datetime.now().isoformat(),
            "workers": int(workers),
            "limit_per_date": None if limit_per_date is None else int(limit_per_date),
            "processed_this_run": int(processed_this_run),
            "skipped_existing_this_run": int(skipped_existing),
            "missing_files_this_run": int(len(missing)),
            "errors_this_run": int(len(errors)),
            "flush_every": int(flush_count),
            "retry_times": int(retry_times),
            "timeout_seconds": int(TIMEOUT_SECONDS),
        },
        "stats": stats,
        "total_annotations": int(len(merged_annotations)),
        "annotations": merged_annotations,
        "missing": missing,
        "errors": errors,
    }
    safe_write_json(out_json, out)
    return stats, len(merged_annotations)

def main(root_path, workers, limit_per_date=None, flush_every=FLUSH_EVERY, retry_times=RETRY_TIMES):
    out_json = os.path.join(root_path, "vlm_filter_results.json")
    existing = safe_read_json(out_json, default={})
    existing_annotations = existing.get("annotations", {}) if isinstance(existing, dict) else {}
    if not isinstance(existing_annotations, dict):
        existing_annotations = {}

    tasks, missing, skipped_existing = gather_tasks(
        root_path=root_path,
        limit_per_date=limit_per_date,
        existing_annotations=existing_annotations,
    )

    start_time = datetime.now().isoformat()

    print("=" * 70)
    print("VLM Filter (PASS/FAIL)")
    print("=" * 70)
    print(f"Root: {root_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Workers: {workers}")
    print(f"limit_per_date: {limit_per_date if limit_per_date is not None else 'ALL'}")
    print(f"Flush every: {flush_every}")
    print(f"Retry times: {retry_times}")
    print(f"Timeout: {TIMEOUT_SECONDS}")
    print(f"Existing annotations: {len(existing_annotations)} (skipped this run: {skipped_existing})")
    print(f"To process this run: {len(tasks)}")
    print(f"Missing required files: {len(missing)}")
    print(f"Output JSON: {out_json}")
    print("=" * 70)

    new_annotations = {}
    errors = []

    completed = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_task = {ex.submit(process_one_task, t, retry_times): t for t in tasks}
        for fut in as_completed(future_to_task):
            t = future_to_task[fut]
            key = t["key"]
            completed += 1
            try:
                r = fut.result()
                verdict = r["verdict"]
                err = r["error"]

                with log_lock:
                    new_annotations[key] = verdict
                    if err:
                        errors.append({"path": key, "error": err})

                with print_lock:
                    if err:
                        print(f"[{completed:6d}/{total:6d}] {key} -> {verdict} | ERROR: {err}")
                    else:
                        print(f"[{completed:6d}/{total:6d}] {key} -> {verdict}")

                if flush_every and (completed % int(flush_every) == 0):
                    with log_lock:
                        stats, total_ann = write_checkpoint(
                            out_json=out_json,
                            root_path=root_path,
                            model_name=MODEL_NAME,
                            start_time=start_time,
                            workers=workers,
                            limit_per_date=limit_per_date,
                            existing_annotations=existing_annotations,
                            new_annotations=new_annotations,
                            missing=missing,
                            errors=errors,
                            processed_this_run=completed,
                            skipped_existing=skipped_existing,
                            flush_count=flush_every,
                            retry_times=retry_times,
                        )
                    with print_lock:
                        print("-" * 70)
                        print(f"Checkpoint saved at {completed} processed. Total annotations={total_ann}. PASS={stats['PASS']} FAIL={stats['FAIL']}")
                        print("-" * 70)

            except Exception as e:
                with log_lock:
                    new_annotations[key] = "FAIL"
                    errors.append({"path": key, "error": f"Exception: {str(e)}"})
                with print_lock:
                    print(f"[{completed:6d}/{total:6d}] {key} -> FAIL | ERROR: Exception: {str(e)}")

                if flush_every and (completed % int(flush_every) == 0):
                    with log_lock:
                        stats, total_ann = write_checkpoint(
                            out_json=out_json,
                            root_path=root_path,
                            model_name=MODEL_NAME,
                            start_time=start_time,
                            workers=workers,
                            limit_per_date=limit_per_date,
                            existing_annotations=existing_annotations,
                            new_annotations=new_annotations,
                            missing=missing,
                            errors=errors,
                            processed_this_run=completed,
                            skipped_existing=skipped_existing,
                            flush_count=flush_every,
                            retry_times=retry_times,
                        )
                    with print_lock:
                        print("-" * 70)
                        print(f"Checkpoint saved at {completed} processed. Total annotations={total_ann}. PASS={stats['PASS']} FAIL={stats['FAIL']}")
                        print("-" * 70)

    stats, total_ann = write_checkpoint(
        out_json=out_json,
        root_path=root_path,
        model_name=MODEL_NAME,
        start_time=start_time,
        workers=workers,
        limit_per_date=limit_per_date,
        existing_annotations=existing_annotations,
        new_annotations=new_annotations,
        missing=missing,
        errors=errors,
        processed_this_run=completed,
        skipped_existing=skipped_existing,
        flush_count=flush_every,
        retry_times=retry_times,
    )

    with print_lock:
        print("\nDone.")
        print(f"Processed this run: {completed}")
        print(f"Total annotations: {total_ann}")
        print(f"Stats: PASS={stats['PASS']} FAIL={stats['FAIL']}")
        print(f"Wrote: {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM filter for edited images (PASS/FAIL)")
    parser.add_argument(
        "--root_path",
        type=str,
        default="KubriCount/train",
        help="Root path containing scene folders (optionally under level/date/scene structure)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Parallel workers."
    )
    parser.add_argument(
        "--limit_per_date",
        type=int,
        default=None,
        help="Max number of scenes to process under each date folder (default: ALL)."
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=FLUSH_EVERY,
        help="Write checkpoint JSON every N processed samples (default 2000)."
    )
    parser.add_argument(
        "--retry_times",
        type=int,
        default=RETRY_TIMES,
        help="Retry API call up to N attempts on failure (default 3)."
    )

    args = parser.parse_args()
    MAX_WORKERS = int(args.workers)

    main(
        root_path=args.root_path,
        workers=MAX_WORKERS,
        limit_per_date=args.limit_per_date,
        flush_every=int(args.flush_every) if args.flush_every else 0,
        retry_times=int(args.retry_times) if args.retry_times else 1,
    )

# Examples:
# python gemini_filter.py --root_path KubriCount/train --workers 120 --flush_every 1000
# python gemini_filter.py --root_path KubriCount/testA --workers 40 --flush_every 2000

# python gemini_filter.py --root_path KubriCount/train --workers 10 --limit_per_date 1
# python gemini_filter.py --root_path KubriCount/train --workers 20

# python gemini_filter.py --root_path KubriCount/train --workers 120 --flush_every 1000 --retry_times 10
# python gemini_filter.py --root_path KubriCount/testA --workers 80 --flush_every 1000
# python gemini_filter.py --root_path KubriCount/testB --workers 80 --flush_every 1000
