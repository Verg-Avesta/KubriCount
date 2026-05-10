# gemini_filter_redo.py
import requests
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
import argparse

# ================= Config (same as gemini_filter.py) =================
BASE_URL = "http://<API_HOST>:<PORT>"
MODEL_NAME = "gemini-3-pro-preview"
API_ENDPOINT = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent"
API_KEY = "sk-<YOUR_API_KEY>"

MAX_WORKERS = 20
FLUSH_EVERY = 2000
RETRY_TIMES = 3
TIMEOUT_SECONDS = 180

# ================= Prompt (same as gemini_filter.py) =================
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

# ================= Thread-safe =================
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
    last_err = None
    attempts = max(1, int(retry_times))
    for _ in range(attempts):
        verdict, err = _call_vlm_filter_once(rgb_path, mask_path, edited_path)
        if err is None and verdict in ("PASS", "FAIL"):
            return verdict, None
        last_err = err or "Unknown error"
    return None, f"Retry exhausted ({attempts} attempts). Last error: {last_err}"

def build_scene_index(root_path, limit_per_date=None):
    idx = {}
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
                scene_path = os.path.join(date_folder_path, scene)
                key = os.path.relpath(scene_path, root_path).replace("\\", "/")
                idx[key] = scene_path

    return idx

def process_one(scene_key, scene_path, retry_times):
    rgb_path = os.path.join(scene_path, "rgba_00000.png")
    seg_path = os.path.join(scene_path, "segmentation_00000.png")
    edited_path = os.path.join(scene_path, "edited_00000.png")

    if not (os.path.exists(rgb_path) and os.path.exists(seg_path) and os.path.exists(edited_path)):
        missing = []
        if not os.path.exists(rgb_path):
            missing.append("rgba_00000.png")
        if not os.path.exists(seg_path):
            missing.append("segmentation_00000.png")
        if not os.path.exists(edited_path):
            missing.append("edited_00000.png")
        return {"key": scene_key, "verdict": "FAIL", "error": f"Missing files: {missing}"}

    verdict, err = call_vlm_filter(rgb_path, seg_path, edited_path, retry_times=retry_times)
    if err is not None:
        return {"key": scene_key, "verdict": "FAIL", "error": err}
    return {"key": scene_key, "verdict": verdict, "error": None}

def update_stats(annotations):
    stats = {"PASS": 0, "FAIL": 0}
    for _, v in annotations.items():
        if v == "PASS":
            stats["PASS"] += 1
        else:
            stats["FAIL"] += 1
    return stats

def write_checkpoint(filter_results_json, data, annotations, redo_results, errors, redo_entry,
                     done, flush_every, retry_times):
    # Apply PASS flips into annotations
    flipped = 0
    for k, v in redo_results.items():
        if v == "PASS" and annotations.get(k) == "FAIL":
            annotations[k] = "PASS"
            flipped += 1

    # Update redo history with checkpoints (append-only)
    history = data.get("redo_history", [])
    if not isinstance(history, list):
        history = []
    history.append({
        **redo_entry,
        "checkpoint_at": datetime.now().isoformat(),
        "processed_this_run": int(done),
        "flipped_fail_to_pass_so_far": int(flipped),
        "errors_so_far": int(len(errors)),
        "flush_every": int(flush_every),
        "retry_times": int(retry_times),
        "timeout_seconds": int(TIMEOUT_SECONDS),
    })
    data["redo_history"] = history

    data["annotations"] = annotations
    data["stats"] = update_stats(annotations)
    data["total_annotations"] = int(len(annotations))
    data["updated_at"] = datetime.now().isoformat()
    data["last_redo"] = {
        "run": {
            **redo_entry,
            "checkpoint_at": datetime.now().isoformat(),
            "processed_this_run": int(done),
            "flush_every": int(flush_every),
            "retry_times": int(retry_times),
            "timeout_seconds": int(TIMEOUT_SECONDS),
        },
        "results": redo_results,
        "errors": errors,
    }

    safe_write_json(filter_results_json, data)
    return flipped, data["stats"]

def main(root_path, workers, filter_results_json=None, limit_per_date=None, flush_every=FLUSH_EVERY, retry_times=RETRY_TIMES):
    if filter_results_json is None:
        filter_results_json = os.path.join(root_path, "vlm_filter_results.json")

    data = safe_read_json(filter_results_json, default={})
    annotations = data.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError("vlm_filter_results.json has invalid format: annotations must be a dict")

    fail_keys = [k for k, v in annotations.items() if v == "FAIL"]

    print("=" * 70)
    print("Redo Filtering for FAIL scenes")
    print("=" * 70)
    print(f"Root: {root_path}")
    print(f"Filter results: {filter_results_json}")
    print(f"Model: {MODEL_NAME}")
    print(f"Workers: {workers}")
    print(f"limit_per_date: {limit_per_date if limit_per_date is not None else 'ALL'}")
    print(f"Flush every: {flush_every}")
    print(f"Retry times: {retry_times}")
    print(f"Timeout: {TIMEOUT_SECONDS}")
    print(f"FAIL targets (current annotations): {len(fail_keys)}")
    print("=" * 70)

    scene_index = build_scene_index(root_path, limit_per_date=limit_per_date)

    targets = []
    missing_paths = []
    for k in fail_keys:
        p = scene_index.get(k)
        if p is None:
            missing_paths.append(k)
            continue
        targets.append((k, p))

    if missing_paths:
        print(f"Warning: {len(missing_paths)} FAIL keys not found under current root/limit_per_date; skipped.")

    if not targets:
        print("Nothing to redo-filter.")
        return

    started_at = datetime.now().isoformat()
    total = len(targets)
    done = 0

    redo_results = {}
    errors = []

    redo_entry = {
        "started_at": started_at,
        "model": MODEL_NAME,
        "target_fail_count": int(len(fail_keys)),
        "target_found_count": int(len(targets)),
        "missing_paths": int(len(missing_paths)),
        "workers": int(workers),
        "limit_per_date": None if limit_per_date is None else int(limit_per_date),
    }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, k, p, retry_times): (k, p) for k, p in targets}
        for fut in as_completed(futs):
            done += 1
            k, _ = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"key": k, "verdict": "FAIL", "error": f"Future exception: {str(e)}"}

            verdict = r["verdict"]
            err = r.get("error")

            with log_lock:
                redo_results[k] = verdict
                if err:
                    errors.append({"path": k, "error": err})

            with print_lock:
                if err:
                    print(f"[{done:6d}/{total:6d}] {k} -> {verdict} | ERROR: {err}")
                else:
                    print(f"[{done:6d}/{total:6d}] {k} -> {verdict}")

            if flush_every and (done % int(flush_every) == 0):
                with log_lock:
                    flipped_so_far, stats = write_checkpoint(
                        filter_results_json=filter_results_json,
                        data=data,
                        annotations=annotations,
                        redo_results=redo_results,
                        errors=errors,
                        redo_entry=redo_entry,
                        done=done,
                        flush_every=flush_every,
                        retry_times=retry_times,
                    )
                with print_lock:
                    print("-" * 70)
                    print(f"Checkpoint saved at {done} processed. Flipped so far={flipped_so_far}. PASS={stats['PASS']} FAIL={stats['FAIL']}")
                    print("-" * 70)

    flipped_final, stats_final = write_checkpoint(
        filter_results_json=filter_results_json,
        data=data,
        annotations=annotations,
        redo_results=redo_results,
        errors=errors,
        redo_entry={**redo_entry, "finished_at": datetime.now().isoformat()},
        done=done,
        flush_every=flush_every,
        retry_times=retry_times,
    )

    print("\nRedo-filter done.")
    print(f"Processed: {done}")
    print(f"Flipped FAIL->PASS (final): {flipped_final}")
    print(f"Updated stats: PASS={stats_final['PASS']} FAIL={stats_final['FAIL']}")
    print(f"Wrote: {filter_results_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redo filter for scenes currently marked FAIL in vlm_filter_results.json")
    parser.add_argument("--root_path", type=str, required=True, help="Dataset root (contains level/date/scene folders).")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers.")
    parser.add_argument(
        "--filter_results_json",
        type=str,
        default=None,
        help="Path to vlm_filter_results.json (default: <root_path>/vlm_filter_results.json)"
    )
    parser.add_argument(
        "--limit_per_date",
        type=int,
        default=None,
        help="Max number of scenes to consider under each date folder (default: ALL)."
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
        filter_results_json=args.filter_results_json,
        limit_per_date=args.limit_per_date,
        flush_every=int(args.flush_every) if args.flush_every else 0,
        retry_times=int(args.retry_times) if args.retry_times else 1,
    )

# python gemini_filter_redo.py --root_path KubriCount/train --workers 120 --flush_every 1000
# python gemini_filter_redo.py --root_path KubriCount/testA --workers 80 --flush_every 200
# python gemini_filter_redo.py --root_path KubriCount/testB --workers 80 --flush_every 200
