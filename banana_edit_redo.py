import requests
import base64
import json
import os
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
import argparse
import numpy as np

# ================= Config (same as banana_edit_level.py) =================
BASE_URL = "http://<API_HOST>:<PORT>"
MODEL_NAME = "gemini-3-pro-image-preview"
API_ENDPOINT = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent"
API_KEY = "sk-<YOUR_API_KEY>"

MAX_WORKERS = 20
RETRY_TIMES = 3
TIMEOUT_SECONDS = 180

# ================= Prompt Templates (copied from banana_edit_level.py) =================
LEVEL_1_PROMPT = """Photorealistic image editing based on input RGB and a single mask for {category}.

PRIMARY DIRECTIVE: PRESERVE GEOMETRY & COUNT
Keep the object silhouettes and the number of {category} instances EXACTLY the same as indicated by the mask.

Task Instructions:
1. Object Texture Editing (Moderate Freedom, Realistic):
   - Improve realism with natural texture/material variations (e.g., subtle color variations, wear, manufacturing differences).
   - Increase intra-category diversity so instances are not identical, but keep all instances clearly {category}.
   - Avoid extreme patterns, logos, text, or implausible materials for this category.
2. Lighting & Shadows:
   - Ensure edited objects remain consistent with the global illumination (high-quality shading and plausible shadows).
3. Background Generation:
   - Generate a clean, high-quality background that is semantically coherent with {category}.
   - Keep perspective consistent (ground plane, scale, depth), and match lighting to the objects.

CRITICAL CONSTRAINTS (Strictly Adhere):
- Strict Mask Adherence: ALL edits to {category} must occur STRICTLY INSIDE the mask boundaries.
  Do NOT expand, shrink, warp, or reshape the objects. Do NOT change the silhouette.
- Zero Tolerance for New Instances: It is STRICTLY FORBIDDEN to generate any additional {category} instances in unmasked regions.
  The number of {category} objects MUST remain exactly the same as shown by the mask.
- Object Integrity: Do not remove, merge, split, or duplicate instances. Do not occlude objects with new content.
- Background Purity: Background should be purely environmental and clean. Do NOT add people, animals, or distracting objects.
- Photorealism: Output must be realistic, with consistent perspective and physically plausible lighting."""

LEVEL_2_SIZE_PROMPT = """Photorealistic image editing based on input RGB and masks for {category} (two size groups).

PRIMARY DIRECTIVE: PRESERVE SIZE DISTINCTION & COUNT
The size difference is the key signal. Preserve each instance's size and silhouette exactly as defined by the masks.

Task Instructions:
1. Object Texture Editing (Moderate Freedom, Realistic):
   - Apply realistic texture/material variations across ALL {category} instances while keeping them clearly {category}.
   - Keep textures coherent and plausible; subtle diversity is preferred over extreme style changes.
   - Ensure texture edits do not reduce perceived size difference between the two groups.
2. Background Generation:
   - Generate a coherent environment that suits {category}.
   - Maintain correct perspective, scale cues, and lighting consistency.
3. Fine Details:
   - Improve realism: consistent contact shadows, mild specular highlights, and material consistency.

CRITICAL CONSTRAINTS (Strictly Adhere):
- Strict Mask Adherence: Use the masks as rigid containers. Do NOT change silhouettes, sizes, or boundaries.
- Preserve Size Difference: Do NOT alter object geometry or perspective in a way that undermines the Large vs Small distinction.
- Zero Tolerance for New Instances: Do NOT add any new {category} instances (neither large nor small) in unmasked regions.
- Object Count Consistency: If the mask contains N instances, the output MUST contain exactly N instances.
- Background Purity: Background must be environmental only; do NOT add people/animals or distracting objects."""

LEVEL_2_COLOR_PROMPT = """Photorealistic background replacement based on input RGB and mask for {category}.

PRIMARY DIRECTIVE: PRESERVE OBJECT APPEARANCE (COLOR/TEXTURE)
The group distinction relies on original colors. You must NOT modify the object colors, textures, or materials inside the mask.

Task Instructions:
1. Objects (Strict Preservation):
   - Keep the original RGB pixels INSIDE the mask exactly unchanged.
   - Do not apply any style transfer, recoloring, or texture editing to masked regions.
2. Background Generation:
   - Replace the unmasked background with a high-quality, aesthetic environment coherent with {category}.
   - Match lighting and shadows so the objects appear naturally placed.

Additional Info:
- Group colors (do not change): {color_a_name} vs {color_b_name}

CRITICAL CONSTRAINTS (Strictly Adhere):
- No Texture/Color Changes: It is STRICTLY PROHIBITED to alter any masked pixels (color/texture/material).
- Strict Mask Adherence: Do NOT bleed edits across mask boundaries.
- Zero Tolerance for New Instances: Do NOT add extra {category} objects in the background.
- Background Purity: Keep background clean; avoid people, animals, text, or clutter.
- Photorealism: Maintain consistent perspective and lighting."""

LEVEL_3_PROMPT = """Photorealistic image editing based on input RGB and separate masks for {category_A} and {category_B}.

PRIMARY DIRECTIVE: CATEGORY CONSISTENCY & COUNT
Keep {category_A} and {category_B} instances unchanged in geometry and count, and keep the two categories clearly distinguishable.

Task Instructions:
1. Object Texture Editing (Moderate Freedom, Realistic):
   - Apply realistic texture/material improvements and subtle variations ONLY inside each mask.
   - Keep each instance clearly within its original category (no category drift).
   - Prefer coherent, plausible materials over extreme style changes.
2. Background Generation:
   - Generate a coherent, high-quality background that plausibly fits BOTH categories together.
   - Maintain consistent lighting direction, shadows, perspective, and scale cues.

CRITICAL CONSTRAINTS (Strictly Adhere):
- Strict Mask Adherence: Edit ONLY inside masks. Do NOT change silhouettes, boundaries, or object placement.
- ZERO TOLERANCE FOR NEW INSTANCES (Most Important):
  It is STRICTLY FORBIDDEN to generate any additional {category_A} or {category_B} objects in unmasked regions.
  The background MUST be purely environmental. Do NOT add duplicates, partial objects, reflections, posters, pictures, toys, or decorative motifs of {category_A}/{category_B}.
- Object Count Consistency: The number of {category_A} and {category_B} instances MUST remain exactly the same as indicated by the masks.
- Background Purity: Do NOT add people, animals, text, logos, or distracting objects that could be confused with the target categories.
- Photorealism: Output must be realistic and consistent with the scene geometry and illumination."""

LEVEL_4_PROMPT = """Photorealistic image compositing based on input RGB and separate masks for {category} Type-A and {category} Type-B.

PRIMARY DIRECTIVE: INTRA-CLASS DISTINCTION & COUNT
Type-A and Type-B must remain distinguishable. Preserve object geometry and count exactly as indicated by the masks.

Task Instructions:
1. Background (Priority):
   - Generate a coherent, photorealistic environment suitable for {category}.
   - Maintain correct perspective, geometry, and lighting.
2. Texture Editing (Conditional, Conservative):
   - You MAY apply subtle texture/material improvements to increase realism and diversity,
     ONLY IF Type-A vs Type-B remains clearly separable.
   - If there is any risk of confusion, do NOT edit object textures; keep original object appearance.

CRITICAL CONSTRAINTS (Strictly Adhere):
- Strict Mask Adherence: Do NOT change silhouettes; edit only within masks.
- Preserve Distinction (Important): Do NOT make Type-A look like Type-B or vice versa.
- Zero Tolerance for New Instances (Important): Do NOT generate any additional {category} instances in unmasked regions.
- Object Count Consistency (Important): Do NOT remove, merge, split, or duplicate objects; count must remain identical.
- Background Purity: Keep background clean and environmental; avoid people/animals and clutter."""

LEVEL_5_PROMPT = """Photorealistic image compositing based on input RGB and separate masks for {category_A} and {category_B}.

PRIMARY DIRECTIVE: INTER-CLASS DISTINCTION & COUNT
Ensure {category_A} and {category_B} remain clearly distinguishable. Preserve geometry and object count exactly as indicated by the masks.

Task Instructions:
1. Background (Priority):
   - Generate a coherent, high-quality environment that plausibly contains BOTH {category_A} and {category_B}.
   - Maintain consistent perspective, ground plane, and contact shadows.
2. Texture Editing (Conditional, Conservative):
   - Apply realistic texture/material improvements and mild variation ONLY IF it does not confuse categories.
   - If uncertain, keep object textures unchanged and focus on background realism.

CRITICAL CONSTRAINTS (Strictly Adhere):
- Strict Mask Adherence: Edit only within masks; do NOT change silhouettes or boundaries.
- Category Consistency (Important): Do NOT turn {category_A} into {category_B} or vice versa.
- Zero Tolerance for New Instances (Important): Do NOT add any new {category_A} or {category_B} objects in unmasked regions.
- Object Count Consistency (Important): Do NOT remove, merge, split, or duplicate any object instances. Every object indicated by the masks MUST remain present and clearly visible in the final image.
- Background Purity: Avoid people, animals, and distracting objects; keep background environmental and clean."""

# ================= Thread-safe =================
log_lock = threading.Lock()

# ================= Helpers =================
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_pil_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_base64_image(b64_str, output_path):
    image_data = base64.b64decode(b64_str)
    image = Image.open(BytesIO(image_data))
    image.load()
    image.save(output_path)

def safe_read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def get_metadata_info(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    level = int(metadata.get("level", 1))

    distinction_type = None
    level2_color_names = None
    groups = metadata.get("groups", []) or []

    if level == 2:
        level_specific_info = metadata.get("level_specific_info", {}) or {}
        distinction_type = level_specific_info.get("distinction_type", "size")

        if distinction_type == "color" and len(groups) >= 2:
            a = groups[0].get("target_color_name")
            b = groups[1].get("target_color_name")
            level2_color_names = {
                "color_a_name": a or "color-A",
                "color_b_name": b or "color-B",
            }

    categories = []
    for group in groups:
        category = (group.get("category", "") or "")
        if category.startswith("trellis_"):
            category = category[len("trellis_"):]
        if category == "bat":
            category = "bat/stick"
        if category:
            categories.append(category)

    unique_categories = []
    for cat in categories:
        if cat not in unique_categories:
            unique_categories.append(cat)

    return {
        "level": level,
        "distinction_type": distinction_type,
        "categories": unique_categories,
        "metadata": metadata,
        "level2_color_names": level2_color_names,
    }

def split_mask_by_group(seg_path, metadata):
    img = Image.open(seg_path)
    palette = img.getpalette()
    mask = np.array(img)

    instances = metadata.get("instances", []) or []
    group0_seg_ids = {i + 1 for i, inst in enumerate(instances) if inst.get("group_index", 0) == 0}
    group1_seg_ids = {i + 1 for i, inst in enumerate(instances) if inst.get("group_index", 0) == 1}

    mask_group0 = np.where(np.isin(mask, list(group0_seg_ids)), mask, 0).astype(np.uint8)
    mask_group1 = np.where(np.isin(mask, list(group1_seg_ids)), mask, 0).astype(np.uint8)

    img_group0 = Image.fromarray(mask_group0, mode="P")
    img_group0.putpalette(palette)
    img_group1 = Image.fromarray(mask_group1, mode="P")
    img_group1.putpalette(palette)

    return img_group0, img_group1

def get_prompt_and_images(folder_path, metadata_info):
    level = int(metadata_info["level"])
    distinction_type = metadata_info["distinction_type"]
    categories = metadata_info["categories"]
    level2_color_names = metadata_info.get("level2_color_names")

    rgb_path = os.path.join(folder_path, "rgba_00000.png")
    seg_path = os.path.join(folder_path, "segmentation_00000.png")

    rgb_b64 = encode_image_to_base64(rgb_path)
    default_category = categories[0] if categories else "object"

    if level == 1:
        prompt = LEVEL_1_PROMPT.format(category=default_category)
        mask_b64 = encode_image_to_base64(seg_path)
        return prompt, [rgb_b64, mask_b64], "L1"

    if level == 2:
        mask_b64 = encode_image_to_base64(seg_path)
        if distinction_type == "color":
            color_a_name = "color-A"
            color_b_name = "color-B"
            if level2_color_names:
                color_a_name = level2_color_names.get("color_a_name", color_a_name)
                color_b_name = level2_color_names.get("color_b_name", color_b_name)
            prompt = LEVEL_2_COLOR_PROMPT.format(
                category=default_category,
                color_a_name=color_a_name,
                color_b_name=color_b_name,
            )
            return prompt, [rgb_b64, mask_b64], "L2_color"
        prompt = LEVEL_2_SIZE_PROMPT.format(category=default_category)
        return prompt, [rgb_b64, mask_b64], "L2_size"

    if level == 3:
        if len(categories) >= 2:
            category_a = categories[0]
            category_b = categories[1]
        else:
            category_a = default_category
            category_b = "object"
        prompt = LEVEL_3_PROMPT.format(category_A=category_a, category_B=category_b)
        m0, m1 = split_mask_by_group(seg_path, metadata_info["metadata"])
        return prompt, [rgb_b64, encode_pil_image_to_base64(m0), encode_pil_image_to_base64(m1)], "L3"

    if level == 4:
        prompt = LEVEL_4_PROMPT.format(category=default_category)
        m0, m1 = split_mask_by_group(seg_path, metadata_info["metadata"])
        return prompt, [rgb_b64, encode_pil_image_to_base64(m0), encode_pil_image_to_base64(m1)], "L4"

    if level == 5:
        if len(categories) >= 2:
            category_a = categories[0]
            category_b = categories[1]
        else:
            category_a = default_category
            category_b = "object"
        prompt = LEVEL_5_PROMPT.format(category_A=category_a, category_B=category_b)
        m0, m1 = split_mask_by_group(seg_path, metadata_info["metadata"])
        return prompt, [rgb_b64, encode_pil_image_to_base64(m0), encode_pil_image_to_base64(m1)], "L5"

    prompt = LEVEL_1_PROMPT.format(category=default_category)
    mask_b64 = encode_image_to_base64(seg_path)
    return prompt, [rgb_b64, mask_b64], f"L{level}_fallback"

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

def redo_targets_from_filter_results(filter_json_path):
    data = safe_read_json(filter_json_path, default={})
    ann = data.get("annotations", {})
    if not isinstance(ann, dict):
        ann = {}
    fail_keys = [k for k, v in ann.items() if v == "FAIL"]
    return data, fail_keys

def _call_edit_api_once(prompt, images_b64):
    parts = [{"text": prompt}]
    for img_b64 in images_b64:
        parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.35,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 4096
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
    return resp

def call_edit_api_with_retry(prompt, images_b64, retry_times):
    attempts = max(1, int(retry_times))
    last_resp = None
    last_err = None

    for _ in range(attempts):
        try:
            resp = _call_edit_api_once(prompt, images_b64)
            last_resp = resp
            if resp is not None and getattr(resp, "status_code", None) == 200:
                return resp, None
            # Non-200: still retry
            if resp is not None:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                last_err = "No response object"
        except requests.exceptions.Timeout:
            last_err = f"Request timeout ({TIMEOUT_SECONDS}s)"
        except requests.exceptions.RequestException as e:
            last_err = f"Request exception: {str(e)}"
        except Exception as e:
            last_err = f"Unexpected exception: {str(e)}"

    return last_resp, f"Retry exhausted ({attempts} attempts). Last error: {last_err}"

def process_one_scene(scene_key, scene_path, retry_times):
    result = {
        "key": scene_key,
        "path": scene_path,
        "success": False,
        "error": None,
        "status_code": None,
        "level": None,
        "level_key": None,
    }

    metadata_path = os.path.join(scene_path, "metadata.json")
    rgb_path = os.path.join(scene_path, "rgba_00000.png")
    seg_path = os.path.join(scene_path, "segmentation_00000.png")
    out_path = os.path.join(scene_path, "edited_00000.png")

    if not (os.path.exists(metadata_path) and os.path.exists(rgb_path) and os.path.exists(seg_path)):
        missing = []
        if not os.path.exists(metadata_path):
            missing.append("metadata.json")
        if not os.path.exists(rgb_path):
            missing.append("rgba_00000.png")
        if not os.path.exists(seg_path):
            missing.append("segmentation_00000.png")
        result["error"] = f"Missing required files: {missing}"
        return result

    try:
        metadata_info = get_metadata_info(metadata_path)
        result["level"] = int(metadata_info["level"])
        prompt, images, level_key = get_prompt_and_images(scene_path, metadata_info)
        result["level_key"] = level_key

        resp, retry_err = call_edit_api_with_retry(prompt, images, retry_times=retry_times)
        if resp is None:
            result["error"] = retry_err or "No response"
            return result

        result["status_code"] = resp.status_code

        if resp.status_code != 200:
            # If we got here, retry is exhausted
            result["error"] = retry_err or f"HTTP Error: {resp.text[:500]}"
            return result

        response_json = resp.json()
        candidates = response_json.get("candidates", [])
        if not candidates:
            prompt_feedback = response_json.get("promptFeedback", {}) or {}
            block_reason = prompt_feedback.get("blockReason", "Unknown")
            result["error"] = f"No candidates. blockReason={block_reason}"
            return result

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason", "")
        parts_out = first_candidate.get("content", {}).get("parts", [])
        if not parts_out:
            result["error"] = f"No parts. finishReason={finish_reason}"
            return result

        image_b64 = None
        text_parts = []
        for p in parts_out:
            if "inlineData" in p:
                image_b64 = p["inlineData"].get("data")
                if image_b64:
                    break
            if "inline_data" in p:
                image_b64 = p["inline_data"].get("data")
                if image_b64:
                    break
            if "text" in p:
                text_parts.append(p["text"])

        if not image_b64:
            preview = ""
            if text_parts:
                joined = "\n".join(text_parts)
                preview = joined[:500] + ("..." if len(joined) > 500 else "")
            result["error"] = f"No image in response parts. finishReason={finish_reason}. text_preview={preview}"
            return result

        save_base64_image(image_b64, out_path)
        result["success"] = True
        return result

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result

def main(root_path, workers, filter_results_json=None, limit_per_date=None, retry_times=RETRY_TIMES):
    if filter_results_json is None:
        filter_results_json = os.path.join(root_path, "vlm_filter_results.json")

    _, fail_keys = redo_targets_from_filter_results(filter_results_json)

    print("=" * 70)
    print("Redo Editing for FAIL scenes")
    print("=" * 70)
    print(f"Root: {root_path}")
    print(f"Filter results: {filter_results_json}")
    print(f"Workers: {workers}")
    print(f"limit_per_date: {limit_per_date if limit_per_date is not None else 'ALL'}")
    print(f"Retry times: {retry_times}")
    print(f"Timeout: {TIMEOUT_SECONDS}")
    print(f"FAIL targets (from annotations): {len(fail_keys)}")
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
        print("Nothing to redo.")
        return

    started_at = datetime.now()
    total = len(targets)
    done = 0

    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one_scene, k, p, retry_times): (k, p) for k, p in targets}
        for fut in as_completed(futs):
            done += 1
            k, _ = futs[fut]
            r = fut.result()

            ok = bool(r.get("success"))
            if ok:
                successes += 1
                status = "OK"
            else:
                failures += 1
                status = "FAIL"

            msg = ""
            if r.get("error"):
                msg = f" | {r['error']}"
            print(f"[{done:6d}/{total:6d}] {status} {k} (level={r.get('level_key')}, http={r.get('status_code')}){msg}")

    dur = datetime.now() - started_at
    print("=" * 70)
    print("Redo editing finished.")
    print(f"Duration: {dur}")
    print(f"Success: {successes}  Fail: {failures}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redo editing for scenes marked FAIL in vlm_filter_results.json")
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
        retry_times=int(args.retry_times) if args.retry_times else 1,
    )

# python banana_edit_redo.py --root_path KubriCount/train --workers 100 --retry_times 3
# python banana_edit_redo.py --root_path KubriCount/testA --workers 80 --retry_times 3
# python banana_edit_redo.py --root_path KubriCount/testB --workers 80 --retry_times 3
