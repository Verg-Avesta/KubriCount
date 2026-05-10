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

# ================= Config =================
BASE_URL = "http://<API_HOST>:<PORT>"
MODEL_NAME = "gemini-3-pro-image-preview"
API_ENDPOINT = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent"
API_KEY = "sk-<YOUR_API_KEY>"

MAX_WORKERS = 20

# ================= Prompt Templates (Levels 1-5) =================
# Principles:
# - Strong "PRIMARY DIRECTIVE" and detailed "Task Instructions"
# - Strict "CRITICAL CONSTRAINTS" modeled after the old prompts
# - Prefer semantic coherence and consistency; texture edits are realistic, not overly free-form.

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

# ================= Logging =================
log_lock = threading.Lock()

processing_log = {
    "start_time": None,
    "end_time": None,
    "total_to_process": 0,
    "total_success": 0,
    "total_failed": 0,
    "total_skipped_already_done": 0,
    "total_skipped_missing_files": 0,
    "level_stats": {},
    "success": [],
    "failed": [],
    "skipped_already_done": [],
    "skipped_missing_files": []
}

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
    image.save(output_path)

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
        "groups": groups,
        "instances": metadata.get("instances", []) or [],
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
        mask_group0, mask_group1 = split_mask_by_group(seg_path, metadata_info["metadata"])
        return prompt, [rgb_b64, encode_pil_image_to_base64(mask_group0), encode_pil_image_to_base64(mask_group1)], "L3"

    if level == 4:
        prompt = LEVEL_4_PROMPT.format(category=default_category)
        mask_group0, mask_group1 = split_mask_by_group(seg_path, metadata_info["metadata"])
        return prompt, [rgb_b64, encode_pil_image_to_base64(mask_group0), encode_pil_image_to_base64(mask_group1)], "L4"

    if level == 5:
        if len(categories) >= 2:
            category_a = categories[0]
            category_b = categories[1]
        else:
            category_a = default_category
            category_b = "object"
        prompt = LEVEL_5_PROMPT.format(category_A=category_a, category_B=category_b)
        mask_group0, mask_group1 = split_mask_by_group(seg_path, metadata_info["metadata"])
        return prompt, [rgb_b64, encode_pil_image_to_base64(mask_group0), encode_pil_image_to_base64(mask_group1)], "L5"

    prompt = LEVEL_1_PROMPT.format(category=default_category)
    mask_b64 = encode_image_to_base64(seg_path)
    return prompt, [rgb_b64, mask_b64], f"L{level}_fallback"

def find_all_scene_folders(root_path, limit_per_date=None, overwrite=False):
    folders_to_process = []
    folders_already_done = []
    folders_missing_files = []

    if not os.path.exists(root_path):
        print(f"Error: path does not exist: {root_path}")
        return [], [], []

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

            scene_names = [s for s in sorted(os.listdir(date_folder_path)) if os.path.isdir(os.path.join(date_folder_path, s))]
            if limit_per_date is not None:
                scene_names = scene_names[: int(limit_per_date)]

            for scene_folder in scene_names:
                scene_folder_path = os.path.join(date_folder_path, scene_folder)

                metadata_path = os.path.join(scene_folder_path, "metadata.json")
                rgba_path = os.path.join(scene_folder_path, "rgba_00000.png")
                seg_path = os.path.join(scene_folder_path, "segmentation_00000.png")
                edited_path = os.path.join(scene_folder_path, "edited_00000.png")

                has_metadata = os.path.exists(metadata_path)
                has_rgba = os.path.exists(rgba_path)
                has_seg = os.path.exists(seg_path)

                if not (has_metadata and has_rgba and has_seg):
                    missing = []
                    if not has_metadata:
                        missing.append("metadata.json")
                    if not has_rgba:
                        missing.append("rgba_00000.png")
                    if not has_seg:
                        missing.append("segmentation_00000.png")
                    folders_missing_files.append({"path": scene_folder_path, "missing_files": missing})
                    continue

                if os.path.exists(edited_path) and not overwrite:
                    folders_already_done.append({"path": scene_folder_path})
                    continue

                folders_to_process.append(scene_folder_path)

    return folders_to_process, folders_already_done, folders_missing_files

def process_single_folder(folder_path, overwrite=False):
    result = {
        "path": folder_path,
        "success": False,
        "error": None,
        "status_code": None,
        "level": None,
        "level_key": None,
        "distinction_type": None,
        "categories": None
    }

    try:
        metadata_path = os.path.join(folder_path, "metadata.json")
        output_path = os.path.join(folder_path, "edited_00000.png")

        if overwrite and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass

        metadata_info = get_metadata_info(metadata_path)
        result["level"] = metadata_info["level"]
        result["distinction_type"] = metadata_info["distinction_type"]
        result["categories"] = metadata_info["categories"]

        prompt, images, level_key = get_prompt_and_images(folder_path, metadata_info)
        result["level_key"] = level_key

        parts = [{"text": prompt}]
        for img_b64 in images:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": img_b64
                }
            })

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

        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            params=params,
            data=json.dumps(payload),
            timeout=180
        )

        result["status_code"] = response.status_code

        if response.status_code != 200:
            error_text = response.text[:500] if len(response.text) > 500 else response.text
            result["error"] = f"HTTP Error: {error_text}"
            return result

        response_json = response.json()
        candidates = response_json.get("candidates", [])
        if not candidates:
            prompt_feedback = response_json.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "Unknown")
            result["error"] = f"No candidates in response. Block reason: {block_reason}"
            return result

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason", "")
        parts_out = first_candidate.get("content", {}).get("parts", [])
        if not parts_out:
            result["error"] = f"No parts in response. Finish reason: {finish_reason}"
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

        if image_b64:
            save_base64_image(image_b64, output_path)
            result["success"] = True
            return result

        preview = ""
        if text_parts:
            joined = "\n".join(text_parts)
            preview = joined[:500] + ("..." if len(joined) > 500 else "")
        result["error"] = f"Model returned no image in parts. finish_reason={finish_reason}. text_preview={preview}"
        return result

    except FileNotFoundError as e:
        result["error"] = f"File not found: {str(e)}"
    except json.JSONDecodeError as e:
        result["error"] = f"JSON decode error: {str(e)}"
    except requests.exceptions.Timeout:
        result["error"] = "Request timeout (180s)"
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request exception: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result

def save_processing_log(root_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"processing_log_{timestamp}.json"
    log_path = os.path.join(root_path, log_filename)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(processing_log, f, indent=2, ensure_ascii=False)

    print(f"\nLog saved to: {log_path}")
    return log_path

def main(root_path, limit_per_date=None, overwrite=False):
    global processing_log

    print("=" * 70)
    print("Batch Image Editing Script (Levels 1-5)")
    print("=" * 70)
    print(f"Root path: {root_path}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Overwrite: {overwrite}")
    if limit_per_date is None:
        print("Per-date limit: ALL")
    else:
        print(f"Per-date limit: {limit_per_date}")
    print(f"API endpoint: {API_ENDPOINT}")
    print("-" * 70)
    print("Levels:")
    print("  L1       : single category, single asset (texture+bg)")
    print("  L2_size  : same asset, two size groups (texture+bg)")
    print("  L2_color : preserve object colors, background only (uses target_color_name)")
    print("  L3       : two categories, one asset each (split mask; texture+bg)")
    print("  L4       : same category, two assets (split mask; conditional texture)")
    print("  L5       : two categories (split mask; conditional texture)")
    print("=" * 70)

    processing_log["start_time"] = datetime.now().isoformat()
    processing_log["root_path"] = root_path

    print("\nScanning folders...")
    folders_to_process, folders_already_done, folders_missing_files = find_all_scene_folders(
        root_path, limit_per_date=limit_per_date, overwrite=overwrite
    )

    processing_log["skipped_already_done"] = folders_already_done
    processing_log["skipped_missing_files"] = folders_missing_files
    processing_log["total_skipped_already_done"] = len(folders_already_done)
    processing_log["total_skipped_missing_files"] = len(folders_missing_files)
    processing_log["total_to_process"] = len(folders_to_process)

    print("\nScan results:")
    print(f"  - To process: {len(folders_to_process)}")
    print(f"  - Skipped (already done): {len(folders_already_done)}")
    print(f"  - Skipped (missing files): {len(folders_missing_files)}")

    if not folders_to_process:
        print("\nNothing to process.")
        processing_log["end_time"] = datetime.now().isoformat()
        save_processing_log(root_path)
        return

    print(f"\nProcessing with ThreadPoolExecutor (workers={MAX_WORKERS})...")
    print("-" * 70)

    completed_count = 0
    total_count = len(folders_to_process)
    level_stats = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_folder = {
            executor.submit(process_single_folder, folder, overwrite): folder
            for folder in folders_to_process
        }

        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            completed_count += 1

            try:
                result = future.result()

                with log_lock:
                    level_key = result.get("level_key", "unknown")
                    if level_key not in level_stats:
                        level_stats[level_key] = {"success": 0, "failed": 0}

                    if result["success"]:
                        processing_log["success"].append({
                            "path": result["path"],
                            "status_code": result["status_code"],
                            "level": result["level"],
                            "level_key": level_key,
                            "distinction_type": result["distinction_type"],
                            "categories": result["categories"],
                        })
                        processing_log["total_success"] += 1
                        level_stats[level_key]["success"] += 1
                        status_icon = "OK"
                        status_msg = f"Success [{level_key}]"
                    else:
                        processing_log["failed"].append({
                            "path": result["path"],
                            "status_code": result["status_code"],
                            "error": result["error"],
                            "level": result["level"],
                            "level_key": level_key,
                            "distinction_type": result["distinction_type"],
                            "categories": result["categories"],
                        })
                        processing_log["total_failed"] += 1
                        level_stats[level_key]["failed"] += 1
                        status_icon = "FAIL"
                        err = result["error"] or ""
                        err_preview = (err[:80] + "...") if len(err) > 80 else err
                        status_msg = f"Failed [{level_key}] - {err_preview}"

                    relative_path = os.path.relpath(result["path"], root_path)
                    print(f"[{completed_count:4d}/{total_count}] {status_icon} {relative_path} | {status_msg}")

            except Exception as e:
                with log_lock:
                    processing_log["failed"].append({
                        "path": folder,
                        "status_code": None,
                        "error": f"Future exception: {str(e)}",
                        "level": None,
                        "level_key": "exception",
                        "distinction_type": None,
                        "categories": None
                    })
                    processing_log["total_failed"] += 1
                    level_stats.setdefault("exception", {"success": 0, "failed": 0})
                    level_stats["exception"]["failed"] += 1

                    relative_path = os.path.relpath(folder, root_path)
                    print(f"[{completed_count:4d}/{total_count}] FAIL {relative_path} | Exception - {str(e)}")

    processing_log["end_time"] = datetime.now().isoformat()
    processing_log["level_stats"] = level_stats

    start_time = datetime.fromisoformat(processing_log["start_time"])
    end_time = datetime.fromisoformat(processing_log["end_time"])
    duration = end_time - start_time
    processing_log["duration_seconds"] = duration.total_seconds()

    save_processing_log(root_path)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Success: {processing_log['total_success']}")
    print(f"Failed: {processing_log['total_failed']}")
    print(f"Skipped (already done): {processing_log['total_skipped_already_done']}")
    print(f"Skipped (missing files): {processing_log['total_skipped_missing_files']}")

    if level_stats:
        print("\nPer-level stats:")
        print("-" * 40)
        for k in sorted(level_stats.keys()):
            s = level_stats[k]
            tot = s["success"] + s["failed"]
            rate = (s["success"] / tot * 100.0) if tot > 0 else 0.0
            print(f"  {k:12s}: {s['success']:4d} success, {s['failed']:4d} failed ({rate:.1f}%)")

    if processing_log["total_success"] > 0:
        avg_time = duration.total_seconds() / processing_log["total_success"]
        print(f"\nAvg time per success: {avg_time:.2f} s")

    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image editing (Levels 1-5)")
    parser.add_argument(
        "--root_path",
        type=str,
        default="KubriCount/train",
        help="Root path containing date folders (and optionally level folders)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Parallel workers (default 20)"
    )
    parser.add_argument(
        "--limit_per_date",
        type=int,
        default=None,
        help="Max number of scene folders to process under each date folder (default: ALL)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing edited_00000.png instead of skipping"
    )

    args = parser.parse_args()
    MAX_WORKERS = int(args.workers)
    main(args.root_path, limit_per_date=args.limit_per_date, overwrite=bool(args.overwrite))

# Examples:
# python banana_edit_level.py --root_path KubriCount/train --workers 20 --limit_per_date 2 --overwrite
# python banana_edit_level.py --root_path KubriCount/train --workers 20 --overwrite

# python banana_edit_level.py --root_path KubriCount/train --workers 120
# python banana_edit_level.py --root_path KubriCount/testA --workers 80
# python banana_edit_level.py --root_path KubriCount/testB --workers 80
