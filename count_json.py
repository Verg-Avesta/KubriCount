#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def convert_bbox_to_absolute(bbox, width: int, height: int) -> List[int]:
    """
    Convert normalized bbox [y_min, x_min, y_max, x_max] to absolute pixel coords [x1,y1,x2,y2].
    NOTE: Raw metadata format is [ymin, xmin, ymax, xmax] (not [xmin,ymin,xmax,ymax]).
    """
    if not bbox or width <= 0 or height <= 0:
        return []
    y1, x1, y2, x2 = bbox
    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]


def convert_center_to_absolute(center, width: int, height: int) -> List[float]:
    """
    Convert normalized center [x,y] to absolute pixel coords [x,y].
    """
    if not center or width <= 0 or height <= 0:
        return []
    x, y = center
    return [x * width, y * height]


def bbox_xyxy_to_polygon(xyxy: List[int]) -> List[List[int]]:
    """
    [x1,y1,x2,y2] -> [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
    """
    if not xyxy or len(xyxy) != 4:
        return []
    x1, y1, x2, y2 = xyxy
    return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]


def _safe_get_resolution(meta: Dict[str, Any]) -> Tuple[int, int]:
    res = None
    if isinstance(meta.get("metadata"), dict):
        res = meta["metadata"].get("resolution")
    if isinstance(res, list) and len(res) >= 2:
        w, h = int(res[0]), int(res[1])
        return w, h

    ir = meta.get("image_resolution")
    if isinstance(ir, int):
        return int(ir), int(ir)

    return 0, 0


def _get_level(meta: Dict[str, Any]) -> int:
    flags = meta.get("flags") if isinstance(meta.get("flags"), dict) else {}
    lvl = flags.get("level", meta.get("level"))
    try:
        return int(lvl)
    except Exception:
        return -1


def _get_split(meta: Dict[str, Any], root: str) -> str:
    si = meta.get("split_info") if isinstance(meta.get("split_info"), dict) else {}
    for k in ("objects_split", "backgrounds_split"):
        v = si.get(k)
        if isinstance(v, str):
            return v

    bi = meta.get("background_info") if isinstance(meta.get("background_info"), dict) else {}
    if isinstance(bi.get("split"), str):
        return bi["split"]

    r = root.lower()
    if "/train" in r:
        return "train"
    if "/test" in r or "/val" in r or "/valid" in r:
        return "test"
    return ""


def _get_config_file(meta: Dict[str, Any]) -> str:
    flags = meta.get("flags") if isinstance(meta.get("flags"), dict) else {}
    cf = flags.get("config_file")
    return cf if isinstance(cf, str) else ""


def _get_level2_mode(meta: Dict[str, Any]) -> str:
    lsi = meta.get("level_specific_info") if isinstance(meta.get("level_specific_info"), dict) else {}
    mode = lsi.get("mode")
    if isinstance(mode, str) and mode:
        return mode
    dt = lsi.get("distinction_type")
    if isinstance(dt, str) and dt:
        return dt
    groups = meta.get("groups")
    if isinstance(groups, list) and len(groups) > 0 and isinstance(groups[0], dict):
        m = groups[0].get("level2_mode")
        if isinstance(m, str) and m:
            return m
    return ""


def _group_specific_category_name(level: int, group: Dict[str, Any], level2_mode: str) -> str:
    base_cat = group.get("category", "")
    if not isinstance(base_cat, str):
        base_cat = ""

    if level == 2:
        m = (level2_mode or "").lower().strip()
        if m == "size":
            sm = group.get("size_mode", "")
            if isinstance(sm, str):
                if "small" in sm:
                    return f"smaller {base_cat}".strip()
                if "large" in sm:
                    return f"larger {base_cat}".strip()
            return base_cat

        if m == "color":
            cn = group.get("target_color_name", "")
            if isinstance(cn, str) and cn.strip():
                return f"{cn.strip()} {base_cat}".strip()
            return base_cat

    return base_cat


def _group_category_map_specific(meta: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    groups = meta.get("groups")
    if not isinstance(groups, list):
        return out

    level = _get_level(meta)
    level2_mode = _get_level2_mode(meta)

    for g in groups:
        if not isinstance(g, dict):
            continue
        gi = g.get("group_index")
        try:
            gi = int(gi)
        except Exception:
            continue
        out[gi] = _group_specific_category_name(level, g, level2_mode)
    return out


def _aggregate_instances_for_group(
    meta: Dict[str, Any],
    group_index: int,
    width: int,
    height: int,
) -> Tuple[List[List[List[int]]], List[List[float]]]:
    polys: List[List[List[int]]] = []
    pts: List[List[float]] = []

    insts = meta.get("instances")
    if not isinstance(insts, list):
        return polys, pts

    for inst in insts:
        if not isinstance(inst, dict):
            continue
        gi = inst.get("group_index", inst.get("group_idx"))
        try:
            gi = int(gi)
        except Exception:
            continue
        if gi != group_index:
            continue

        bboxes = inst.get("bboxes")
        if isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list):
            bbox_abs = convert_bbox_to_absolute(bboxes[0], width, height)
            poly = bbox_xyxy_to_polygon(bbox_abs)
            if poly:
                polys.append(poly)

        img_pos = inst.get("image_positions")
        if isinstance(img_pos, list) and len(img_pos) > 0 and isinstance(img_pos[0], list):
            c = convert_center_to_absolute(img_pos[0], width, height)
            if c:
                pts.append(c)

    return polys, pts


def build_items_for_scene(scene_dir: str, root: str) -> List[Dict[str, Any]]:
    edited_path = os.path.join(scene_dir, "edited_00000.png")
    meta_path = os.path.join(scene_dir, "metadata.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    level = _get_level(meta)
    split = _get_split(meta, root)
    config_file = _get_config_file(meta)

    w, h = _safe_get_resolution(meta)
    is_single = bool(meta.get("is_single_type", False))

    group_cat = _group_category_map_specific(meta)

    pos_gi, neg_gi = 0, 1

    pos_boxes, pos_points = _aggregate_instances_for_group(meta, pos_gi, w, h)
    neg_boxes, neg_points = _aggregate_instances_for_group(meta, neg_gi, w, h)

    pos_count = len(pos_points) if pos_points else len(pos_boxes)
    neg_count = len(neg_points) if neg_points else len(neg_boxes)

    pos_category = group_cat.get(pos_gi, "")
    neg_category = group_cat.get(neg_gi, "")

    meta_block: Dict[str, Any] = {
        "level": level,
        "split": split,
        "config_file": config_file,
    }
    if level == 2:
        m = _get_level2_mode(meta)
        if m:
            meta_block["level2_mode"] = m

    image_abs = os.path.abspath(edited_path)

    def make_item(
        primary_count: int,
        primary_cat: str,
        primary_boxes: List[List[List[int]]],
        primary_points: List[List[float]],
        secondary_count: int,
        secondary_cat: str,
        secondary_boxes: List[List[List[int]]],
        secondary_points: List[List[float]],
        secondary_enabled: bool,
    ) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "image_id": image_abs,
            "count": int(primary_count),
            "box_examples_coordinates": primary_boxes,
            "points": primary_points,
            "H": int(h),
            "W": int(w),
            "category": primary_cat,
            "metadata": meta_block,
            "negative_count": 0,
            "negative_category": "",
            "negative_box_examples_coordinates": [],
            "negative_points": [],
        }

        if secondary_enabled:
            item["negative_count"] = int(secondary_count)
            item["negative_category"] = secondary_cat
            item["negative_box_examples_coordinates"] = secondary_boxes
            item["negative_points"] = secondary_points

        return item

    items: List[Dict[str, Any]] = []

    if is_single or level == 1:
        items.append(
            make_item(
                primary_count=pos_count,
                primary_cat=pos_category,
                primary_boxes=pos_boxes,
                primary_points=pos_points,
                secondary_count=0,
                secondary_cat="",
                secondary_boxes=[],
                secondary_points=[],
                secondary_enabled=False,
            )
        )
        return items

    # Level 2-5: always dual-type; output two items with swapped positive/negative
    items.append(
        make_item(
            primary_count=pos_count,
            primary_cat=pos_category,
            primary_boxes=pos_boxes,
            primary_points=pos_points,
            secondary_count=neg_count,
            secondary_cat=neg_category,
            secondary_boxes=neg_boxes,
            secondary_points=neg_points,
            secondary_enabled=True,
        )
    )
    items.append(
        make_item(
            primary_count=neg_count,
            primary_cat=neg_category,
            primary_boxes=neg_boxes,
            primary_points=neg_points,
            secondary_count=pos_count,
            secondary_cat=pos_category,
            secondary_boxes=pos_boxes,
            secondary_points=pos_points,
            secondary_enabled=True,
        )
    )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract counting metadata from Kubric outputs using vlm_filter_results.json (keep PASS only)."
    )
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root path, e.g. KubriCount/train",
    )
    ap.add_argument(
        "--filter_results",
        type=str,
        default=None,
        help="Path to vlm_filter_results.json (default: <root>/vlm_filter_results.json)",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path, e.g. <root>/extracted_metadata.json",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, missing files/parse errors will raise instead of skipping.",
    )

    args = ap.parse_args()
    root = os.path.abspath(args.root)
    filt_path = args.filter_results or os.path.join(root, "vlm_filter_results.json")
    out_path = os.path.abspath(args.out)

    with open(filt_path, "r", encoding="utf-8") as f:
        filt = json.load(f)

    ann_map = filt.get("annotations")
    if not isinstance(ann_map, dict):
        raise ValueError(f'Invalid vlm_filter_results.json: expected dict at key "annotations", got {type(ann_map)}')

    pass_items = [(rel, status) for rel, status in ann_map.items() if status == "PASS" and isinstance(rel, str)]

    extracted: List[Dict[str, Any]] = []
    skipped = 0

    for rel, _ in tqdm(pass_items, desc="Extract PASS scenes", unit="scene"):
        scene_dir = os.path.join(root, rel)
        meta_path = os.path.join(scene_dir, "metadata.json")
        edited_path = os.path.join(scene_dir, "edited_00000.png")

        if not (os.path.isfile(meta_path) and os.path.isfile(edited_path)):
            msg = f"Missing files under: {scene_dir}"
            if args.strict:
                raise FileNotFoundError(msg)
            skipped += 1
            continue

        try:
            extracted.extend(build_items_for_scene(scene_dir, root))
        except Exception:
            if args.strict:
                raise
            skipped += 1
            continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(extracted)} items to: {out_path}")
    print(f"PASS scenes in filter: {len(pass_items)}")
    if skipped:
        print(f"Skipped {skipped} PASS scenes due to missing files or parse errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Examples:
# python count_json.py --root KubriCount/train --out KubriCount/train/extracted_metadata.json --strict
# python count_json.py --root KubriCount/testA --out KubriCount/testA/extracted_metadata.json --strict
# python count_json.py --root KubriCount/testB --out KubriCount/testB/extracted_metadata.json --strict
