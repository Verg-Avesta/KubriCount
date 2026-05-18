#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class Meter:
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    n: float = 0.0  # weighted

    def add(self, pred: float, gt: float, weight: float = 1.0):
        err = float(pred) - float(gt)
        self.sum_abs += abs(err) * float(weight)
        self.sum_sq += (err * err) * float(weight)
        self.n += float(weight)

    def mae(self) -> float:
        return self.sum_abs / self.n if self.n > 0 else 0.0

    def rmse(self) -> float:
        return math.sqrt(self.sum_sq / self.n) if self.n > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"n": float(self.n), "mae": float(self.mae()), "rmse": float(self.rmse())}


def polygon_to_xyxy(poly: Any) -> Optional[List[float]]:
    # poly: [[x,y], [x,y], [x,y], [x,y]] (or more points)
    if not isinstance(poly, list) or len(poly) < 4:
        return None
    xs, ys = [], []
    for p in poly:
        if not isinstance(p, list) or len(p) != 2:
            return None
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    return [min(xs), min(ys), max(xs), max(ys)]


def clamp_xyxy(xyxy: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(x1, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    x2 = max(0.0, min(x2, float(w)))
    y2 = max(0.0, min(y2, float(h)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def get_group_keys(item: Dict[str, Any]) -> Tuple[str, int, str]:
    md = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    split = md.get("split", "unknown")
    try:
        level = int(md.get("level", -1))
    except Exception:
        level = -1
    level2_mode = md.get("level2_mode", "all")
    if not isinstance(level2_mode, str) or level2_mode.strip() == "":
        level2_mode = "unknown"
    return str(split), int(level), str(level2_mode)


def resolve_image_path(item: Dict[str, Any], base_image_dir: str = "") -> str:
    image_id = item.get("image_id", "")
    if not isinstance(image_id, str):
        return ""
    if os.path.isabs(image_id):
        return image_id
    if base_image_dir:
        return os.path.join(base_image_dir, image_id)
    return image_id


def select_first_k_exemplars(item: Dict[str, Any], k: int) -> Optional[List[List[float]]]:
    polys = item.get("box_examples_coordinates", [])
    if not isinstance(polys, list) or len(polys) == 0:
        return None

    try:
        H = int(item["H"])
        W = int(item["W"])
    except Exception:
        # fallback: if metadata lacks H/W, we still try with image size later
        H, W = 1024, 1024

    boxes: List[List[float]] = []
    for poly in polys:
        xyxy = polygon_to_xyxy(poly)
        if xyxy is None:
            continue
        boxes.append(clamp_xyxy(xyxy, w=W, h=H))

    if len(boxes) == 0:
        return None

    out = boxes[:k]
    while len(out) < k:
        out.append(out[-1])
    return out


def _count_from_rex_predictions(predictions: Any) -> int:
    if not isinstance(predictions, dict):
        return 0
    total = 0
    for _, pts in predictions.items():
        if isinstance(pts, list):
            total += len(pts)
    return int(total)


def main():
    ap = argparse.ArgumentParser("Rex-Omni visual-prompting evaluation on Kubric merged metadata (CountGD-style)")
    ap.add_argument("--metadata_path", required=True, type=str)
    ap.add_argument("--base_image_dir", default="", type=str)
    ap.add_argument("--output_dir", required=True, type=str)

    # Rex-Omni args
    ap.add_argument(
        "--model_code_dir",
        type=str,
        default=os.environ.get("REX_OMNI_MODEL_CODE_DIR", ""),
        help="Path to the Rex-Omni source directory. If omitted, imports must be available on PYTHONPATH.",
    )
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--backend", default="vllm", type=str)
    ap.add_argument("--max_tokens", default=4096, type=int)
    ap.add_argument("--temperature", default=0.0, type=float)
    ap.add_argument("--top_p", default=0.05, type=float)
    ap.add_argument("--top_k", default=1, type=int)
    ap.add_argument("--repetition_penalty", default=1.05, type=float)

    # Eval args
    ap.add_argument("--seed", default=42, type=int, help="Kept for CLI compatibility; exemplar selection uses the first k boxes.")
    ap.add_argument("--num_exemplars", default=3, type=int)
    ap.add_argument("--max_items", default=None, type=int)

    # Optional: store per-item raw predictions (may be large)
    ap.add_argument("--save_raw_predictions", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))
    from rex_omni import RexOmniWrapper

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"metadata must be a JSON list, got {type(data)}")
    if args.max_items is not None:
        data = data[: int(args.max_items)]

    # Init Rex-Omni (batch size=1 as requested)
    rex_model = RexOmniWrapper(
        model_path=args.model_path,
        backend=args.backend,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
    )

    overall = Meter()
    by_split: Dict[str, Meter] = {}
    by_level: Dict[int, Meter] = {}
    by_split_level2mode: Dict[Tuple[str, str], Meter] = {}

    per_item_path = os.path.join(args.output_dir, "per_item.jsonl")
    fout = open(per_item_path, "w", encoding="utf-8")

    num_ok = 0
    num_fail = 0

    for idx, it in enumerate(data):
        image_path = resolve_image_path(it, args.base_image_dir)
        split, level, level2_mode = get_group_keys(it)

        gt = float(it.get("count", 0.0))
        category = str(it.get("category", "")).strip()

        w = 2.0 if level == 1 else 1.0

        pred = 0.0
        success = False
        err_msg: Optional[str] = None
        pred_points = None
        raw_predictions = None
        prompt_boxes = None

        try:
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("Missing image_id")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            if level <= 0:
                raise ValueError(f"Invalid level={level}")

            prompt_boxes = select_first_k_exemplars(it, k=int(args.num_exemplars))
            if not prompt_boxes:
                raise ValueError("No valid box_examples_coordinates to build visual prompts")

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Rex-Omni visual prompting inference
            results = rex_model.inference(
                images=image,
                task="visual_prompting",
                visual_prompt_boxes=prompt_boxes,  # expects pixel xyxy
            )

            if not isinstance(results, list) or len(results) == 0:
                raise RuntimeError("Empty inference results")

            r0 = results[0]
            if not isinstance(r0, dict):
                raise RuntimeError(f"Unexpected result type: {type(r0)}")

            if bool(r0.get("success", False)):
                raw_predictions = r0.get("extracted_predictions", {})
                pred = float(_count_from_rex_predictions(raw_predictions))
                pred_points = r0.get("pred_points", None)  # if provided by your wrapper; else stays None
                success = True
                num_ok += 1
            else:
                err_msg = str(r0.get("error", "Unknown inference error"))
                # failure counted as pred=0
                pred = 0.0
                success = False
                num_fail += 1

        except Exception as e:
            err_msg = str(e)
            pred = 0.0
            success = False
            num_fail += 1

        overall.add(pred, gt, weight=w)
        by_split.setdefault(split, Meter()).add(pred, gt, weight=w)
        by_level.setdefault(level, Meter()).add(pred, gt, weight=w)
        if level == 2:
            by_split_level2mode.setdefault((split, level2_mode), Meter()).add(pred, gt, weight=w)

        rec: Dict[str, Any] = {
            "idx": int(idx),
            "image_id": image_path,
            "split": split,
            "level": int(level),
            "level2_mode": (level2_mode if level == 2 else None),
            "category": category,
            "gt": float(gt),
            "pred": float(pred),
            "abs_err": float(abs(pred - gt)),
            "weight": float(w),
            "success": bool(success),
            "error": err_msg,
            "prompt_boxes": prompt_boxes,
        }
        if pred_points is not None:
            rec["pred_points"] = pred_points
        if args.save_raw_predictions:
            rec["raw_predictions"] = raw_predictions

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (idx + 1) % 50 == 0:
            print(
                f"[{idx+1}/{len(data)}] ok={num_ok} fail={num_fail} "
                f"overall MAE={overall.mae():.4f} RMSE={overall.rmse():.4f}"
            )

    fout.close()

    report = {
        "metadata_path": os.path.abspath(args.metadata_path),
        "model_path": os.path.abspath(args.model_path),
        "backend": args.backend,
        "num_exemplars": int(args.num_exemplars),
        "exemplar_selection": "first_k",
        "max_items": (None if args.max_items is None else int(args.max_items)),
        "success": {"ok": int(num_ok), "fail": int(num_fail), "total": int(len(data))},
        "overall": overall.to_dict(),
        "by_split": {k: v.to_dict() for k, v in by_split.items()},
        "by_level": {str(k): v.to_dict() for k, v in by_level.items()},
        "by_split_level2mode": {f"{k[0]}/L2/{k[1]}": v.to_dict() for k, v in by_split_level2mode.items()},
    }

    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saved:", out_json)
    print("Saved:", per_item_path)
    print("Overall:", report["overall"])


if __name__ == "__main__":
    main()
