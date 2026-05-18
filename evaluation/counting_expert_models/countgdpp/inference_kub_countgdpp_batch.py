#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch


# -------------------------
# Metrics
# -------------------------
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


# -------------------------
# KubriCount metadata helpers
# -------------------------
def polygon_to_xyxy(poly: Any) -> Optional[List[float]]:
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


def select_first_k_exemplars_from_key(
    item: Dict[str, Any],
    key: str,
    k: int,
) -> Optional[List[List[float]]]:
    polys = item.get(key, [])
    if not isinstance(polys, list) or len(polys) == 0:
        return None

    H = int(item["H"])
    W = int(item["W"])

    boxes = []
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


def parse_kv_options(kvs):
    if kvs is None:
        return None
    out = {}
    for raw in kvs:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid option '{item}', expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        lower = value.lower()
        if lower in ("true", "false"):
            parsed = lower == "true"
        else:
            try:
                parsed = int(value)
            except Exception:
                try:
                    parsed = float(value)
                except Exception:
                    parsed = value
        out[key] = parsed
    return out


# -------------------------
# CountGD++: build model & transforms (mirrors app.py)
# -------------------------
def get_args_parser_countgdpp():
    # Keep it aligned with app.py arg structure where needed.
    parser = argparse.ArgumentParser("CountGD++ batch eval", add_help=False)
    parser.add_argument("--device", default="cuda", help="device for inference")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override config options in k=v format",
    )
    parser.add_argument("--pretrain_model_path", default="checkpoints/countgd_plusplus.pth")
    parser.add_argument("--amp", action="store_true", help="mixed precision inference")
    return parser


def build_transform_countgdpp():
    # Must use datasets.transforms_app as in app.py (not datasets_inference.transforms)
    import datasets.transforms_app as T

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return T.Compose([T.RandomResize([800], max_size=1333), normalize])


def build_model_countgdpp(args) -> torch.nn.Module:
    from util.slconfig import SLConfig

    # app.py loads cfg_app.py and forces bert-base-uncased
    cfg = SLConfig.fromfile("cfg_app.py")

    # replicate the hard-coded merge in app.py
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})

    # allow user overrides (optional)
    opt_dict = parse_kv_options(getattr(args, "options", None))
    if opt_dict:
        cfg.merge_from_dict(opt_dict)

    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    # deterministic seeds (same as app.py)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    from models.GroundingDINO import groundingdino_app

    build_func = groundingdino_app.build_groundingdino
    model, _, _ = build_func(args)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval().to(torch.device(args.device))
    return model


# -------------------------
# CountGD++ prompt + prediction (mirrors app.py)
# -------------------------
def find_dot_split_idx(token_ids_1d: torch.Tensor) -> int:
    # '.' token id 1012 (as app.py assumes)
    for i in range(token_ids_1d.shape[0]):
        if int(token_ids_1d[i].item()) == 1012:
            return i
    return int(token_ids_1d.shape[0] - 1)


def count_from_model_output_pos_only(model_output: Dict[str, Any], conf_thresh: float) -> int:
    # For pos-only mode: just Stage1 filtering in app.py.
    input_ids = model_output["input_ids"][0]
    logits = model_output["pred_logits"].sigmoid()[0]  # (nq, L)
    dot_idx = find_dot_split_idx(input_ids)

    pos_logits = logits[:, : (dot_idx + 1)]
    box_mask = pos_logits.max(dim=-1).values > conf_thresh
    return int(box_mask.sum().item())


def count_from_model_output_posneg(model_output: Dict[str, Any], conf_thresh: float) -> int:
    # For pos+neg mode: Stage1 + Stage2 filtering in app.py.
    input_ids = model_output["input_ids"][0]
    logits = model_output["pred_logits"].sigmoid()[0]  # (nq, L)
    dot_idx = find_dot_split_idx(input_ids)

    pos_logits = logits[:, : (dot_idx + 1)]
    neg_logits = logits[:, (dot_idx + 1) :]

    # Stage 1: keep boxes with sufficient positive confidence
    box_mask = pos_logits.max(dim=-1).values > conf_thresh
    pos_logits = pos_logits[box_mask, :]
    neg_logits = neg_logits[box_mask, :]

    if pos_logits.numel() == 0:
        return 0

    # Stage 2: keep boxes where pos_max > neg_max
    box_mask2 = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
    return int(box_mask2.sum().item())


def run_countgdpp_single(
    model: torch.nn.Module,
    transform,
    device: torch.device,
    image: Image.Image,
    pos_text: str,
    pos_ex_boxes_xyxy: List[List[float]],
    prompt_mode: str,
    neg_text: Optional[str] = None,
    neg_ex_boxes_xyxy: Optional[List[List[float]]] = None,
    conf_thresh: float = 0.23,
    amp: bool = False,
) -> int:
    """
    Follow app.py's forward signature and caption format.
    """
    # Apply same transform pipeline to:
    # - input image (for main image branch)
    # - exemplar image branch (here we use the same image as exemplar image, like your CountGD script)
    #
    # app.py preprocess() calls:
    #   input_image, _ = transform(image, None)
    #   input_image_exemplar, exemplar = transform(image, {"exemplars": tensor(exemplar_boxes)})
    input_image, _ = transform(image, None)

    pos_ex_tensor = torch.tensor(pos_ex_boxes_xyxy, dtype=torch.float32)
    input_image_pos_ex, pos_target = transform(image, {"exemplars": pos_ex_tensor})
    pos_exemplars = [pos_target["exemplars"].to(device)]  # list length 1

    input_images = input_image.unsqueeze(0).to(device)  # (1,3,H,W)
    input_image_pos_exemplars = input_image_pos_ex.unsqueeze(0).to(device)

    # Caption construction MUST match app.py: "pos . neg1 . neg2 ."
    caption = (pos_text.strip() + " . ").strip()
    negative_images = []
    neg_exemplars = []
    neg_texts = []

    if prompt_mode == "posneg":
        if neg_text is None:
            neg_text = ""
        if neg_ex_boxes_xyxy is None:
            neg_ex_boxes_xyxy = []

        if len(neg_text.strip()) > 0 or len(neg_ex_boxes_xyxy) > 0:
            neg_texts.append(neg_text.strip())

            neg_ex_tensor = torch.tensor(neg_ex_boxes_xyxy, dtype=torch.float32) if len(neg_ex_boxes_xyxy) > 0 else torch.zeros((0, 4), dtype=torch.float32)
            input_image_neg_ex, neg_target = transform(image, {"exemplars": neg_ex_tensor})
            neg_ex = neg_target["exemplars"].to(device)

            from util.misc import nested_tensor_from_tensor_list

            negative_images.append(
                nested_tensor_from_tensor_list(input_image_neg_ex.unsqueeze(0).to(device))
            )
            neg_exemplars.append([neg_ex])

        for t in neg_texts:
            caption = caption + t + " . "

    # Wrap like app.py:
    # model(
    #   nested_tensor_from_tensor_list(input_images),
    #   nested_tensor_from_tensor_list(input_image_pos_exemplars),
    #   pos_exemplars,
    #   negative_images,
    #   neg_exemplars,
    #   captions=[caption],
    # )
    with torch.no_grad():
        from util.misc import nested_tensor_from_tensor_list

        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            if prompt_mode == "pos":
                model_output = model(
                    nested_tensor_from_tensor_list(input_images),
                    nested_tensor_from_tensor_list(input_image_pos_exemplars),
                    pos_exemplars,
                    [],  # no negatives
                    [],  # no negatives
                    captions=[caption],
                )
                return count_from_model_output_pos_only(model_output, conf_thresh=conf_thresh)

            if prompt_mode == "posneg":
                model_output = model(
                    nested_tensor_from_tensor_list(input_images),
                    nested_tensor_from_tensor_list(input_image_pos_exemplars),
                    pos_exemplars,
                    negative_images,
                    neg_exemplars,
                    captions=[caption],
                )
                return count_from_model_output_posneg(model_output, conf_thresh=conf_thresh)

            raise ValueError(f"Unknown prompt_mode={prompt_mode}")


# -------------------------
# Main batch loop (keeps your dataset grouping/weighting)
# -------------------------
def main():
    parent = get_args_parser_countgdpp()
    ap = argparse.ArgumentParser("CountGD++ batch evaluation on Kubric metadata", parents=[parent])

    ap.add_argument("--metadata_path", required=True, type=str)
    ap.add_argument("--base_image_dir", default="", type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument(
        "--model_code_dir",
        type=str,
        default=os.environ.get("COUNTGDPP_MODEL_CODE_DIR", ""),
        help="Path to the CountGD++ source directory. If omitted, imports must be available on PYTHONPATH.",
    )

    ap.add_argument("--seed", default=42, type=int, help="Kept for CLI compatibility; exemplar selection uses the first k boxes.")
    ap.add_argument("--num_exemplars", default=3, type=int)

    ap.add_argument("--batch_size", default=1, type=int)  # CountGD++ forward is easiest per-sample; keep 1 by default
    ap.add_argument("--max_items", default=None, type=int)

    ap.add_argument("--conf_thresh", default=0.23, type=float)

    ap.add_argument(
        "--prompt_mode",
        default="posneg",
        choices=["pos", "posneg"],
        help="pos: only positive text+exemplars; posneg: positive+negative text+exemplars",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))

    device = torch.device(args.device)
    model = build_model_countgdpp(args)
    transform = build_transform_countgdpp()

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"metadata must be a JSON list, got {type(data)}")
    if args.max_items is not None:
        data = data[: int(args.max_items)]

    overall = Meter()
    by_split: Dict[str, Meter] = {}
    by_level: Dict[int, Meter] = {}
    by_split_level2mode: Dict[Tuple[str, str], Meter] = {}

    per_item_path = os.path.join(args.output_dir, "per_item.jsonl")
    fout = open(per_item_path, "w", encoding="utf-8")

    # Keep batching interface similar, but we’ll run per sample inside batch
    bs = int(args.batch_size)
    for start in range(0, len(data), bs):
        batch = data[start : start + bs]
        for it in batch:
            image_path = resolve_image_path(it, args.base_image_dir)
            if not isinstance(image_path, str) or not os.path.exists(image_path):
                continue

            split, level, level2_mode = get_group_keys(it)
            if level <= 0:
                continue

            pos_cat = str(it.get("category", "")).strip()
            if not pos_cat:
                continue

            pos_boxes = select_first_k_exemplars_from_key(
                it, key="box_examples_coordinates", k=args.num_exemplars
            )
            if pos_boxes is None:
                continue

            neg_cat = str(it.get("negative_category", "")).strip()
            neg_boxes = None
            if args.prompt_mode == "posneg":
                neg_boxes = select_first_k_exemplars_from_key(
                    it, key="negative_box_examples_coordinates", k=args.num_exemplars
                )
                # Allow negative text without boxes.
                if neg_boxes is None:
                    neg_boxes = []

            image = Image.open(image_path).convert("RGB")

            pred = run_countgdpp_single(
                model=model,
                transform=transform,
                device=device,
                image=image,
                pos_text=pos_cat,
                pos_ex_boxes_xyxy=pos_boxes,
                prompt_mode=args.prompt_mode,
                neg_text=neg_cat if args.prompt_mode == "posneg" else None,
                neg_ex_boxes_xyxy=neg_boxes if args.prompt_mode == "posneg" else None,
                conf_thresh=float(args.conf_thresh),
                amp=bool(args.amp),
            )

            gt = float(it.get("count", 0))

            # KubriCount aggregation rule.
            w = 2.0 if level == 1 else 1.0

            overall.add(pred, gt, weight=w)
            by_split.setdefault(split, Meter()).add(pred, gt, weight=w)
            by_level.setdefault(level, Meter()).add(pred, gt, weight=w)
            if level == 2:
                by_split_level2mode.setdefault((split, level2_mode), Meter()).add(pred, gt, weight=w)

            fout.write(
                json.dumps(
                    {
                        "image_id": image_path,
                        "split": split,
                        "level": level,
                        "level2_mode": level2_mode if level == 2 else None,
                        "prompt_mode": args.prompt_mode,
                        "positive_category": pos_cat,
                        "negative_category": neg_cat if args.prompt_mode == "posneg" else None,
                        "gt": gt,
                        "pred": float(pred),
                        "abs_err": abs(float(pred) - gt),
                        "weight": w,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if (start // bs + 1) % 10 == 0:
            print(f"[batch {start//bs+1}] overall MAE={overall.mae():.4f} RMSE={overall.rmse():.4f}")

    fout.close()

    report = {
        "metadata_path": os.path.abspath(args.metadata_path),
        "model_ckpt": os.path.abspath(args.pretrain_model_path),
        "conf_thresh": float(args.conf_thresh),
        "num_exemplars": int(args.num_exemplars),
        "exemplar_selection": "first_k",
        "batch_size": int(args.batch_size),
        "prompt_mode": args.prompt_mode,
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
