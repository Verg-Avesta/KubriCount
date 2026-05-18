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

import torch


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


def _parse_kv_options(kvs):
    if kvs is None:
        return None
    if isinstance(kvs, dict):
        return kvs
    if not isinstance(kvs, (list, tuple)):
        raise TypeError(f"--options must be list/tuple of k=v, got {type(kvs)}")

    out = {}
    for s in kvs:
        s = str(s).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Invalid option '{s}', expected k=v")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()

        vl = v.lower()
        if vl in ("true", "false"):
            vv = (vl == "true")
        else:
            try:
                vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v
        out[k] = vv
    return out


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


def resolve_image_path(item: Dict[str, Any], base_image_dir: str = "") -> str:
    image_id = item.get("image_id", "")
    if not isinstance(image_id, str):
        return ""
    if os.path.isabs(image_id):
        return image_id
    if base_image_dir:
        return os.path.join(base_image_dir, image_id)
    return image_id


def build_transform():
    import datasets_inference.transforms as T

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose([T.RandomResize([800], max_size=1333), normalize])


def find_end_idx(token_ids_1d: torch.Tensor) -> int:
    # '.' token id 1012 in your codebase
    for i in range(token_ids_1d.shape[0]):
        if int(token_ids_1d[i].item()) == 1012:
            return i
    return int(token_ids_1d.shape[0])


def predict_counts_batch(outputs: Dict[str, Any], box_threshold: float, text_threshold: float) -> List[int]:
    """
    outputs["pred_logits"]: (bs, nq, max_text_len)
    outputs["token"]["input_ids"]: (bs, seq_len)
    Return: list[int] length bs
    """
    logits = outputs["pred_logits"].sigmoid()  # (bs, nq, L)
    input_ids = outputs["token"]["input_ids"]  # (bs, T)

    bs = logits.shape[0]
    preds = []
    for b in range(bs):
        sample_logits = logits[b]  # (nq, L)
        end_idx = find_end_idx(input_ids[b])
        lo = 1
        hi = max(lo + 1, int(end_idx))

        box_mask = sample_logits.max(dim=-1).values > box_threshold
        sample_logits = sample_logits[box_mask, :]
        if sample_logits.numel() == 0:
            preds.append(0)
            continue

        text_mask = (sample_logits[:, lo:hi] > text_threshold).sum(dim=-1) == (hi - lo)
        sample_logits = sample_logits[text_mask, :]
        preds.append(int(sample_logits.shape[0]))
    return preds


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


def select_first_k_exemplars(item: Dict[str, Any], k: int) -> Optional[List[List[float]]]:
    polys = item.get("box_examples_coordinates", [])
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


def build_model(args) -> torch.nn.Module:
    from util.slconfig import SLConfig
    from models.registry import MODULE_BUILD_FUNCS

    cfg = SLConfig.fromfile(args.config)
    opt_dict = _parse_kv_options(args.options)
    if opt_dict:
        cfg.merge_from_dict(opt_dict)

    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(args, k) or getattr(args, k) in (None, ""):
            setattr(args, k, v)

    if not hasattr(args, "modelname") or not args.modelname:
        args.modelname = "groundingdino"

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    if build_func is None:
        raise ValueError(f"Unknown modelname={args.modelname}")

    model, _, _ = build_func(args)
    ckpt = torch.load(args.pretrain_model_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(torch.device(args.device))
    return model


def main():
    ap = argparse.ArgumentParser("CountGD batch evaluation on Kubric old-style metadata")
    ap.add_argument("--config", "-c", required=True, type=str)
    ap.add_argument("--options", nargs="+", default=None)
    ap.add_argument("--pretrain_model_path", required=True, type=str)
    ap.add_argument("--metadata_path", required=True, type=str)
    ap.add_argument("--base_image_dir", default="", type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument(
        "--model_code_dir",
        type=str,
        default=os.environ.get("COUNTGD_MODEL_CODE_DIR", ""),
        help="Path to the CountGD source directory. If omitted, imports must be available on PYTHONPATH.",
    )

    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--seed", default=42, type=int, help="Kept for CLI compatibility; exemplar selection uses the first k boxes.")
    ap.add_argument("--num_exemplars", default=3, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--num_workers", default=0, type=int)  # kept for compatibility, not used

    ap.add_argument("--box_threshold", default=0.23, type=float)
    ap.add_argument("--text_threshold", default=0.0, type=float)

    ap.add_argument("--max_items", default=None, type=int)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))

    model = build_model(args)
    transform = build_transform()
    device = torch.device(args.device)

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"metadata must be a JSON list, got {type(data)}")
    if args.max_items is not None:
        data = data[: int(args.max_items)]

    overall = Meter()
    by_split: Dict[str, Meter] = {}
    by_level: Dict[int, Meter] = {}
    by_split_level: Dict[Tuple[str, int], Meter] = {}
    by_split_level2mode: Dict[Tuple[str, str], Meter] = {}

    per_item_path = os.path.join(args.output_dir, "per_item.jsonl")
    fout = open(per_item_path, "w", encoding="utf-8")

    def process_batch(batch_items: List[Dict[str, Any]]):
        images_t: List[torch.Tensor] = []
        exemplars_t: List[torch.Tensor] = []
        captions: List[str] = []
        gts: List[float] = []
        weights: List[float] = []
        groups: List[Tuple[str, int, str]] = []
        image_ids: List[str] = []
        categories: List[str] = []

        for it in batch_items:
            image_path = resolve_image_path(it, args.base_image_dir)
            if not isinstance(image_path, str) or not os.path.exists(image_path):
                continue

            split, level, level2_mode = get_group_keys(it)
            if level <= 0:
                continue

            category = str(it.get("category", "")).strip()
            caption = (category + " .") if category else " ."

            ex_boxes = select_first_k_exemplars(it, k=args.num_exemplars)
            if ex_boxes is None:
                continue

            # load image
            image = Image.open(image_path).convert("RGB")

            # transform with exemplars
            ex_tensor = torch.tensor(ex_boxes, dtype=torch.float32)
            img_t, target = transform(image, {"exemplars": ex_tensor})
            images_t.append(img_t)
            exemplars_t.append(target["exemplars"])
            captions.append(caption)

            gt = float(it.get("count", 0))
            gts.append(gt)

            w = 2.0 if level == 1 else 1.0
            weights.append(w)
            groups.append((split, level, level2_mode))
            image_ids.append(image_path)
            categories.append(category)

        if len(images_t) == 0:
            return

        from util.misc import nested_tensor_from_tensor_list

        samples = nested_tensor_from_tensor_list(images_t).to(device)
        exemplars_t = [ex.to(device) for ex in exemplars_t]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(getattr(args, "amp", False))):
                outputs = model(
                    samples,
                    exemplars_t,
                    [torch.tensor([0], device=device) for _ in exemplars_t],
                    captions=captions,
                )

        pred_counts = predict_counts_batch(outputs, args.box_threshold, args.text_threshold)

        for i in range(len(pred_counts)):
            pred = float(pred_counts[i])
            gt = float(gts[i])
            w = float(weights[i])
            split, level, level2_mode = groups[i]

            overall.add(pred, gt, weight=w)
            by_split.setdefault(split, Meter()).add(pred, gt, weight=w)
            by_level.setdefault(level, Meter()).add(pred, gt, weight=w)
            by_split_level.setdefault((split, level), Meter()).add(pred, gt, weight=w)
            if level == 2:
                by_split_level2mode.setdefault((split, level2_mode), Meter()).add(pred, gt, weight=w)

            fout.write(
                json.dumps(
                    {
                        "image_id": image_ids[i],
                        "split": split,
                        "level": level,
                        "level2_mode": level2_mode if level == 2 else None,
                        "category": categories[i],
                        "gt": gt,
                        "pred": pred,
                        "abs_err": abs(pred - gt),
                        "weight": w,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # main loop: simple batching over list (keeps code minimal)
    bs = int(args.batch_size)
    for start in range(0, len(data), bs):
        process_batch(data[start : start + bs])
        if (start // bs + 1) % 10 == 0:
            print(f"[batch {start//bs+1}] overall MAE={overall.mae():.4f} RMSE={overall.rmse():.4f}")

    fout.close()

    report = {
        "metadata_path": os.path.abspath(args.metadata_path),
        "model_ckpt": os.path.abspath(args.pretrain_model_path),
        "config": os.path.abspath(args.config),
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "num_exemplars": int(args.num_exemplars),
        "exemplar_selection": "first_k",
        "batch_size": int(args.batch_size),
        "overall": overall.to_dict(),
        "by_split": {k: v.to_dict() for k, v in by_split.items()},
        "by_level": {str(k): v.to_dict() for k, v in by_level.items()},
        "by_split_level": {f"{k[0]}/L{k[1]}": v.to_dict() for k, v in by_split_level.items()},
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
