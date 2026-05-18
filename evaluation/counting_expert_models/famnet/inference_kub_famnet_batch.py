import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm


def _boxes_4pts_to_yxyx_abs(box_examples_coordinates):
    """
    Convert KUBRIC/FSC style 4-point boxes to FamNet format [y1, x1, y2, x2] in original pixels.
    Input box: [[x1,y1],[x1,y2],[x2,y2],[x2,y1]] (axis-aligned)
    """
    rects = []
    for b in box_examples_coordinates:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        rects.append([y1, x1, y2, x2])
    return rects


def _group_key(item, group_by: str):
    md = item.get("metadata", {}) or {}
    level = md.get("level", None)
    split = md.get("split", None)
    mode = md.get("mode", None)

    if group_by == "level":
        return (level,)
    if group_by == "split_level":
        return (split, level)
    if group_by == "split_level_mode":
        return (split, level, mode)
    raise ValueError(f"Unknown group_by: {group_by}")


def _resolve_image_path(item, base_image_dir: str = ""):
    image_id = item["image_id"]
    if os.path.isabs(image_id):
        return image_id
    if base_image_dir:
        return os.path.join(base_image_dir, image_id)
    return image_id


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, required=True)
    ap.add_argument("--base_image_dir", type=str, default="")
    ap.add_argument(
        "--model_code_dir",
        type=str,
        default=os.environ.get("FAMNET_MODEL_CODE_DIR", ""),
        help="Path to the LearningToCountEverything source directory. If omitted, imports must be available on PYTHONPATH.",
    )
    ap.add_argument("--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_objects", type=int, default=3, help="Max exemplars to use (use fewer if not available).")
    ap.add_argument("--group_by", type=str, default="split_level",
                    choices=["level", "split_level", "split_level_mode"])
    ap.add_argument("--double_level1", action="store_true")
    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "max"],
                    help="Pooling mode in CountRegressor (keep consistent with training).")
    args = ap.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))
    from model import CountRegressor, Resnet50FPN
    from utils import MAPS, Scales, Transform, extract_features

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # Build model (same as demo)
    resnet50_conv = Resnet50FPN().to(device)
    regressor = CountRegressor(6, pool=args.pool).to(device)

    sd = torch.load(args.model_path, map_location="cpu")
    regressor.load_state_dict(sd)
    resnet50_conv.eval()
    regressor.eval()

    with open(args.metadata, "r") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    if args.max_items > 0:
        items = items[: args.max_items]

    stats = defaultdict(lambda: {"n": 0, "ae": 0.0, "se": 0.0})
    total = {"n": 0, "ae": 0.0, "se": 0.0}

    for item in tqdm(items, desc="FamNet inference", total=len(items), dynamic_ncols=True):
        image_path = _resolve_image_path(item, args.base_image_dir)
        gt_count = float(item["count"])

        boxes_raw = item.get("box_examples_coordinates", None) or []
        if len(boxes_raw) == 0:
            pred_count = 0.0
        else:
            rects = _boxes_4pts_to_yxyx_abs(boxes_raw)
            rects = rects[: max(1, min(len(rects), int(args.num_objects)))]  # use up to num_objects

            # FamNet expects sample={'image': PIL, 'lines_boxes': [[y1,x1,y2,x2],...]}
            image = Image.open(image_path).convert("RGB")
            sample = {"image": image, "lines_boxes": rects}
            sample = Transform(sample)
            img_t, boxes_t = sample["image"], sample["boxes"]  # img_t: CxHxW, boxes_t: 1xMx5 ([0,y1,x1,y2,x2])

            img_t = img_t.to(device)
            boxes_t = boxes_t.to(device)

            features = extract_features(resnet50_conv, img_t.unsqueeze(0), boxes_t.unsqueeze(0), MAPS, Scales)
            output = regressor(features)
            pred_count = float(output.sum().item())

        err = pred_count - gt_count
        ae = abs(err)
        se = err * err

        gkey = _group_key(item, args.group_by)

        weight = 1
        if args.double_level1:
            md = item.get("metadata", {}) or {}
            if md.get("level", None) == 1:
                weight = 2

        stats[gkey]["n"] += weight
        stats[gkey]["ae"] += ae * weight
        stats[gkey]["se"] += se * weight

        total["n"] += weight
        total["ae"] += ae * weight
        total["se"] += se * weight

    def _fmt_group(k):
        if args.group_by == "level":
            return f"level={k[0]}"
        if args.group_by == "split_level":
            return f"split={k[0]} level={k[1]}"
        if args.group_by == "split_level_mode":
            return f"split={k[0]} level={k[1]} mode={k[2]}"
        return str(k)

    for k in sorted(stats.keys(), key=lambda x: tuple([-1 if v is None else v for v in x])):
        n = stats[k]["n"]
        mae = stats[k]["ae"] / max(1, n)
        rmse = math.sqrt(stats[k]["se"] / max(1, n))
        print(f"[{_fmt_group(k)}] n={n} MAE={mae:.4f} RMSE={rmse:.4f}")

    mae = total["ae"] / max(1, total["n"])
    rmse = math.sqrt(total["se"] / max(1, total["n"]))
    print(f"[TOTAL] n={total['n']} MAE={mae:.4f} RMSE={rmse:.4f}")
    print(f"famnet_ckpt={args.model_path}")


if __name__ == "__main__":
    main()
