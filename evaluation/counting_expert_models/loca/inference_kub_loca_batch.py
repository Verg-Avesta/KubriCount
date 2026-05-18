import argparse
import json
import math
import os
import sys
from collections import defaultdict
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms as T


def _load_image_rgb(path):
    return Image.open(path).convert("RGB")


def _img_transform(image_size):
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size, image_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _boxes_kubric_to_xyxy_abs(box_examples_coordinates):
    """
    KubriCount metadata stores each exemplar box as four points:
      [[x1,y1],[x1,y2],[x2,y2],[x2,y1]] (axis-aligned)
    FSC147 is the same idea.
    Return: Tensor [N,4] in xyxy absolute pixel coords in the ORIGINAL image space.
    """
    boxes = []
    for b in box_examples_coordinates:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)


def _scale_boxes_to_model(boxes_xyxy_abs, orig_w, orig_h, image_size):
    """
    Match FSC147Dataset behavior:
      bboxes / [W,H,W,H] * image_size
    boxes_xyxy_abs: [N,4] in original pixel coords.
    return: [N,4] in resized image pixel coords (0..image_size).
    """
    scale = torch.tensor([image_size / orig_w, image_size / orig_h,
                          image_size / orig_w, image_size / orig_h], dtype=torch.float32)
    boxes = boxes_xyxy_abs * scale
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size)

    # Ensure x1<x2, y1<y2
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Avoid degenerate boxes
    min_side = 2.0
    boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + min_side)
    boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + min_side)
    boxes[:, 2] = boxes[:, 2].clamp(0, image_size)
    boxes[:, 3] = boxes[:, 3].clamp(0, image_size)
    return boxes


def _group_key(item, group_by):
    md = item.get("metadata", {}) or {}
    level = md.get("level", None)
    split = md.get("split", None)
    mode = md.get("mode", None)  # if you have size/color mode stored; else None

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
        default=os.environ.get("LOCA_MODEL_CODE_DIR", ""),
        help="Path to the LOCA source directory. If omitted, imports must be available on PYTHONPATH.",
    )
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--num_objects", type=int, default=3, help="How many exemplar boxes to use.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--group_by", type=str, default="split_level",
                    choices=["level", "split_level", "split_level_mode"])
    ap.add_argument("--double_level1", action="store_true",
                    help="If set, level==1 samples are counted twice in aggregation (KubriCount weighting).")
    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--swav_backbone", action="store_true")
    ap.add_argument("--pre_norm", action="store_true")
    ap.add_argument("--zero_shot", action="store_true",
                    help="If set, ignore bboxes count and run LoCA zero-shot mode (not recommended for this task).")

    # Optional offline backbone weights for reproducible runs without downloads.
    ap.add_argument("--resnet_weights", type=str, default="")
    ap.add_argument("--swav_weights", type=str, default="")

    args = ap.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))
    from models.loca import build_model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build LOCA with same args keys expected by build_model(args)
    # We emulate an args-like object by attaching attributes.
    class _A:  # simple namespace
        pass
    a = _A()
    a.image_size = args.image_size
    a.num_enc_layers = 3
    a.num_ope_iterative_steps = 3
    a.num_objects = args.num_objects
    a.zero_shot = args.zero_shot
    a.emb_dim = 256
    a.num_heads = 8
    a.kernel_dim = 3
    a.backbone = "resnet50"
    a.swav_backbone = args.swav_backbone
    a.backbone_lr = 0.0
    a.reduction = 8
    a.dropout = 0.1
    a.pre_norm = args.pre_norm

    # Optional offline weight paths. The LOCA checkout must support these arguments.
    a.resnet_weights = args.resnet_weights
    a.swav_weights = args.swav_weights

    model = build_model(a).to(device)

    ckpt_path = os.path.join(args.model_path, f"{args.model_name}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Be tolerant to DDP checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tfm = _img_transform(args.image_size)

    with open(args.metadata, "r") as f:
        data = json.load(f)

    # metadata could be list[dict] or dict[id->dict]; normalize to list
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = list(data)

    if args.max_items > 0:
        items = items[: args.max_items]

    # Metrics accumulators per group
    stats = defaultdict(lambda: {"n": 0, "ae": 0.0, "se": 0.0})
    total = {"n": 0, "ae": 0.0, "se": 0.0}

    for item in tqdm(items, desc="LoCA inference", total=len(items)):
        image_path = _resolve_image_path(item, args.base_image_dir)
        gt_count = float(item["count"])

        img_pil = _load_image_rgb(image_path)
        orig_w, orig_h = img_pil.size
        img = tfm(img_pil).unsqueeze(0).to(device)

        # Positive exemplars
        boxes_raw = item.get("box_examples_coordinates", None)
        if boxes_raw is None:
            boxes_raw = []

        # If no exemplars, skip model forward and treat prediction as 0
        if len(boxes_raw) == 0:
            pred_count = 0.0
        else:
            boxes_xyxy_abs = _boxes_kubric_to_xyxy_abs(boxes_raw)  # [M,4] in original pixels
            m = int(boxes_xyxy_abs.size(0))
            if m <= 0:
                pred_count = 0.0
            else:
                # Use up to args.num_objects, but pad (repeat last) to exactly args.num_objects
                n_target = int(args.num_objects)
                n_use = min(m, n_target)

                boxes_xyxy_abs = boxes_xyxy_abs[:n_use]
                boxes_xyxy_resized = _scale_boxes_to_model(boxes_xyxy_abs, orig_w, orig_h, args.image_size)

                if n_use < n_target:
                    pad = boxes_xyxy_resized[-1:].repeat(n_target - n_use, 1)
                    boxes_xyxy_resized = torch.cat([boxes_xyxy_resized, pad], dim=0)

                bboxes = boxes_xyxy_resized.unsqueeze(0).to(device)  # [1,n_target,4]

                pred_dmap, _ = model(img, bboxes)
                pred_count = float(pred_dmap.flatten(1).sum(dim=1).item())

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

    # Print results
    def _fmt_group(k):
        if args.group_by == "level":
            return f"level={k[0]}"
        if args.group_by == "split_level":
            return f"split={k[0]} level={k[1]}"
        if args.group_by == "split_level_mode":
            return f"split={k[0]} level={k[1]} mode={k[2]}"
        return str(k)

    keys_sorted = sorted(stats.keys(), key=lambda x: tuple([-1 if v is None else v for v in x]))
    for k in keys_sorted:
        n = stats[k]["n"]
        mae = stats[k]["ae"] / max(1, n)
        rmse = math.sqrt(stats[k]["se"] / max(1, n))
        print(f"[{_fmt_group(k)}] n={n} MAE={mae:.4f} RMSE={rmse:.4f}")

    mae = total["ae"] / max(1, total["n"])
    rmse = math.sqrt(total["se"] / max(1, total["n"]))
    print(f"[TOTAL] n={total['n']} MAE={mae:.4f} RMSE={rmse:.4f}")
    print(f"ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
