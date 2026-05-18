import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm


def _load_image_tensor(path: str) -> torch.Tensor:
    # demo.py uses T.ToTensor()(PIL) before resize_and_pad
    return T.ToTensor()(Image.open(path).convert("RGB"))


def _boxes_4pts_to_xyxy_abs(box_examples_coordinates):
    boxes = []
    for b in box_examples_coordinates:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)


def _pad_boxes_to_n(boxes_xyxy, n_target: int):
    n = int(boxes_xyxy.size(0))
    if n >= n_target:
        return boxes_xyxy[:n_target]
    pad = boxes_xyxy[-1:].repeat(n_target - n, 1)
    return torch.cat([boxes_xyxy, pad], dim=0)


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
        default=os.environ.get("GECO_MODEL_CODE_DIR", ""),
        help="Path to the GeCo source directory. If omitted, imports must be available on PYTHONPATH.",
    )
    ap.add_argument("--model_path", type=str, required=True, help="Directory containing GeCo.pth or checkpoint file.")
    ap.add_argument("--model_name", type=str, default="GeCo", help="Checkpoint stem; expects <model_name>.pth")
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--num_objects", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--group_by", type=str, default="split_level",
                    choices=["level", "split_level", "split_level_mode"])
    ap.add_argument("--double_level1", action="store_true")
    ap.add_argument("--max_items", type=int, default=-1)

    # GeCo args / switches
    ap.add_argument("--reduction", type=int, default=16)
    ap.add_argument("--zero_shot", action="store_true")
    ap.add_argument("--output_masks", action="store_true")

    # Offline SAM ViT-H weights
    ap.add_argument("--sam_vit_h_weights", type=str, default="")

    args = ap.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))
    from models.geco_infer import build_model
    from utils.data import resize_and_pad

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build args namespace expected by build_model(args)
    class _A:
        pass

    a = _A()
    a.model_name = args.model_name
    a.model_path = args.model_path
    a.image_size = args.image_size
    a.num_objects = args.num_objects
    a.emb_dim = 256
    a.num_heads = 8
    a.kernel_dim = 1
    a.backbone_lr = 0.0
    a.reduction = args.reduction
    a.zero_shot = args.zero_shot
    a.output_masks = args.output_masks
    a.sam_vit_h_weights = args.sam_vit_h_weights

    model = build_model(a).to(device)

    ckpt_path = os.path.join(args.model_path, f"{args.model_name}.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with open(args.metadata, "r") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    if args.max_items > 0:
        items = items[: args.max_items]

    stats = defaultdict(lambda: {"n": 0, "ae": 0.0, "se": 0.0})
    total = {"n": 0, "ae": 0.0, "se": 0.0}

    for item in tqdm(items, desc="GeCo inference", total=len(items), dynamic_ncols=True):
        image_path = _resolve_image_path(item, args.base_image_dir)
        gt_count = float(item["count"])

        boxes_raw = item.get("box_examples_coordinates", None) or []
        if len(boxes_raw) == 0 and not args.zero_shot:
            pred_count = 0.0
        else:
            image = _load_image_tensor(image_path)

            if args.zero_shot:
                # GeCo forward still expects bboxes; give dummy padded boxes (content shouldn't matter in zero-shot)
                b = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
                b = _pad_boxes_to_n(b, args.num_objects)
            else:
                b = _boxes_4pts_to_xyxy_abs(boxes_raw)
                if b.numel() == 0 or b.size(0) == 0:
                    pred_count = 0.0
                    b = None
                else:
                    # take up to N then pad to num_objects
                    n_use = min(int(b.size(0)), int(args.num_objects))
                    b = b[:n_use]
                    b = _pad_boxes_to_n(b, int(args.num_objects))

            if b is None:
                pred_count = 0.0
            else:
                # Follow demo: resize_and_pad before normalize
                img, bboxes, scale = resize_and_pad(image, b, full_stretch=False)
                img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                img = img.unsqueeze(0).to(device)
                bboxes = bboxes.unsqueeze(0).to(device)

                outputs, _, _, _, _masks = model(img, bboxes)

                # outputs[i]["pred_boxes"] after refine is normalized by img.shape[-1]
                # So count = number of predicted boxes in outputs[0]["pred_boxes"][0]
                if outputs is None or len(outputs) == 0:
                    pred_count = 0.0
                else:
                    pb = outputs[0].get("pred_boxes", None)
                    if pb is None or pb.numel() == 0:
                        pred_count = 0.0
                    else:
                    # Mirror the demo NMS/thresholding selection:
                        # keep boxes with box_v > max/4 then NMS.
                        # If box_v missing, fallback to all boxes.
                        box_v = outputs[0].get("box_v", None)
                        if box_v is None or box_v.numel() == 0:
                            pred_count = float(pb.shape[1]) if pb.dim() == 3 else float(pb.shape[0])
                        else:
                            thr = 4.0
                            mask = (box_v > (box_v.max() / thr))
                            pb_sel = pb[mask]
                            scores_sel = box_v[mask]
                            if pb_sel.numel() == 0:
                                pred_count = 0.0
                            else:
                                keep = torch.ops.torchvision.nms(pb_sel, scores_sel, 0.5)
                                pred_count = float(len(keep))

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
    print(f"ckpt={os.path.join(args.model_path, f'{args.model_name}.pth')}")
    if args.sam_vit_h_weights:
        print(f"sam_vit_h_weights={args.sam_vit_h_weights}")


if __name__ == "__main__":
    main()
