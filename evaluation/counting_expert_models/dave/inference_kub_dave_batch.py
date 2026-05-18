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


def _load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _img_transform(image_size: int):
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size, image_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _boxes_4pts_to_xyxy_abs(box_examples_coordinates):
    """
    KubriCount format (FSC147-like): each box is four axis-aligned points:
      [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
    Return: Tensor [N,4] in xyxy absolute pixel coords in ORIGINAL image space.
    """
    boxes = []
    for b in box_examples_coordinates:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)


def _scale_boxes_to_resized_space(boxes_xyxy_abs, orig_w, orig_h, image_size):
    """
    Match DAVE/FSC resize convention:
      bboxes / [W,H,W,H] * image_size
    Return: [N,4] in resized image pixel coords.
    """
    scale = torch.tensor(
        [image_size / orig_w, image_size / orig_h, image_size / orig_w, image_size / orig_h],
        dtype=torch.float32,
    )
    boxes = boxes_xyxy_abs * scale

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size)

    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Avoid degenerate boxes
    min_side = 2.0
    boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + min_side).clamp(0, image_size)
    boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + min_side).clamp(0, image_size)
    return boxes


def _pad_boxes_to_n(boxes_xyxy, n_target: int):
    """
    boxes_xyxy: [N,4]
    If N<n_target, repeat last box to reach n_target.
    """
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
        default=os.environ.get("DAVE_MODEL_CODE_DIR", ""),
        help="Path to the DAVE source directory. If omitted, imports must be available on PYTHONPATH.",
    )
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True, help="e.g. DAVE_3_shot or DAVE_0_shot")
    ap.add_argument("--verification_ckpt", type=str, default="verification.pth",
                    help="Path (relative to model_path or absolute) to verification.pth")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--num_objects", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--group_by", type=str, default="split_level",
                    choices=["level", "split_level", "split_level_mode"])
    ap.add_argument("--double_level1", action="store_true")
    ap.add_argument("--max_items", type=int, default=-1)

    # DAVE args that affect architecture
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--swav_backbone", action="store_true")
    ap.add_argument("--reduction", type=int, default=8)
    ap.add_argument("--pre_norm", action="store_true")
    ap.add_argument("--use_query_pos_emb", action="store_true")
    ap.add_argument("--use_objectness", action="store_true")
    ap.add_argument("--use_appearance", action="store_true")
    ap.add_argument("--zero_shot", action="store_true")
    ap.add_argument("--two_passes", action="store_true",
                    help="Match demo_zero.py behavior (optional); default is single pass.")

    # Optional offline backbone weights. The DAVE checkout must support these arguments.
    ap.add_argument("--resnet_weights", type=str, default="")
    ap.add_argument("--swav_weights", type=str, default="")

    args = ap.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))
    from models.dave import build_model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build args namespace expected by build_model(args)
    class _A:
        pass

    a = _A()
    a.model_name = args.model_name
    a.model_path = args.model_path
    a.backbone = args.backbone
    a.swav_backbone = args.swav_backbone
    a.reduction = args.reduction
    a.image_size = args.image_size
    a.fcos_pred_size = args.image_size
    a.num_enc_layers = 3
    a.num_dec_layers = 3
    a.emb_dim = 256
    a.num_heads = 8
    a.kernel_dim = 3
    a.dropout = 0.1
    a.backbone_lr = 0.0
    a.pre_norm = args.pre_norm
    a.num_objects = args.num_objects

    a.zero_shot = args.zero_shot
    a.prompt_shot = False
    a.use_query_pos_emb = args.use_query_pos_emb
    a.use_objectness = args.use_objectness
    a.use_appearance = args.use_appearance

    # thresholds default from your arg_parser.py
    a.d_s = 1.0
    a.m_s = 0.0
    a.i_thr = 0.55
    a.d_t = 3.0
    a.s_t = 0.008
    a.norm_s = False
    a.egv = 0.132
    a.det_train = False

    # Optional offline weights.
    a.resnet_weights = args.resnet_weights
    a.swav_weights = args.swav_weights

    model = build_model(a).to(device)

    ckpt_path = os.path.join(args.model_path, f"{args.model_name}.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Load verification feature-transform weights (as in demo/demo_zero)
    ver_path = args.verification_ckpt
    if not os.path.isabs(ver_path):
        ver_path = os.path.join(args.model_path, ver_path)
    ver = torch.load(ver_path, map_location="cpu")
    ver_sd = ver["model"] if isinstance(ver, dict) and "model" in ver else ver
    pretrained_dict_feat = {}
    for k, v in ver_sd.items():
        if "feat_comp" in k:
            # Accept both "feat_comp.xxx" and "module.feat_comp.xxx"
            kk = k.split("feat_comp.", 1)[1] if "feat_comp." in k else k
            kk = kk.replace("module.", "")
            pretrained_dict_feat[kk] = v
    if hasattr(model, "feat_comp") and pretrained_dict_feat:
        model.feat_comp.load_state_dict(pretrained_dict_feat, strict=True)

    model.eval()

    tfm = _img_transform(args.image_size)

    with open(args.metadata, "r") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    if args.max_items > 0:
        items = items[: args.max_items]

    stats = defaultdict(lambda: {"n": 0, "ae": 0.0, "se": 0.0})
    total = {"n": 0, "ae": 0.0, "se": 0.0}

    for item in tqdm(items, desc="DAVE inference", total=len(items), dynamic_ncols=True):
        image_path = _resolve_image_path(item, args.base_image_dir)
        gt_count = float(item["count"])

        img_pil = _load_image_rgb(image_path)
        orig_w, orig_h = img_pil.size
        img = tfm(img_pil).unsqueeze(0).to(device)

        boxes_raw = item.get("box_examples_coordinates", None) or []
        if len(boxes_raw) == 0:
            pred_count = 0.0
        else:
            boxes_xyxy_abs = _boxes_4pts_to_xyxy_abs(boxes_raw)
            if boxes_xyxy_abs.numel() == 0 or boxes_xyxy_abs.size(0) == 0:
                pred_count = 0.0
            else:
                # Take up to num_objects, then pad to num_objects (repeat last)
                n_target = int(args.num_objects)
                n_use = min(int(boxes_xyxy_abs.size(0)), n_target)
                boxes_xyxy_abs = boxes_xyxy_abs[:n_use]

                boxes_xyxy = _scale_boxes_to_resized_space(
                    boxes_xyxy_abs, orig_w=orig_w, orig_h=orig_h, image_size=args.image_size
                )
                boxes_xyxy = _pad_boxes_to_n(boxes_xyxy, n_target)

                bboxes = boxes_xyxy.unsqueeze(0).to(device)  # [1,3,4]

                # DAVE forward signature: forward(x_img, bboxes, name='', dmap=None, classes=None)
                density_map, _, _, predicted_bboxes = model(img, bboxes=bboxes, name=os.path.basename(image_path))

                # Optional second pass like demo_zero.py
                if args.two_passes and predicted_bboxes is not None and hasattr(predicted_bboxes, "box"):
                    boxes_pred = predicted_bboxes.box
                    if boxes_pred.numel() > 0:
                        # Reproduce demo_zero scaling heuristic (roughly)
                        scale_y = min(1.0, 50 / (boxes_pred[:, 2] - boxes_pred[:, 0]).mean().item())
                        scale_x = min(1.0, 50 / (boxes_pred[:, 3] - boxes_pred[:, 1]).mean().item())

                        if scale_x < 1.0 or scale_y < 1.0:
                            scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                            scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                        else:
                            scale_y = min(max(1.0, 11 / (boxes_pred[:, 2] - boxes_pred[:, 0]).mean().item()), 1.9)
                            scale_x = min(max(1.0, 11 / (boxes_pred[:, 3] - boxes_pred[:, 1]).mean().item()), 1.9)
                            scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                            scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size

                        if scale_x != 1.0 or scale_y != 1.0:
                            resize_ = T.Resize(
                                (int(args.image_size * scale_y), int(args.image_size * scale_x)),
                                antialias=True,
                            )
                            img_resized = resize_(img)
                            # pad_image exists in utils.data; avoid importing by reimplementing minimal padding
                            pad_h = max(args.image_size - img_resized.shape[2], 0)
                            pad_w = max(args.image_size - img_resized.shape[3], 0)
                            if pad_h > 0 or pad_w > 0:
                                img_resized = torch.nn.functional.pad(img_resized, (0, pad_w, 0, pad_h))
                            density_map, _, _, predicted_bboxes = model(img_resized, bboxes=bboxes, name=os.path.basename(image_path))

                pred_count = float(density_map.flatten(1).sum(dim=1).item())

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
    print(f"ckpt={ckpt_path}")
    print(f"verification_ckpt={ver_path}")


if __name__ == "__main__":
    main()
