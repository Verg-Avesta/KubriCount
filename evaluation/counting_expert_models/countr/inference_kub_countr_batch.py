import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm


class measure_time(object):
    def __enter__(self):
        import time
        self._time = time
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = (self._time.perf_counter_ns() - self.start) / 1e9


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


def _resize_like_demo(pil_img: Image.Image, new_h: int = 384):
    # Match demo: resize so height=384, width rounded down to multiple of 16
    W, H = pil_img.size
    new_W = 16 * int((W / H * new_h) / 16)
    scale_h = float(new_h) / H
    scale_w = float(new_W) / W
    img = transforms.Resize((new_h, new_W))(pil_img)
    img = transforms.ToTensor()(img)  # no normalization in demo
    return img, scale_w, scale_h


def _boxes4pts_to_xyxy_abs(box_examples_coordinates):
    boxes = []
    for b in box_examples_coordinates:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append([x1, y1, x2, y2])
    return boxes


def _build_exemplar_patches(image_t, boxes_xyxy_abs, scale_w, scale_h, max_ex=3):
    """
    image_t: [3,H',W'] resized tensor (H'=384)
    boxes_xyxy_abs: list[[x1,y1,x2,y2]] in ORIGINAL pixel coords
    Return:
      boxes: Tensor [max_ex, 3, 64, 64]
      pos: list[[y1,x1,y2,x2]] in RESIZED image pixel coords (inclusive coords like demo)
      valid: number of real exemplars before pad
    """
    Hn, Wn = image_t.shape[1], image_t.shape[2]

    # Scale to resized coordinates, then crop patches
    pos = []
    patches = []
    for (x1, y1, x2, y2) in boxes_xyxy_abs[:max_ex]:
        rx1 = int(x1 * scale_w)
        ry1 = int(y1 * scale_h)
        rx2 = int(x2 * scale_w)
        ry2 = int(y2 * scale_h)

        # Clamp to valid
        rx1 = max(0, min(rx1, Wn - 1))
        rx2 = max(0, min(rx2, Wn - 1))
        ry1 = max(0, min(ry1, Hn - 1))
        ry2 = max(0, min(ry2, Hn - 1))
        if rx2 < rx1:
            rx1, rx2 = rx2, rx1
        if ry2 < ry1:
            ry1, ry2 = ry2, ry1

        pos.append([ry1, rx1, ry2, rx2])

        # demo uses y1:y2+1, x1:x2+1 (inclusive)
        patch = image_t[:, ry1:ry2 + 1, rx1:rx2 + 1]
        patch = transforms.Resize((64, 64))(patch)
        patches.append(patch)

    valid = len(patches)
    if valid == 0:
        return None, None, 0

    # Pad to max_ex by repeating last
    while len(patches) < max_ex:
        patches.append(patches[-1].clone())
        pos.append(pos[-1][:])

    boxes = torch.stack(patches, dim=0)  # [3,3,64,64]
    return boxes, pos, valid


def _run_sliding_window(samples, boxes, model, device):
    """
    samples: [1,3,H,W] (H=384), boxes: [1,3,3,64,64]
    Return density_map [H,W] on device.
    """
    _, _, h, w = samples.shape
    density_map = torch.zeros([h, w], device=device)
    start = 0
    prev = -1

    with torch.no_grad():
        while start + 383 < w:
            output, = model(samples[:, :, :, start:start + 384], boxes, 3)
            output = output.squeeze(0)  # [H,384] ?

            b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
            d1 = b1(output[:, 0:prev - start + 1])
            b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
            d2 = b2(output[:, prev - start + 1:384])

            b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
            density_map_l = b3(density_map[:, 0:start])
            density_map_m = b1(density_map[:, start:prev + 1])
            b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
            density_map_r = b4(density_map[:, prev + 1:w])

            density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

            prev = start + 383
            start = start + 128
            if start + 383 >= w:
                if start == w - 384 + 128:
                    break
                start = w - 384

    return density_map


def run_one_image_like_demo(samples, boxes, pos, model, device):
    """
    Returns pred_cnt (float) and elapsed time object.
    """
    _, _, h, w = samples.shape

    # Decide 9-grid path if any exemplar is too small (<10 both dims)
    s_cnt = 0
    for rect in pos:
        if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
            s_cnt += 1

    with measure_time() as et:
        if s_cnt >= 1:
            r_images = [
                TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)),
                TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)),
                TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)),
                TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)),
                TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)),
                TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)),
                TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)),
                TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)),
                TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)),
            ]

            pred_cnt = 0.0
            density_map_last = None
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map_last = _run_sliding_window(r_image, boxes, model, device)
                pred_cnt += torch.sum(density_map_last / 60.0).item()
            density_map = density_map_last
        else:
            density_map = _run_sliding_window(samples, boxes, model, device)
            pred_cnt = torch.sum(density_map / 60.0).item()

    # Exemplar-region correction (as demo)
    e_cnt = 0.0
    for rect in pos:
        e_cnt += torch.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60.0).item()
    e_cnt = e_cnt / 3.0
    if e_cnt > 1.8:
        pred_cnt /= e_cnt

    return float(pred_cnt), et


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, required=True)
    ap.add_argument("--base_image_dir", type=str, default="")
    ap.add_argument(
        "--model_code_dir",
        type=str,
        default=os.environ.get("COUNTR_MODEL_CODE_DIR", ""),
        help="Path to the CounTR source directory. If omitted, imports must be available on PYTHONPATH.",
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint-XXX.pth")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_objects", type=int, default=3)
    ap.add_argument("--group_by", type=str, default="split_level",
                    choices=["level", "split_level", "split_level_mode"])
    ap.add_argument("--double_level1", action="store_true")
    ap.add_argument("--max_items", type=int, default=-1)
    args = ap.parse_args()

    if args.model_code_dir:
        sys.path.insert(0, os.path.abspath(args.model_code_dir))
    import models_mae_cross

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = models_mae_cross.__dict__["mae_vit_base_patch16"](norm_pix_loss="store_true")
    model.to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    with open(args.metadata, "r") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    if args.max_items > 0:
        items = items[: args.max_items]

    stats = defaultdict(lambda: {"n": 0, "ae": 0.0, "se": 0.0})
    total = {"n": 0, "ae": 0.0, "se": 0.0}

    for item in tqdm(items, desc="CounTR inference", total=len(items), dynamic_ncols=True):
        image_path = _resolve_image_path(item, args.base_image_dir)
        gt_count = float(item["count"])

        boxes_raw = item.get("box_examples_coordinates", None) or []
        if len(boxes_raw) == 0:
            pred_count = 0.0
        else:
            pil = Image.open(image_path).convert("RGB")
            img_t, scale_w, scale_h = _resize_like_demo(pil, new_h=384)

            boxes_xyxy_abs = _boxes4pts_to_xyxy_abs(boxes_raw)
            boxes_t, pos, valid = _build_exemplar_patches(
                img_t, boxes_xyxy_abs, scale_w, scale_h, max_ex=int(args.num_objects)
            )
            if valid == 0:
                pred_count = 0.0
            else:
                samples = img_t.unsqueeze(0).to(device, non_blocking=True)
                boxes = boxes_t.unsqueeze(0).to(device, non_blocking=True)
                pred_count, _ = run_one_image_like_demo(samples, boxes, pos, model, device)

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
    print(f"ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
