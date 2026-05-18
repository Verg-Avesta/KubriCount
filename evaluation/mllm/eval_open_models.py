#!/usr/bin/env python3
"""Evaluate local/open-source multimodal LLMs on KubriCount metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any

from tqdm import tqdm

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch
    import torchvision.transforms as T
    import transformers
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    torch = None
    T = None
    transformers = None
    Image = None
    process_vision_info = None
    InterpolationMode = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_NAMES = {
    "internvl2_5-1b",
    "internvl2_5-4b",
    "internvl2_5-8b",
    "internvl2_5-38b",
    "internvl2_5-78b",
    "internvl3-1b",
    "internvl3-8b",
    "internvl3-14b",
    "internvl3-38b",
    "internvl3-78b",
    "internvl3_5-1b",
    "internvl3_5-4b",
    "internvl3_5-8b",
    "internvl3_5-14b",
    "internvl3_5-38b",
}
MOLMO_NAMES = {"molmoe-1b-0924", "molmo-7b-o-0924", "molmo-7b-d-0924", "molmo-72b-0924"}
MOLMO2_NAMES = {"molmo2-8b", "molmo2-o-7b", "molmo2-4b"}
QWEN2_5_NAMES = {
    "qwen2_5vl-3b",
    "qwen2_5vl-7b",
    "qwen2_5vl-32b",
    "qwen2_5vl-72b",
    "spaceqwen-3b",
    "spacethinker-qwen2_5vl-3b",
    "SpaceR",
}
QWEN3_NAMES = {"qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b", "qwen3vl-30b-a3b", "qwen3vl-235b-a22b"}
LLAVA_NAMES = {"llava1_5-7b", "llava1_5-13b", "llava-ov-7b", "llava-ov-72b"}
LLAMA_NAMES = {"llama", "llama-cot"}
KIMI_NAMES = {"kimivl-3b", "kimivl-3b-thinking"}
SPATIALBOT_NAMES = {"spatialbot-3b"}
MODEL_NAMES = sorted(
    INTERNVL_NAMES
    | MOLMO_NAMES
    | MOLMO2_NAMES
    | QWEN2_5_NAMES
    | QWEN3_NAMES
    | LLAVA_NAMES
    | LLAMA_NAMES
    | KIMI_NAMES
    | SPATIALBOT_NAMES
)


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def load_metadata_list(metadata_path: str) -> list[dict[str, Any]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"metadata must be a JSON list, got: {type(data)}")
    return [item for item in data if isinstance(item, dict)]


def get_level(item: dict[str, Any]) -> int | None:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return safe_int(metadata.get("level"), default=None)


def get_level2_mode(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    mode = metadata.get("level2_mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    return "all"


def extract_first_int(text: str | None) -> int | None:
    if text is None:
        return None
    match = re.search(r"(-?\d+)", str(text).strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


class WeightedMeter:
    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_w = 0.0

    def add(self, pred: float, gt: float, weight: float = 1.0) -> None:
        err = float(pred) - float(gt)
        w = float(weight)
        self.sum_abs += abs(err) * w
        self.sum_sq += (err * err) * w
        self.sum_w += w

    def to_dict(self) -> dict[str, float]:
        if self.sum_w <= 0:
            return {"n_eff": 0.0, "mae": 0.0, "rmse": 0.0}
        return {
            "n_eff": float(self.sum_w),
            "mae": float(self.sum_abs / self.sum_w),
            "rmse": float(math.sqrt(self.sum_sq / self.sum_w)),
        }


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    area_threshold = 0.5 * image_size * image_size
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > area_threshold * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    sorted_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, sorted_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def load_internvl_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images]).to(torch.float16)


def bbox_poly_to_xyxy(poly: Any) -> list[float] | None:
    if not isinstance(poly, list) or len(poly) != 4:
        return None
    try:
        xs = [float(point[0]) for point in poly]
        ys = [float(point[1]) for point in poly]
    except Exception:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def fmt_xyxy(xyxy: list[float] | None) -> str:
    if not xyxy or len(xyxy) != 4:
        return "N/A"
    x1, y1, x2, y2 = xyxy
    return f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"


def build_count_prompt(item: dict[str, Any]) -> str:
    level = get_level(item)
    category = str(item.get("category", "")).strip()
    negative_category = str(item.get("negative_category", "")).strip()
    tail = "Directly output the total number as an integer only. Do not output any other words. If unsure, guess a number."

    if level == 1:
        return f"Please count all objects of category '{category}' in the image. {tail}"
    if level in (2, 3, 5):
        return (
            f"Please count all objects of category '{category}' in the image, "
            f"and ignore objects of category '{negative_category}'. {tail}"
        )
    if level == 4:
        pos_boxes = item.get("box_examples_coordinates", []) or []
        neg_boxes = item.get("negative_box_examples_coordinates", []) or []
        pos_xyxy = bbox_poly_to_xyxy(pos_boxes[0]) if len(pos_boxes) > 0 else None
        neg_xyxy = bbox_poly_to_xyxy(neg_boxes[0]) if len(neg_boxes) > 0 else None
        return (
            f"In the image there are two different types of objects that share the same category name '{category}'. "
            f"Type A has an example bounding box {fmt_xyxy(pos_xyxy)}. "
            f"Type B has an example bounding box {fmt_xyxy(neg_xyxy)}. "
            f"Please count ONLY Type A objects and ignore Type B objects. {tail}"
        )
    return f"Please count all objects of category '{category}' in the image. {tail}"


def resolve_image_path(item: dict[str, Any], base_image_dir: str = "") -> str:
    image_id = item.get("image_id", "")
    if not isinstance(image_id, str):
        return ""
    if os.path.isabs(image_id):
        return image_id
    if base_image_dir:
        return os.path.join(base_image_dir, image_id)
    return image_id


def weight_for_level(level: int | None) -> float:
    return 2.0 if level == 1 else 1.0


def load_model_and_components(args: argparse.Namespace):
    require_runtime_dependencies()
    model_name = args.model_name
    common_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        common_kwargs["attn_implementation"] = args.attn_implementation

    if model_name in INTERNVL_NAMES:
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float16, **common_kwargs)
        processor = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    elif model_name in LLAMA_NAMES:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.float16, **common_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    elif model_name in {"llava1_5-7b", "llava1_5-13b"}:
        cls = getattr(transformers, "LlavaForConditionalGeneration")
        model = cls.from_pretrained(args.model_path, torch_dtype=torch.float16, **common_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    elif model_name in {"llava-ov-7b", "llava-ov-72b"}:
        cls = getattr(transformers, "LlavaOnevisionForConditionalGeneration")
        model = cls.from_pretrained(args.model_path, torch_dtype=torch.float16, **common_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    elif model_name in QWEN2_5_NAMES:
        cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")
        model = cls.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, **common_kwargs)
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            use_fast=False,
            trust_remote_code=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    elif model_name in QWEN3_NAMES:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, **common_kwargs)
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            use_fast=False,
            trust_remote_code=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    elif model_name in KIMI_NAMES:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, **common_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    elif model_name in SPATIALBOT_NAMES:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, **common_kwargs)
        model.model.vision_tower.load_model()
        model.model.vision_tower.to(model.device)
        processor = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    elif model_name in MOLMO_NAMES:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    elif model_name in MOLMO2_NAMES:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, dtype="auto", device_map="auto")
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, trust_remote_code=True, dtype="auto", device_map="auto")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.eval()
    print_device_map(model, model_name)
    return model, processor


def print_device_map(model: torch.nn.Module, model_name: str) -> None:
    print(f"\n=== Device map for {model_name} ===")
    if hasattr(model, "hf_device_map"):
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")
    else:
        print(f"Model device: {next(model.parameters()).device}")
    print("=" * 40)


def load_images_for_model(image_path: str, model_name: str, internvl_max_num: int) -> list[Any]:
    if model_name in INTERNVL_NAMES:
        return [load_internvl_image(image_path, input_size=448, max_num=internvl_max_num)]
    image = Image.open(image_path).convert("RGB")
    if image.size[0] <= 3 or image.size[1] <= 3:
        image = image.resize((32, 32), Image.BICUBIC)
    return [image]


def generate_response(
    model: torch.nn.Module,
    processor: Any,
    *,
    model_name: str,
    image_path: str,
    images: list[Any],
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
) -> str:
    do_sample = temperature > 0
    use_cache = True
    device = next(model.parameters()).device
    assistant_prompt = ""

    with torch.no_grad():
        if model_name in INTERNVL_NAMES:
            input_text = f"<image>\n{assistant_prompt} {prompt}".strip()
            num_patches_list = [image.size(0) for image in images]
            images_cat = torch.cat(images, dim=0).to(device)
            response = model.chat(
                processor,
                images_cat,
                input_text,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
                generation_config={
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                },
            )
            return response.strip()

        if model_name in LLAMA_NAMES | LLAVA_NAMES:
            image_content = [{"type": "image"} for _ in images]
            messages = [
                {"role": "assistant", "content": [{"type": "text", "text": assistant_prompt}]},
                {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]},
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images, input_text, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=do_sample,
            )
            response = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            return strip_assistant_prefix(response)

        if model_name in QWEN2_5_NAMES | QWEN3_NAMES:
            messages = [
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]},
            ]
            input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[input_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=do_sample,
            )
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        if model_name in KIMI_NAMES:
            messages = [
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}]},
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            inputs = processor(images=images, text=input_text, return_tensors="pt", padding=True).to(device)
            output_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=do_sample,
            )
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        if model_name in SPATIALBOT_NAMES:
            image_placeholders = "\n".join(f"<image {i + 1}>" for i in range(len(images)))
            prompt_text = f"{assistant_prompt} USER: {image_placeholders}\n{prompt} ASSISTANT:"
            text_chunks = [processor(chunk).input_ids for chunk in prompt_text.split(image_placeholders)]
            input_ids_list = text_chunks[0]
            for i in range(len(images)):
                input_ids_list += [-(201 + i)]
            input_ids_list += text_chunks[1]
            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
            image_tensor = model.process_images(images, model.config).to(dtype=model.dtype, device=device)
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                repetition_penalty=1.0,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                do_sample=do_sample,
            )[0]
            return processor.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True).strip()

        if model_name in MOLMO_NAMES:
            proc_inputs = processor.process(images=images, text=prompt, return_tensors="pt")
            moved = {}
            for key, value in proc_inputs.items():
                if not torch.is_tensor(value):
                    continue
                if value.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    moved[key] = value.to(device=model.device, dtype=getattr(model, "dtype", torch.float16)).unsqueeze(0)
                else:
                    moved[key] = value.to(device=model.device).unsqueeze(0)
            gen_cfg = transformers.GenerationConfig(
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
                stop_strings="<|endoftext|>",
            )
            output = model.generate_from_batch(moved, gen_cfg, tokenizer=processor.tokenizer)
            generated_tokens = output[0, moved["input_ids"].size(1) :]
            return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if model_name in MOLMO2_NAMES:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + [{"type": "image", "image": image} for image in images],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=use_cache,
            )
            input_len = inputs["input_ids"].size(1)
            generated_tokens = generated_ids[0, input_len:]
            return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    raise ValueError(f"Unknown model_name in generate_response: {model_name}")


def strip_assistant_prefix(text: str) -> str:
    for marker in ("assistant", "ASSISTANT"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text


def compute_metrics(results: list[dict[str, Any]], level2_modes: list[str]) -> dict[str, Any]:
    by_level: dict[int, WeightedMeter] = defaultdict(WeightedMeter)
    by_level_mode: dict[tuple[int, str], WeightedMeter] = defaultdict(WeightedMeter)
    overall_all = WeightedMeter()
    overall_no_l4 = WeightedMeter()

    for row in results:
        level = safe_int(row.get("level"), default=None)
        pred = row.get("pred_count")
        gt = row.get("gt_count")
        if level is None or pred is None or gt is None:
            continue
        mode = row.get("level2_mode", "all") if level == 2 else "all"
        weight = float(row.get("weight", weight_for_level(level)))
        by_level[level].add(pred, gt, weight)
        by_level_mode[(level, str(mode))].add(pred, gt, weight)
        overall_all.add(pred, gt, weight)
        if level != 4:
            overall_no_l4.add(pred, gt, weight)

    metrics: dict[str, Any] = {
        "overall_all_levels": overall_all.to_dict(),
        "overall_excluding_level4": overall_no_l4.to_dict(),
        "by_level": {str(level): by_level[level].to_dict() for level in sorted(by_level)},
        "by_level2_mode": {},
    }
    if 2 in by_level:
        for mode in level2_modes:
            metrics["by_level2_mode"][mode] = by_level_mode.get((2, mode), WeightedMeter()).to_dict()
    return metrics


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local/open-source MLLMs on KubriCount.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True, choices=MODEL_NAMES)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--base_image_dir", default="")
    parser.add_argument("--output_dir", default="./eval_results_kubric_open_models")
    parser.add_argument("--levels", default="1,2,3,4,5")
    parser.add_argument("--level2_modes", default="size,color")
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--default_pred", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device_map", default="balanced")
    parser.add_argument("--attn_implementation", default="")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--internvl_max_num", type=int, default=32)
    return parser.parse_args()


def require_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing dependency for local model evaluation: "
            f"{_IMPORT_ERROR.name}. Install the packages in evaluation/mllm/requirements.txt "
            "and any model-specific extras such as flash-attn when needed."
        )


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    torch.manual_seed(42)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    levels = parse_csv_ints(args.levels)
    level2_modes = parse_csv_strings(args.level2_modes)
    run_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "all_results.json")
    metrics_path = os.path.join(run_dir, "metrics.json")

    model, processor = load_model_and_components(args)
    items = load_metadata_list(args.metadata_path)
    filtered = [item for item in items if get_level(item) in set(levels)]
    if args.max_items and args.max_items > 0:
        filtered = filtered[: args.max_items]
    print(f"Loaded {len(items)} metadata items, using {len(filtered)} after level filter: {levels}")

    all_results: list[dict[str, Any]] = []
    for idx, item in enumerate(tqdm(filtered, desc=f"Eval KubriCount [{args.model_name}]")):
        image_path = resolve_image_path(item, args.base_image_dir)
        if not image_path or not os.path.exists(image_path):
            continue

        level = get_level(item)
        mode = get_level2_mode(item) if level == 2 else "all"
        prompt = build_count_prompt(item)
        images = load_images_for_model(image_path, args.model_name, args.internvl_max_num)

        response = generate_response(
            model,
            processor,
            model_name=args.model_name,
            image_path=image_path,
            images=images,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
        )
        pred = extract_first_int(response)
        if pred is None:
            pred = args.default_pred

        gt = item.get("count")
        if gt is None:
            gt = len(item.get("points", []) or [])
        gt = float(gt)
        weight = weight_for_level(level)

        all_results.append(
            {
                "image_id": item.get("image_id", ""),
                "image_path": image_path,
                "level": level,
                "level2_mode": mode,
                "category": item.get("category", ""),
                "negative_category": item.get("negative_category", ""),
                "gt_count": gt,
                "pred_count": int(pred),
                "weight": weight,
                "prompt": prompt,
                "raw_response": response,
            }
        )

        if args.save_interval > 0 and (idx + 1) % args.save_interval == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    metrics = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "metadata_path": args.metadata_path,
        "filters": {"levels": levels, "level2_modes_report": level2_modes},
        "num_results": len(all_results),
    }
    metrics.update(compute_metrics(all_results, level2_modes))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_metrics(args.model_name, metrics, levels, level2_modes)
    print(f"Saved per-sample results: {results_path}")
    print(f"Saved metrics: {metrics_path}")


def print_metrics(model_name: str, metrics: dict[str, Any], levels: list[int], level2_modes: list[str]) -> None:
    print("\n" + "=" * 90)
    print(f"Metrics for {model_name}")
    print("=" * 90)
    for level_str in sorted(metrics.get("by_level", {}), key=lambda x: int(x)):
        meter = metrics["by_level"][level_str]
        print(f"Level {level_str} | N_eff={meter['n_eff']:8.1f} | MAE={meter['mae']:.4f} | RMSE={meter['rmse']:.4f}")
        if int(level_str) == 2:
            for mode in level2_modes:
                mode_meter = metrics.get("by_level2_mode", {}).get(mode, {"n_eff": 0.0, "mae": 0.0, "rmse": 0.0})
                print(
                    f"Level 2 / mode={mode:>5s} | N_eff={mode_meter['n_eff']:8.1f} "
                    f"| MAE={mode_meter['mae']:.4f} | RMSE={mode_meter['rmse']:.4f}"
                )
    overall = metrics["overall_all_levels"]
    no_l4 = metrics["overall_excluding_level4"]
    print("-" * 90)
    print(f"Overall (levels {levels}, with L4) | N_eff={overall['n_eff']:8.1f} | MAE={overall['mae']:.4f} | RMSE={overall['rmse']:.4f}")
    print(f"Overall (levels {levels}, no   L4) | N_eff={no_l4['n_eff']:8.1f} | MAE={no_l4['mae']:.4f} | RMSE={no_l4['rmse']:.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
