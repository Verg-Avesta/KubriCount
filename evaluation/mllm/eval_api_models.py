#!/usr/bin/env python3
"""Evaluate API-based multimodal LLMs on KubriCount metadata.

Supported providers:
  - openai: OpenAI-compatible /v1/chat/completions vision API.
  - anthropic: Anthropic-compatible /v1/messages API.
  - gemini: Google Gemini-compatible generateContent API.

Credentials are intentionally read from command-line arguments or environment
variables. Do not commit API keys or private gateway URLs.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm


DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


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


def extract_last_int(text: str | None) -> int | None:
    if text is None:
        return None
    numbers = re.findall(r"(-?\d+)", str(text))
    if not numbers:
        return None
    try:
        return int(numbers[-1])
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

    def mae(self) -> float:
        return self.sum_abs / self.sum_w if self.sum_w > 0 else 0.0

    def rmse(self) -> float:
        return math.sqrt(self.sum_sq / self.sum_w) if self.sum_w > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "n_eff": float(self.sum_w),
            "mae": float(self.mae()),
            "rmse": float(self.rmse()),
        }


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
    tail = (
        "Answer with a single integer only. Do not include words, punctuation, "
        "reasoning steps, or explanations. If uncertain, guess one integer."
    )

    if level == 1:
        return f"Count all objects of category '{category}' in the image. {tail}"

    if level in (2, 3, 5):
        return (
            f"Count all objects of category '{category}' in the image and "
            f"ignore objects of category '{negative_category}'. {tail}"
        )

    if level == 4:
        pos_boxes = item.get("box_examples_coordinates", []) or []
        neg_boxes = item.get("negative_box_examples_coordinates", []) or []
        pos_xyxy = bbox_poly_to_xyxy(pos_boxes[0]) if len(pos_boxes) > 0 else None
        neg_xyxy = bbox_poly_to_xyxy(neg_boxes[0]) if len(neg_boxes) > 0 else None
        return (
            f"There are two different object types that share category name '{category}'. "
            f"Type A example bounding box: {fmt_xyxy(pos_xyxy)}. "
            f"Type B example bounding box: {fmt_xyxy(neg_xyxy)}. "
            f"Count ONLY Type A objects and ignore Type B objects. {tail}"
        )

    return f"Count all objects of category '{category}' in the image. {tail}"


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


def canonical_box_for_key(box_poly: Any) -> str:
    if not isinstance(box_poly, list) or len(box_poly) != 4:
        return "NA"
    points = []
    for point in box_poly:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            points.append((int(round(float(point[0]))), int(round(float(point[1])))))
        except Exception:
            continue
    if len(points) != 4:
        return "NA"
    return ",".join(f"{x}:{y}" for x, y in points)


def make_sample_key(item: dict[str, Any]) -> str:
    image_id = str(item.get("image_id", "")).strip()
    level = get_level(item)
    category = str(item.get("category", "")).strip()
    negative_category = str(item.get("negative_category", "")).strip()
    base = f"{image_id}|L{level}|pos={category}|neg={negative_category}"
    if level != 4:
        return base

    pos_boxes = item.get("box_examples_coordinates", []) or []
    neg_boxes = item.get("negative_box_examples_coordinates", []) or []
    pos0 = canonical_box_for_key(pos_boxes[0]) if len(pos_boxes) > 0 else "NA"
    neg0 = canonical_box_for_key(neg_boxes[0]) if len(neg_boxes) > 0 else "NA"
    return base + f"|pos_box0={pos0}|neg_box0={neg0}"


def image_to_data_uri(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_to_base64_block(image_path: str) -> dict[str, Any]:
    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/png"
    if ext in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif ext == ".webp":
        media_type = "image/webp"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    }


def image_to_gemini_inline_data(image_path: str) -> dict[str, Any]:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"inlineData": {"mimeType": mime, "data": b64}}


def should_retry(status_code: int | None) -> bool:
    return status_code is None or status_code in (408, 409, 425, 429, 500, 502, 503, 504)


def call_with_retry(
    fn,
    *,
    max_retries: int = 8,
    base_sleep_s: float = 1.0,
    max_sleep_s: float = 30.0,
) -> dict[str, Any]:
    last = None
    for attempt in range(max_retries + 1):
        try:
            out = fn()
            status_code = out.get("status_code")
            if should_retry(status_code):
                last = out
                if attempt >= max_retries:
                    return out
                sleep_s = min(max_sleep_s, base_sleep_s * (2**attempt))
                time.sleep(sleep_s * (0.8 + 0.4 * random.random()))
                continue
            return out
        except Exception as exc:
            last = {"status_code": None, "json": None, "text": f"exception: {exc!r}"}
            if attempt >= max_retries:
                return last
            sleep_s = min(max_sleep_s, base_sleep_s * (2**attempt))
            time.sleep(sleep_s * (0.8 + 0.4 * random.random()))
    return last if last is not None else {"status_code": None, "json": None, "text": "unknown failure"}


def post_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    timeout_s: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    return parse_http_response(response)


def post_anthropic(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    timeout_s: int,
    max_tokens: int,
    temperature: float,
    anthropic_version: str,
    disable_thinking: bool,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/messages"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [
            {
                "role": "user",
                "content": [
                    image_to_base64_block(image_path),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    return parse_http_response(response)


def post_gemini(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    timeout_s: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + f"/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}, image_to_gemini_inline_data(image_path)],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }
    response = requests.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    return parse_http_response(response)


def parse_http_response(response: requests.Response) -> dict[str, Any]:
    out: dict[str, Any] = {"status_code": response.status_code, "text": response.text}
    try:
        out["json"] = response.json() if response.content else None
    except Exception:
        out["json"] = None
    return out


def extract_response_text(provider: str, response_json: dict[str, Any] | None) -> str:
    if not isinstance(response_json, dict):
        return ""
    try:
        if provider == "openai":
            return str(response_json["choices"][0]["message"]["content"]).strip()
        if provider == "anthropic":
            parts = response_json.get("content", [])
            texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict) and part.get("type") == "text"]
            return "\n".join(texts).strip()
        if provider == "gemini":
            candidates = response_json.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict) and "text" in part]
            return "\n".join(texts).strip()
    except Exception:
        return ""
    return ""


def load_existing_results(results_path: str) -> tuple[list[dict[str, Any]], set[str]]:
    if not os.path.exists(results_path):
        return [], set()
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return [], set()
    keys = {str(row["sample_key"]) for row in data if isinstance(row, dict) and row.get("sample_key")}
    return data, keys


def compute_metrics_from_results(
    all_results: list[dict[str, Any]], level2_modes_report: list[str]
) -> dict[str, Any]:
    by_level: dict[int, WeightedMeter] = defaultdict(WeightedMeter)
    by_level_mode: dict[tuple[int, str], WeightedMeter] = defaultdict(WeightedMeter)
    overall_all = WeightedMeter()
    overall_no_l4 = WeightedMeter()

    for row in all_results:
        if not isinstance(row, dict):
            continue
        level = safe_int(row.get("level"), default=None)
        if level is None:
            continue
        pred = row.get("pred_count")
        gt = row.get("gt_count")
        if pred is None or gt is None:
            continue
        mode = row.get("level2_mode", "all") if level == 2 else "all"
        weight = row.get("weight")
        if weight is None:
            weight = weight_for_level(level)
        by_level[level].add(float(pred), float(gt), float(weight))
        by_level_mode[(level, str(mode))].add(float(pred), float(gt), float(weight))
        overall_all.add(float(pred), float(gt), float(weight))
        if level != 4:
            overall_no_l4.add(float(pred), float(gt), float(weight))

    metrics: dict[str, Any] = {
        "overall_all_levels": overall_all.to_dict(),
        "overall_excluding_level4": overall_no_l4.to_dict(),
        "by_level": {str(level): by_level[level].to_dict() for level in sorted(by_level)},
        "by_level2_mode": {},
    }
    if 2 in by_level:
        for mode in level2_modes_report:
            meter = by_level_mode.get((2, mode), WeightedMeter())
            metrics["by_level2_mode"][mode] = meter.to_dict()
    return metrics


def process_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_path = resolve_image_path(item, args.base_image_dir)
    level = get_level(item)
    mode = get_level2_mode(item) if level == 2 else "all"
    prompt = build_count_prompt(item)
    gt = item.get("count")
    if gt is None:
        gt = len(item.get("points", []) or [])
    gt = float(gt)

    base_row: dict[str, Any] = {
        "sample_key": make_sample_key(item),
        "image_id": item.get("image_id", ""),
        "image_path": image_path,
        "level": level,
        "level2_mode": mode,
        "category": item.get("category", ""),
        "negative_category": item.get("negative_category", ""),
        "gt_count": gt,
        "pred_count": None,
        "weight": weight_for_level(level),
        "prompt": prompt,
        "raw_response": "",
        "status_code": None,
        "api_error_text": "",
    }

    if not image_path or not os.path.exists(image_path):
        base_row["api_error_text"] = "missing image"
        return base_row

    def do_call() -> dict[str, Any]:
        common_kwargs = {
            "base_url": args.base_url,
            "api_key": args.api_key,
            "model": args.model,
            "prompt": prompt,
            "image_path": image_path,
            "timeout_s": args.timeout_s,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.provider == "openai":
            return post_openai_compatible(**common_kwargs)
        if args.provider == "anthropic":
            return post_anthropic(
                **common_kwargs,
                anthropic_version=args.anthropic_version,
                disable_thinking=bool(args.disable_thinking),
            )
        if args.provider == "gemini":
            return post_gemini(**common_kwargs)
        raise ValueError(f"Unsupported provider: {args.provider}")

    response = call_with_retry(do_call, max_retries=args.max_retries)
    text = extract_response_text(args.provider, response.get("json"))
    pred = extract_last_int(text)
    if pred is None:
        pred = args.default_pred

    base_row.update(
        {
            "pred_count": int(pred),
            "raw_response": text,
            "status_code": response.get("status_code"),
            "api_error_text": response.get("text", ""),
        }
    )
    return base_row


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate API MLLMs on KubriCount.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--base_image_dir", default="")
    parser.add_argument("--output_dir", default="./eval_results_kubric_api")
    parser.add_argument("--base_url", default=None, help="Provider endpoint. Defaults to the public provider URL.")
    parser.add_argument("--api_key", default=None, help="API key. Prefer the provider-specific environment variable.")
    parser.add_argument("--levels", default="1,2,3,4,5")
    parser.add_argument("--level2_modes", default="size,color")
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--timeout_s", type=int, default=180)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--default_pred", type=int, default=0)
    parser.add_argument("--anthropic_version", default="2023-06-01")
    parser.add_argument("--disable_thinking", type=int, default=1)
    parser.add_argument(
        "--record_base_url",
        type=int,
        default=0,
        help="Set to 1 to write the runtime base_url into metrics.json.",
    )
    args = parser.parse_args()

    args.base_url = args.base_url or DEFAULT_BASE_URLS[args.provider]
    env_name = API_KEY_ENV[args.provider]
    args.api_key = args.api_key or os.environ.get(env_name)
    if not args.api_key:
        raise ValueError(f"Missing API key. Pass --api_key or set {env_name}.")
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"{args.provider}_{args.model}".replace("/", "_")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    results_path = os.path.join(run_dir, "all_results.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    levels = parse_csv_ints(args.levels)
    level2_modes_report = parse_csv_strings(args.level2_modes)

    all_results, done_keys = load_existing_results(results_path)
    if done_keys:
        print(f"Resume enabled: found {len(done_keys)} completed samples in {results_path}")

    items = load_metadata_list(args.metadata_path)
    filtered = [item for item in items if get_level(item) in set(levels)]
    if args.max_items and args.max_items > 0:
        filtered = filtered[: args.max_items]
    todo = [item for item in filtered if make_sample_key(item) not in done_keys]
    print(f"Loaded {len(items)} metadata items, level-filtered {len(filtered)}, todo {len(todo)}")

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
        futures = [executor.submit(process_one, item, args) for item in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.provider} [{args.model}]"):
            result = future.result()
            key = str(result.get("sample_key", ""))
            if key and key in done_keys:
                continue
            all_results.append(result)
            if key:
                done_keys.add(key)
            completed_since_save += 1
            if args.save_interval > 0 and completed_since_save >= args.save_interval:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                completed_since_save = 0

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    metrics: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "metadata_path": args.metadata_path,
        "filters": {"levels": levels, "level2_modes_report": level2_modes_report},
        "num_results": len(all_results),
    }
    if args.record_base_url:
        metrics["base_url"] = args.base_url
    metrics.update(compute_metrics_from_results(all_results, level2_modes_report))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_metrics(metrics, level2_modes_report)
    print(f"Saved results: {results_path}")
    print(f"Saved metrics: {metrics_path}")


def print_metrics(metrics: dict[str, Any], level2_modes_report: list[str]) -> None:
    print("\n" + "=" * 90)
    print(f"Metrics for {metrics['provider']} model={metrics['model']}")
    print("=" * 90)
    for level_str in sorted(metrics.get("by_level", {}), key=lambda x: int(x)):
        meter = metrics["by_level"][level_str]
        print(f"Level {level_str} | N_eff={meter['n_eff']:8.1f} | MAE={meter['mae']:.4f} | RMSE={meter['rmse']:.4f}")
        if int(level_str) == 2:
            for mode in level2_modes_report:
                mode_meter = metrics.get("by_level2_mode", {}).get(mode, {"n_eff": 0.0, "mae": 0.0, "rmse": 0.0})
                print(
                    f"Level 2 / mode={mode:>5s} | N_eff={mode_meter['n_eff']:8.1f} "
                    f"| MAE={mode_meter['mae']:.4f} | RMSE={mode_meter['rmse']:.4f}"
                )
    overall = metrics["overall_all_levels"]
    no_l4 = metrics["overall_excluding_level4"]
    print("-" * 90)
    print(f"Overall (with L4) | N_eff={overall['n_eff']:8.1f} | MAE={overall['mae']:.4f} | RMSE={overall['rmse']:.4f}")
    print(f"Overall (no   L4) | N_eff={no_l4['n_eff']:8.1f} | MAE={no_l4['mae']:.4f} | RMSE={no_l4['rmse']:.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
