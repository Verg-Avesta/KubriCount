# MLLM Evaluation

This folder contains KubriCount evaluation code for multimodal large language models.

- `eval_api_models.py`: API-based models, including OpenAI-compatible GPT endpoints, Anthropic-compatible Claude endpoints, and Gemini.
- `eval_open_models.py`: local/open-source vision-language models loaded with Hugging Face Transformers.

Both scripts evaluate the same KubriCount metadata format and write:

- `all_results.json`: per-query predictions and raw model responses.
- `metrics.json`: weighted MAE/RMSE overall, by level, and by Level-2 mode.

Level 1 queries are weighted by 2.0 in the final aggregation. Levels 2-5 use weight 1.0 because those levels can provide two swapped target/distractor queries per image.

## Dataset Paths

After downloading the released dataset, place or link it as:

```text
KubriCount/
+-- merged_test_metadata.json
+-- testA/
+-- testB/
```

The examples below assume this layout. If your images are stored elsewhere, update `--base_image_dir`.

## API Models

Credentials should be provided through environment variables. Do not hard-code API keys or private gateway URLs in scripts.

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

python evaluation/mllm/eval_api_models.py \
  --provider openai \
  --model gpt-4o \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --output_dir eval_results_kubric_api \
  --num_workers 8
```

For a custom OpenAI-compatible endpoint:

```bash
python evaluation/mllm/eval_api_models.py \
  --provider openai \
  --base_url "https://your-openai-compatible-endpoint.example.com" \
  --model gpt-4o \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount
```

Claude:

```bash
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

python evaluation/mllm/eval_api_models.py \
  --provider anthropic \
  --model claude-sonnet-4-5-20250929 \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount
```

Gemini:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

python evaluation/mllm/eval_api_models.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount
```

Useful options:

- `--levels 1,2,3,4,5`: choose which levels to evaluate.
- `--max_items 100`: run a quick smoke test.
- `--save_interval 50`: periodically save partial results.
- `--record_base_url 1`: include the runtime endpoint in `metrics.json`; disabled by default to reduce accidental leakage of private gateway URLs.

## Local/Open-Source Models

Install model-specific dependencies first. Most models require recent `transformers`, `accelerate`, `torch`, `torchvision`, `Pillow`, `qwen-vl-utils`, and sometimes `flash-attn`. To use FlashAttention where supported, pass `--attn_implementation flash_attention_2`.

Example:

```bash
python evaluation/mllm/eval_open_models.py \
  --model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --model_name qwen2_5vl-7b \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --output_dir eval_results_kubric_open_models
```

For a quick run:

```bash
python evaluation/mllm/eval_open_models.py \
  --model_path /path/to/model \
  --model_name qwen2_5vl-7b \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --max_items 20
```

Supported `--model_name` values include Qwen2.5-VL, Qwen3-VL, InternVL 2.5/3/3.5, LLaVA, Llama vision, Kimi-VL, SpatialBot, Molmo, and Molmo2 variants listed in `eval_open_models.py`.

## Notes

- The scripts expect metadata to be a JSON list, such as `merged_test_metadata.json`.
- `image_id` may be absolute or relative to `--base_image_dir`.
- API keys are never written to output files.
- `metrics.json` omits `base_url` by default.
