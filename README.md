# KubriCount: Count Anything at Any Granularity

Official code release and dataset-generation pipeline for **Count Anything at Any Granularity**.

[🏡 Project Page](https://verg-avesta.github.io/KubriCount/) | [📄 Paper](https://arxiv.org/abs/2605.10887) | [🤗 Dataset](https://huggingface.co/datasets/liuchang666/KubriCount) | [📦 Assets & Docker Images](https://huggingface.co/datasets/liuchang666/KubriCount-assets)

KubriCount is a large-scale synthetic benchmark for **multi-grained visual counting**. The project targets open-world counting settings where the intended counting granularity must be explicit: identity, attribute, category, instance type, or concept. This repository provides the code used to construct KubriCount: controllable 3D synthesis, mask-conditioned image editing, and VLM-based filtering for dense instance-level supervision with controlled distractors.

## Highlights

- **Multi-grained counting benchmark** with five explicit semantic levels.
- **Fully automatic data scaling pipeline** built around 3D asset curation, Kubric-based prototype synthesis, consistent image editing, and automatic quality filtering.
- **Dense annotations** including counts, center points, 2D/3D boxes, masks, and metadata.
- **Large-scale dataset** with 110,507 images, 157 categories, about 7.3M annotated objects, and up to 250 objects per image.
- **Controlled generalization splits** covering seen categories, unseen assets, and unseen categories.

## Dataset

The released KubriCount dataset is available on Hugging Face:

[🤗 liuchang666/KubriCount](https://huggingface.co/datasets/liuchang666/KubriCount)

The dataset can be used directly and does **not** require running the generation pipeline in this repository. The pipeline is provided for reproducibility and future dataset construction.

Download the dataset using the Hugging Face CLI:

```bash
pip install -U huggingface_hub

hf download liuchang666/KubriCount \
  --repo-type dataset \
  --local-dir ./KubriCount
```

After downloading and extracting the dataset shards, the restored dataset should be placed under:

```text
KubriCount/
```

KubriCount contains five counting levels with train/test splits designed for controlled generalization:

- `train`: about 100K images from seen categories, excluding held-out TestA assets
- `testA`: about 5K images with unseen assets within seen categories
- `testB`: about 5K images with unseen categories

For Levels 2-5, each image can define two counting queries by swapping the target and distractor groups, yielding about 198K queries in total. The benchmark includes counts, center points, 2D/3D boxes, masks, and metadata for multi-grained evaluation.

## External Resources for Data Generation

The following resources are only required to reproduce or extend the data generation pipeline. They are not needed when using the released KubriCount dataset directly.

The HDRI backgrounds, Trellis-generated 3D assets, and prebuilt Docker images are hosted in a separate Hugging Face repository:

[🤗 liuchang666/KubriCount-assets](https://huggingface.co/datasets/liuchang666/KubriCount-assets)

The resource repository contains six archive files:

| File | Type | Description |
| --- | --- | --- |
| `kubricdockerhub_kubruntu_cpu.tar` | Docker image | Docker image for running Kubric on CPU. |
| `kubruntu-gpu.tar` | Docker image | Docker image for running Kubric on GPU. |
| `shapenet.tar` | Docker image | Docker image for ShapeNet-to-Kubric conversion and Trellis asset preprocessing. |
| `HDRI_haven.zip` | Assets | HDRI backgrounds collected from Poly Haven. The archive includes `HDRI_haven/` and `HDRI_haven.json`. |
| `HDRI_t2l.zip` | Assets | Synthetic HDRI backgrounds generated with Text2Light. The archive includes `HDRI_t2l/` and `HDRI_t2l.json`. |
| `trellis.zip` | Assets | 3D assets generated with Trellis. |

### Download the Asset Archives

From the root of this repository, run:

```bash
pip install -U huggingface_hub
mkdir -p assets

hf download liuchang666/KubriCount-assets \
  HDRI_haven.zip HDRI_t2l.zip trellis.zip \
  --repo-type dataset \
  --local-dir ./assets
```

Extract the downloaded archives under `assets/`:

```bash
unzip assets/HDRI_haven.zip -d assets/
unzip assets/HDRI_t2l.zip -d assets/
unzip assets/trellis.zip -d assets/
```

After extraction, the resource layout should include:

```text
assets/
├── HDRI_haven.zip
├── HDRI_haven/
├── HDRI_haven.json
├── HDRI_t2l.zip
├── HDRI_t2l/
├── HDRI_t2l.json
├── trellis.zip
└── ...                 # Extracted Trellis assets
```

The ZIP archives may be removed after successful extraction if disk space is limited.

### Download the Docker Images

Download the three prebuilt Docker image archives into `docker-image/`:

```bash
mkdir -p docker-image

hf download liuchang666/KubriCount-assets \
  kubricdockerhub_kubruntu_cpu.tar \
  kubruntu-gpu.tar \
  shapenet.tar \
  --repo-type dataset \
  --local-dir ./docker-image
```

The downloaded files should follow this layout:

```text
docker-image/
├── kubricdockerhub_kubruntu_cpu.tar
├── kubruntu-gpu.tar
└── shapenet.tar
```

Load the Docker image required for your workflow:

```bash
# Kubric CPU environment
docker load -i docker-image/kubricdockerhub_kubruntu_cpu.tar

# Kubric GPU environment
docker load -i docker-image/kubruntu-gpu.tar

# ShapeNet conversion and Trellis preprocessing environment
docker load -i docker-image/shapenet.tar
```

You only need to load the Docker images required by the stage you intend to run.

### ShapeNet Assets

ShapeNet assets are **not redistributed** in the KubriCount resource repository due to licensing restrictions.

Please obtain ShapeNet from the [official ShapeNet website](https://shapenet.org/) and follow its license and terms of use. The `shapenet.tar` file in the resource repository is a **Docker image for preprocessing**; it does not contain the ShapeNet dataset itself.

The utilities under `shapenet2kubric/` and `scripts_urdf/` can be used to convert and preprocess separately downloaded ShapeNet assets.

## Generation Pipeline Overview

KubriCount is generated in four stages:

1. **3D asset curation**: build a categorized object asset bank from labeled 3D datasets and controllable 3D generation.
2. **Prototype synthesis**: use Kubric, PyBullet, and Blender to render controllable multi-object scenes with exact instance-level metadata.
3. **Consistent image editing**: improve visual realism while preserving object topology and annotations.
4. **Automatic data filtering**: use a VLM inspector to reject samples with layout drift, count changes, identity corruption, background hallucination, or severe artifacts.

## Granularity Levels

KubriCount defines five counting levels. Each level specifies a target set and, when applicable, a controlled distractor set that differs by one semantic factor in the hierarchy.

| Level | Granularity | Description |
| --- | --- | --- |
| L1 | Identity-level | Count all instances of a single object type. |
| L2 | Attribute-level | Count objects distinguished by size or color. |
| L3 | Category-level | Count one category while excluding another category. |
| L4 | Instance-level | Count one instance type within the same category. |
| L5 | Concept-level | Count a category/concept with multiple instance types and semantically plausible distractors. |

## Repository Layout

```text
.
├── kubric/              # Core Kubric-based rendering and simulation package
├── docker/              # Dockerfiles for building runtime environments
├── evaluation/          # Evaluation scripts for MLLMs and expert counting models
├── assets/              # Downloaded and extracted HDRI and 3D asset resources
├── docker-image/        # Downloaded prebuilt Docker image archives
├── KubriCount/          # Released dataset downloaded separately
├── scripts_urdf/        # Trellis asset preprocessing utilities
├── shapenet2kubric/     # ShapeNet-to-Kubric conversion utilities
├── config_dense.json    # Dense scene generation configuration
├── config_gpt.json      # Default scene generation configuration
├── render_level.py      # Main multi-grained scene generation script
├── run.sh               # CPU rendering entry point
└── run_gpu.sh           # GPU rendering entry point
```

## Quick Start

### Use the Released Dataset

Download KubriCount from Hugging Face:

```bash
pip install -U huggingface_hub

hf download liuchang666/KubriCount \
  --repo-type dataset \
  --local-dir ./KubriCount
```

After extracting the dataset shards, the restored directory should contain:

```text
KubriCount/
├── train/
├── testA/
└── testB/
```

No rendering assets, Docker images, or API credentials are needed for dataset-only use.

### Reproduce or Extend the Generation Pipeline

The full generation pipeline requires:

- HDRI backgrounds and 3D assets
- A Kubric-compatible Docker environment
- Separately obtained ShapeNet assets for ShapeNet-based generation or preprocessing
- API access for the image-editing and VLM-filtering stages

Download and extract the released HDRI and Trellis resources:

```bash
pip install -U huggingface_hub
mkdir -p assets

hf download liuchang666/KubriCount-assets \
  HDRI_haven.zip HDRI_t2l.zip trellis.zip \
  --repo-type dataset \
  --local-dir ./assets

unzip assets/HDRI_haven.zip -d assets/
unzip assets/HDRI_t2l.zip -d assets/
unzip assets/trellis.zip -d assets/
```

Download the prebuilt Docker image archives:

```bash
mkdir -p docker-image

hf download liuchang666/KubriCount-assets \
  kubricdockerhub_kubruntu_cpu.tar \
  kubruntu-gpu.tar \
  shapenet.tar \
  --repo-type dataset \
  --local-dir ./docker-image
```

Load the CPU or GPU Kubric image before rendering:

```bash
# CPU environment
docker load -i docker-image/kubricdockerhub_kubruntu_cpu.tar

# GPU environment
docker load -i docker-image/kubruntu-gpu.tar
```

See [External Resources for Data Generation](#external-resources-for-data-generation) for the complete resource list, preprocessing image, and ShapeNet licensing note.

Example CPU rendering command:

```bash
bash run.sh 1 1 train random config_gpt.json
```

Example GPU rendering command:

```bash
bash run_gpu.sh 1 all 1 train random config_gpt.json
```

Generated scenes are written under `KubriCount/`.

Post-processing and filtering scripts:

```bash
# Initial mask-conditioned image editing.
python banana_edit_level.py \
  --root_path KubriCount/train \
  --workers 20 \
  --overwrite

# Iterative re-editing for samples that need another editing pass.
python banana_edit_redo.py \
  --root_path KubriCount/train \
  --workers 20 \
  --retry_times 3

# Initial VLM-based PASS/FAIL filtering.
python gemini_filter.py \
  --root_path KubriCount/train \
  --workers 20 \
  --flush_every 1000

# Iterative re-checking after re-editing.
python gemini_filter_redo.py \
  --root_path KubriCount/train \
  --workers 20 \
  --flush_every 1000
```

### Evaluate MLLMs

MLLM evaluation scripts are available under `evaluation/mllm/`. They support API-based models and local Hugging Face vision-language models:

```bash
python evaluation/mllm/eval_api_models.py --help
python evaluation/mllm/eval_open_models.py --help
```

See [`evaluation/mllm/README.md`](evaluation/mllm/README.md) for setup and example commands.

### Evaluate Counting Expert Models

KubriCount inference adapters for FamNet, LOCA, CounTR, DAVE, GeCo, Rex-Omni, CountGD++, and CountGD are available under `evaluation/counting_expert_models/`. These adapters keep the original model imports but do not vendor third-party model code or checkpoints.

```bash
python evaluation/counting_expert_models/famnet/inference_kub_famnet_batch.py --help
python evaluation/counting_expert_models/loca/inference_kub_loca_batch.py --help
python evaluation/counting_expert_models/countr/inference_kub_countr_batch.py --help
python evaluation/counting_expert_models/dave/inference_kub_dave_batch.py --help
python evaluation/counting_expert_models/geco/inference_kub_geco_batch.py --help
python evaluation/counting_expert_models/rex_omni/inference_kub_rex_omni.py --help
python evaluation/counting_expert_models/countgdpp/inference_kub_countgdpp_batch.py --help
python evaluation/counting_expert_models/countgd/inference_kub_countgd_batch.py --help
```

See [`evaluation/counting_expert_models/README.md`](evaluation/counting_expert_models/README.md) for setup notes and example commands.

## Citation

If you find this project useful, please cite:

```bibtex
@article{liu2026count,
  title={Count Anything at Any Granularity},
  author={Liu, Chang and Wu, Haoning and Xie, Weidi},
  journal={arXiv preprint arXiv:2605.10887},
  year={2026}
}
```

## Acknowledgements

This project builds on the excellent [Kubric](https://github.com/google-research/kubric) data generation framework. We thank the Kubric authors and contributors for making their rendering and simulation infrastructure publicly available.

The released HDRI resources include backgrounds collected from Poly Haven and backgrounds generated with Text2Light. The released 3D asset resources include assets generated with Trellis.

## License

This repository includes code derived from Kubric and is released under the Apache License 2.0. See [LICENSE](LICENSE).

External datasets and third-party resources may be subject to their respective licenses. In particular, ShapeNet assets are not redistributed with KubriCount and must be obtained from the official ShapeNet website under its applicable terms.
