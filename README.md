# KubriCount: Count Anything at Any Granularity

Official code release and dataset-generation pipeline for **Count Anything at Any Granularity**.

[🏡 Project Page](https://verg-avesta.github.io/KubriCount/) | [📄 Paper](https://arxiv.org/abs/2605.10887) | [🤗 Dataset](https://huggingface.co/datasets/liuchang666/KubriCount)

KubriCount is a large-scale synthetic benchmark for **multi-grained visual counting**. The project targets open-world counting settings where the intended counting granularity must be explicit: identity, attribute, category, instance type, or concept. This repository provides the code used to construct KubriCount: controllable 3D synthesis, mask-conditioned image editing, and VLM-based filtering for dense instance-level supervision with controlled distractors.

## Highlights

- **Multi-grained counting benchmark** with five explicit semantic levels.
- **Fully automatic data scaling pipeline** built around 3D asset curation, Kubric-based prototype synthesis, consistent image editing, and automatic quality filtering.
- **Dense annotations** including counts, center points, 2D/3D boxes, masks, and metadata.
- **Large-scale dataset** with 110,507 images, 157 categories, about 7.3M annotated objects, and up to 250 objects per image.
- **Controlled generalization splits** covering seen categories, unseen assets, and unseen categories.

## Dataset

The KubriCount dataset is available on Hugging Face:

https://huggingface.co/datasets/liuchang666/KubriCount

The dataset can be used directly and does **not** require running the generation pipeline in this repository. The pipeline is provided for reproducibility and future dataset construction.

After downloading or extracting the Hugging Face dataset, place it under:

```text
KubriCount/
```

KubriCount contains five counting levels with train/test splits designed for controlled generalization:

- `train`: about 100K images from seen categories, excluding held-out TestA assets
- `testA`: about 5K images with unseen assets within seen categories
- `testB`: about 5K images with unseen categories

For Levels 2-5, each image can define two counting queries by swapping the target and distractor groups, yielding about 198K queries in total. The benchmark includes counts, center points, 2D/3D boxes, masks, and metadata for multi-grained evaluation.

## External Resources for Data Generation

The following resources are only needed if you want to reproduce or extend the data generation pipeline. They are not required for using the released Hugging Face dataset.

Large generated resources are intentionally not tracked by git. This repository keeps placeholder directories:

```text
assets/         # 3D assets, HDRIs, and asset manifests; download link coming soon
docker-image/   # Prebuilt Docker image archives; download link coming soon
```

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
├── assets/              # Placeholder for external 3D assets and manifests
├── docker-image/        # Placeholder for external prebuilt Docker image archives
├── KubriCount/          # Placeholder for the Hugging Face dataset
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

Download KubriCount from Hugging Face and extract it into `KubriCount/`. No rendering assets, Docker images, or API credentials are needed for dataset-only use.

```text
KubriCount/
├── train/
├── testA/
└── testB/
```

### Reproduce or Extend the Generation Pipeline

The full generation pipeline requires external assets, a Kubric-compatible Docker environment, and API access for the image editing / VLM filtering stages. Prebuilt Docker image archives and asset bundles will be linked here once released.

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
python banana_edit_level.py --root_path KubriCount/train --workers 20 --overwrite

# Iterative re-editing for samples that need another editing pass.
python banana_edit_redo.py --root_path KubriCount/train --workers 20 --retry_times 3

# Initial VLM-based PASS/FAIL filtering.
python gemini_filter.py --root_path KubriCount/train --workers 20 --flush_every 1000

# Iterative re-checking after re-editing.
python gemini_filter_redo.py --root_path KubriCount/train --workers 20 --flush_every 1000
```

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

## License

This repository includes code derived from Kubric and is released under the Apache License 2.0. See [LICENSE](LICENSE).
