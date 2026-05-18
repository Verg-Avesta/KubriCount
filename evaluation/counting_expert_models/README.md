# Counting Expert Model Inference

This folder contains the KubriCount inference adapters for eight counting expert baselines:

- `famnet/inference_kub_famnet_batch.py` for FamNet.
- `loca/inference_kub_loca_batch.py` for LOCA.
- `countr/inference_kub_countr_batch.py` for CounTR.
- `dave/inference_kub_dave_batch.py` for DAVE.
- `geco/inference_kub_geco_batch.py` for GeCo.
- `rex_omni/inference_kub_rex_omni.py` for Rex-Omni.
- `countgdpp/inference_kub_countgdpp_batch.py` for CountGD++.
- `countgd/inference_kub_countgd_batch.py` for CountGD.

Only the KubriCount inference adapters are included here. The original model code, checkpoints, and third-party assets are not copied into this repository. Download each upstream model repository and checkpoint separately, then pass its source directory with `--model_code_dir`.

## Important Reproducibility Note

Exact numerical results can be sensitive to implementation details such as PyTorch, torchvision, PIL, CUDA/cuDNN kernels, GPU architecture, image resizing behavior, and checkpoint-loading behavior. Small MAE/RMSE differences may therefore appear across machines or software versions even when using the same checkpoints and metadata.

 The KubriCount benchmark is robust to the choice of visual exemplars: in our checks, randomly sampling three exemplars and deterministically using the first three exemplars produced very close results. To make released evaluations easier to reproduce, these adapters consistently use the first three valid exemplar boxes from the metadata for visual prompting. 
 
 Minor numerical variations did not change the relative trends among the evaluated models.

## Metadata

The scripts expect KubriCount metadata as a JSON list. `image_id` may be either:

- an absolute image path; or
- a path relative to `--base_image_dir`.

Use `--double_level1` to match the KubriCount aggregation rule where Level 1 receives weight 2.

## FamNet

```bash
python evaluation/counting_expert_models/famnet/inference_kub_famnet_batch.py \
  --metadata KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --model_code_dir /path/to/LearningToCountEverything \
  --model_path /path/to/FamNet_Save1.pth \
  --device cuda \
  --num_objects 3 \
  --group_by level \
  --double_level1
```

The script preserves imports from the original FamNet code:

```python
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
```

## LOCA

```bash
python evaluation/counting_expert_models/loca/inference_kub_loca_batch.py \
  --metadata KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --model_code_dir /path/to/loca \
  --model_path /path/to/loca/ckpt \
  --model_name loca_few_shot \
  --image_size 512 \
  --num_objects 3 \
  --group_by level \
  --double_level1 \
  --swav_backbone \
  --pre_norm
```

If your LOCA checkout supports offline backbone weights, you can also pass:

```bash
--swav_weights /path/to/swav_800ep_pretrain.pth.tar \
--resnet_weights /path/to/resnet50-0676ba61.pth
```

The script preserves the original model import:

```python
from models.loca import build_model
```

## CounTR

```bash
python evaluation/counting_expert_models/countr/inference_kub_countr_batch.py \
  --metadata KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --model_code_dir /path/to/CounTR \
  --ckpt /path/to/FSC147.pth \
  --device cuda \
  --num_objects 3 \
  --group_by level \
  --double_level1
```

The script preserves the original model import:

```python
import models_mae_cross
```

## DAVE

```bash
python evaluation/counting_expert_models/dave/inference_kub_dave_batch.py \
  --metadata KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --model_code_dir /path/to/DAVE \
  --model_path /path/to/DAVE/ckpt \
  --model_name base_3_shot \
  --image_size 512 \
  --num_objects 3 \
  --group_by level \
  --double_level1 \
  --swav_backbone \
  --pre_norm \
  --use_query_pos_emb \
  --use_objectness \
  --use_appearance
```

If your DAVE checkout supports offline backbone weights, you can also pass:

```bash
--swav_weights /path/to/swav_800ep_pretrain.pth.tar \
--resnet_weights /path/to/resnet50-0676ba61.pth
```

The script preserves the original model import:

```python
from models.dave import build_model
```

## GeCo

```bash
python evaluation/counting_expert_models/geco/inference_kub_geco_batch.py \
  --metadata KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --model_code_dir /path/to/GeCo \
  --model_path /path/to/GeCo/ckpt \
  --model_name GeCo \
  --num_objects 3 \
  --group_by level \
  --double_level1 \
  --sam_vit_h_weights /path/to/sam_vit_h_4b8939.pth
```

The script preserves the original model imports:

```python
from models.geco_infer import build_model
from utils.data import resize_and_pad
```

## Rex-Omni

```bash
python evaluation/counting_expert_models/rex_omni/inference_kub_rex_omni.py \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --output_dir eval_results/rex_omni \
  --model_code_dir /path/to/Rex-Omni \
  --model_path /path/to/Rex-Omni/checkpoint-or-hf-dir \
  --backend vllm \
  --num_exemplars 3
```

The script preserves the original model import:

```python
from rex_omni import RexOmniWrapper
```

## CountGD++

CountGD++ supports two KubriCount inference modes:

- `--prompt_mode pos`: positive category text and positive exemplar boxes only.
- `--prompt_mode posneg`: positive category/exemplars plus negative category/exemplars.

Positive-only:

```bash
python evaluation/counting_expert_models/countgdpp/inference_kub_countgdpp_batch.py \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --output_dir eval_results/countgdpp_pos \
  --model_code_dir /path/to/CountGDPlusPlus \
  --pretrain_model_path /path/to/countgd_plusplus.pth \
  --prompt_mode pos \
  --batch_size 32 \
  --num_exemplars 3 \
  --conf_thresh 0.23 \
  --amp
```

Positive plus negative:

```bash
python evaluation/counting_expert_models/countgdpp/inference_kub_countgdpp_batch.py \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --output_dir eval_results/countgdpp_posneg \
  --model_code_dir /path/to/CountGDPlusPlus \
  --pretrain_model_path /path/to/countgd_plusplus.pth \
  --prompt_mode posneg \
  --batch_size 32 \
  --num_exemplars 3 \
  --conf_thresh 0.23 \
  --amp
```

The script preserves imports from the original CountGD++ code:

```python
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from models.GroundingDINO import groundingdino_app
```

## CountGD

```bash
python evaluation/counting_expert_models/countgd/inference_kub_countgd_batch.py \
  --metadata_path KubriCount/merged_test_metadata.json \
  --base_image_dir KubriCount \
  --output_dir eval_results/countgd \
  --model_code_dir /path/to/CountGD \
  -c /path/to/CountGD/config/cfg_fsc147_vit_b.py \
  --pretrain_model_path /path/to/countgd_checkpoint.pth \
  --batch_size 32 \
  --num_exemplars 3 \
  --box_threshold 0.23 \
  --text_threshold 0.0 \
  --options text_encoder_type=checkpoints/bert-base-uncased max_text_len=512
```

The script preserves imports from the original CountGD code:

```python
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from models.registry import MODULE_BUILD_FUNCS
```
