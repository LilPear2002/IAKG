# IAKG: Intent-Aware Knowledge Graph-enhanced Course Recommendation

This repository contains the PyTorch implementation of **IAKG**, an intent-aware knowledge graph enhanced course recommendation model built on top of the SSLRec framework.

The current code focuses on **intent-driven personalization** rather than a separate denoising or teacher-student distillation pipeline. In this implementation, user intent prototypes are used to reweight KG edges and to mix multiple intent-specific KG views during recommendation.

## Overview

IAKG combines three signals:

- **Collaborative filtering structure**: LightGCN is used to model user-item interaction patterns.
- **Intent-aware personalization**: a small set of learnable intent prototypes captures coarse user preference modes.
- **Knowledge graph enhancement**: an RGAT-style encoder propagates information over the KG, with edge weights guided by intent prototypes.

The final recommendation score is built from:

- a structural user-item representation from the interaction graph
- an intent-conditioned KG item representation

## Main Idea

For each intent prototype:

1. The model computes an intent-aware score for every KG edge.
2. The KG encoder propagates entity embeddings with those edge weights.
3. The resulting item embeddings are treated as one intent-specific KG view.

At inference time, a user's intent attention weights are used to combine these intent-specific KG views, producing a personalized item representation.

## Environment

```bash
conda create -n iakg python=3.10 -y
conda activate iakg
pip install numpy==1.26.4 scipy==1.11.4
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install PyYAML tqdm
```

## Code Structure

```text
.
|-- config/
|   |-- configurator.py
|   `-- modelconf/
|       `-- iakg.yml
|-- data_utils/
|   |-- data_handler_kg.py
|   `-- datasets_kg.py
|-- datasets/
|   `-- kg/
|       |-- mooccube_kg/
|       |-- mooper_kg/
|       `-- coco_kg/
|-- models/
|   |-- iakg.py
|   `-- loss_utils.py
|-- trainer/
|   |-- trainer.py
|   |-- metrics.py
|   `-- logger.py
`-- main.py
```

## Datasets

The repository includes three knowledge graph enhanced course recommendation datasets:

| Statistics | MOOCCube | MOOPer | COCO |
|------------|----------|--------|------|
| Users | 34,917 | 28,702 | 24,036 |
| Courses | 698 | 233 | 8,196 |
| Interactions | 273,397 | 267,849 | 374,065 |
| Density | 98.88% | 95.99% | 99.81% |
| Entities | 239,440 | 10,184 | 11,237 |
| Relations | 7 | 8 | 5 |
| Triplets | 739,344 | 26,315 | 104,983 |

## Train

```bash
python main.py --model iakg
```

You can choose a dataset with:

```bash
python main.py --model iakg --dataset mooper
python main.py --model iakg --dataset mooc
python main.py --model iakg --dataset coco
```

## Configuration

Edit `config/modelconf/iakg.yml` to adjust hyperparameters.

Key settings include:

- `embedding_size`
- `layer_num`
- `kg_layer_num`
- `intent_size`
- `reg_weight`
- `cl_weight`
- `temperature`

## Implementation Notes

- The current implementation does **not** include teacher-student distillation.
- The intent module is used to build personalized KG views and to weight item representations.
- If you want to emphasize the method in a paper, describing it as **intent-driven personalization with knowledge-graph reweighting** is more accurate than centering the narrative on denoising.

## Citation

If this work is helpful, please cite the paper and the SSLRec framework.
