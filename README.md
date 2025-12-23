# IACD: Intent-Aware Collaborative Denoising for Knowledge Graph-Enhanced Course Recommendation

Official PyTorch implementation of **IACD** (Intent-Aware Collaborative Denoising), a novel knowledge graph-enhanced recommendation model with collaborative knowledge distillation.This implementation is built upon the [SSLRec](https://github.com/HKUDS/SSLRec) framework. We thank the authors for their excellent work.

## 🔮 Overview

IACD addresses the challenge of noisy knowledge graph edges in recommendation systems through:
- **Intent-Aware Denoising**: User intent-guided edge weight learning
- **Collaborative Knowledge Distillation**: Student-teacher framework for KG denoising
- **Relational Graph Attention**: RGAT-based knowledge graph propagation
- **Contrastive Learning**: Multi-view self-supervised learning for enhanced representations

## 📝 Environment

You can run the following commands to create a conda environment:

```
conda create -n iacd python=3.10 -y
conda activate iacd
pip install numpy==1.26.4 scipy==1.11.4
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install PyYAML tqdm
```

😉 The codes are developed based on the [SSLRec](https://github.com/HKUDS/SSLRec) framework.

## 👉 Code Structure

```
.
├── config/                 # Configuration files
│   ├── configurator.py    # Configuration parser
│   └── modelconf/         # Model-specific configs
│       └── iacd.yml       # IACD hyperparameters
├── data_utils/            # Data loading and preprocessing
│   ├── data_handler_kg.py # KG data handler
│   └── datasets_kg.py     # PyTorch dataset classes
├── datasets/              # Dataset directory
│   └── kg/               # Knowledge graph datasets
│       └── mooccube_kg/  # MOOCCube dataset
|       └── coco_kg/  # COCO dataset
|       └── mooper_kg/  # MOOPer dataset
├── models/                # Model implementations
│   ├── iacd.py           # IACD model
│   └── loss_utils.py     # Loss functions
│   
├── trainer/               # Training utilities
│   ├── trainer.py        # Training loop
│   ├── metrics.py        # Evaluation metrics
│   └── logger.py         # Logging utilities
└── main.py               # Main entry point
```

## 📚 Datasets Statistics

| Statistics | MOOCCube | MOOPer | COCO |
|------------|----------|--------|------|
| # Users | 34,917 | 28,702 | 24,036 |
| # Courses | 698 | 233 | 8,196 |
| # Interactions | 273,397 | 267,849 | 374,065 |
| # Density | 98.88% | 95.99% | 99.81% |
| **Knowledge Graph** | | | |
| # Entities | 239,440 | 10,184 | 11,237 |
| # Relations | 7 | 8 | 5 |
| # Triplets | 739,344 | 26,315 | 104,983 |



## 🚀 How to run the codes

### Training

```bash
python main.py --model iacd
```

### Configuration

Modify hyperparameters in `config/modelconf/iacd.yml`:
```yaml
model:
  embedding_size: 64        # Embedding dimension
  layer_num: 2             # LightGCN layers
  kg_layer_num: 2          # RGAT layers
  intent_size: 4          # Number of intent prototypes
  
  reg_weight: 1.0e-4       # L2 regularization
  cl_weight: 0.01          # Contrastive learning weight
  distill_weight: 0.1      # Student distillation weight
  teacher_weight: 0.1      # Teacher supervision weight
  temperature: 0.2         # Contrastive temperature
```

## 🌟 Citation

If you find this work is helpful to your research, please consider citing our paper and the [SSLRec](https://github.com/HKUDS/SSLRec) framework.
