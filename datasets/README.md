# Dataset Directory

This directory contains knowledge graph-enhanced recommendation datasets.

## Directory Structure

```
datasets/
└── kg/
    ├── mooccube_kg/
    ├── last-fm_kg/
    ├── mind_kg/
    └── alibaba-fashion_kg/
```

## Data Format

Each dataset directory should contain the following files:

### 1. `train.txt`
Training user-item interactions. Each line represents one user's interactions:
```
user_id item_id1 item_id2 item_id3 ...
```

Example:
```
0 10 25 37 89
1 5 12 45
2 3 8 15 22 31
```

### 2. `test.txt`
Test user-item interactions. Same format as `train.txt`:
```
user_id item_id1 item_id2 ...
```

### 3. `kg_final.txt`
Knowledge graph triplets. Each line represents one triplet:
```
head_entity_id relation_id tail_entity_id
```

Example:
```
0 1 100
0 2 150
10 3 200
```

**Note**: Entity IDs should be consistent with item IDs. Items are typically the first N entities in the knowledge graph.

### 4. Optional Files

- `user_list.txt`: List of user IDs (one per line)
- `item_list.txt`: List of item IDs (one per line)
- `entity_list.txt`: List of entity IDs (one per line)
- `relation_list.txt`: List of relation IDs (one per line)

## Dataset Statistics

### MOOCCube
- Users: 7,047
- Items (Courses): 721
- Interactions: 97,842
- Entities: 3,064
- Relations: 10
- KG Triplets: 23,791

### Last.FM
- Users: 1,872
- Items (Artists): 3,846
- Interactions: 42,346
- Entities: 9,366
- Relations: 60
- KG Triplets: 15,518

### MIND
- Users: 10,000
- Items (News): 9,432
- Interactions: 500,000
- Entities: 20,000
- Relations: 15
- KG Triplets: 100,000

### Alibaba-Fashion
- Users: 114,737
- Items (Products): 30,040
- Interactions: 1,781,093
- Entities: 81,614
- Relations: 51
- KG Triplets: 1,689,988

## Download

Due to size limitations, datasets are not included in this repository. 

Please download from:
- **MOOCCube**: [MOOCCube Official](http://moocdata.cn/data/MOOCCube)
- **Last.FM**: [HetRec 2011](https://grouplens.org/datasets/hetrec-2011/)
- **MIND**: [Microsoft MIND](https://msnews.github.io/)
- **Alibaba-Fashion**: Contact authors or use preprocessed version

After downloading, preprocess the data to match the format described above and place in the corresponding directories.

## Preprocessing

If you need to preprocess raw datasets, you can create a preprocessing script. Example structure:

```python
# preprocess.py
def preprocess_dataset(raw_data_path, output_path):
    # 1. Load raw data
    # 2. Build user-item interactions
    # 3. Build knowledge graph
    # 4. Save in required format
    pass
```

## Citation

If you use these datasets, please cite the original papers:

**MOOCCube**:
```bibtex
@inproceedings{yu2020mooccube,
  title={MOOCCube: A Large-scale Data Repository for NLP Applications in MOOCs},
  author={Yu, Jifan and Luo, Gan and Xiao, Tong and others},
  booktitle={ACL},
  year={2020}
}
```

**Last.FM**:
```bibtex
@inproceedings{cantador2011hetrec,
  title={Second workshop on information heterogeneity and fusion in recommender systems},
  author={Cantador, Iv{\'a}n and Brusilovsky, Peter and Kuflik, Tsvi},
  booktitle={RecSys},
  year={2011}
}
```
