# AICL-KG: Adversarial Intent Contrastive Learning for Knowledge Graph-enhanced Recommendation

## 项目简介

AICL-KG 是一个基于知识图谱的推荐系统模型，结合了对抗学习、意图解耦和对比学习技术，专门用于教育推荐场景。

### 核心创新

1. **对抗学习去噪**: 学习知识图谱边的重要性权重，自适应过滤噪声
2. **多维意图解耦**: 建模用户的多种学习意图，提升推荐准确性
3. **对比学习增强**: 通过对比学习增强表示学习能力

## 环境配置

### Python 版本

推荐使用 Python 3.10

### 快速安装

**方式一：使用安装脚本（推荐）**

Linux/Mac:
```bash
bash install.sh
```

Windows:
```bash
install.bat
```

**方式二：手动安装**

```bash
# 创建虚拟环境（推荐）
conda create -n aiclkg python=3.10 -y
conda activate aiclkg

# 安装基础依赖
pip install numpy==1.26.4 scipy==1.11.4

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 安装 DGL (CUDA 11.8)
pip install dgl==1.1.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

# 安装 PyG 相关库 (CUDA 11.8)
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 安装其他依赖
pip install faiss-cpu PyYAML tensorboard torchdata==0.6.1 pandas pydantic scikit-learn tqdm
```

### 其他 CUDA 版本

如果您使用的是其他 CUDA 版本，请访问以下链接获取对应的安装命令：
- PyTorch: https://pytorch.org/get-started/previous-versions/
- DGL: https://www.dgl.ai/pages/start.html
- PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### CPU 版本

如果只使用 CPU 训练：

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==1.1.2
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
pip install faiss-cpu PyYAML tensorboard torchdata==0.6.1 pandas pydantic scikit-learn tqdm
```

## 数据集准备

本项目使用 MOOC 数据集。数据集应放置在 `datasets/kg/mooccube_kg/` 目录下，包含以下文件：

- `train.txt`: 训练集用户-物品交互数据
- `test.txt`: 测试集用户-物品交互数据
- `kg_final.txt`: 知识图谱三元组数据
- `item_list.txt`: 物品 ID 映射（可选）
- `entity_list.txt`: 实体 ID 映射（可选）
- `user_list.txt`: 用户 ID 映射（可选）
- `relation_list.txt`: 关系 ID 映射（可选）

### 数据格式

**train.txt / test.txt 格式**:
```
user_id item_id1 item_id2 item_id3 ...
```

**kg_final.txt 格式**:
```
head_id relation_id tail_id
```

## 快速开始

### 完整示例

```bash
# 1. 进入项目目录
cd AICL-KG

# 2. 激活虚拟环境（如果使用）
conda activate aiclkg

# 3. 使用 GPU 训练（推荐）
python main.py --model aiclkg --device cuda --cuda 0

# 4. 或使用 CPU 训练（较慢）
python main.py --model aiclkg --device cpu
```

训练完成后，你会看到类似以下的输出：
```
Epoch 0: recall@5=0.0234, ndcg@5=0.0156, hr@5=0.0234
Epoch 1: recall@5=0.0245, ndcg@5=0.0163, hr@5=0.0245
...
Test set: recall@5=0.0267, ndcg@5=0.0178, hr@5=0.0267
```

## 使用说明

### 基础训练

```bash
# 默认使用 GPU 训练（自动检测）
python main.py --model aiclkg

# 指定 GPU 设备
python main.py --model aiclkg --cuda 0

# 使用 CPU 训练
python main.py --model aiclkg --device cpu
```

**注意**: 
- 默认情况下，程序会自动使用 GPU（如果可用）
- 使用 `--cuda` 参数可以指定 GPU 设备编号（默认为 0）
- 使用 `--device cpu` 强制使用 CPU 训练

### 修改超参数

编辑 `config/modelconf/aiclkg.yml` 文件来调整超参数：

```yaml
optimizer:
  lr: 1.0e-3  # 学习率
  
train:
  epoch: 500  # 训练轮数
  batch_size: 1024  # 批次大小
  patience: 10  # 早停耐心值
  
model:
  embedding_size: 64  # 嵌入维度
  layer_num: 2  # LightGCN 层数
  kg_layer_num: 2  # KG GNN 层数
  num_intents: 4  # 意图数量
  cl_weight: 0.01  # 对比学习权重
```

### 切换数据集

```bash
python main.py --model aiclkg --dataset <dataset_name>
```

## 输出结果

### 日志文件

训练日志保存在 `log/aiclkg/` 目录下，包含：
- 每个 epoch 的损失值
- 验证集和测试集的评估指标（Recall, NDCG, HR）

### 模型检查点

如果在配置文件中启用了 `save_model: true`，训练好的模型会保存在 `checkpoint/aiclkg/` 目录下。

## 评估指标

本项目支持以下评估指标：

- **Recall@K**: 召回率
- **NDCG@K**: 归一化折损累积增益
- **HR@K**: 命中率
- **Precision@K**: 精确率
- **MRR@K**: 平均倒数排名

默认评估 K=[5, 10, 20]

## 可复现性

为确保实验结果可复现，请在配置文件中设置：

```yaml
train:
  reproducible: true
  seed: 2025
```

## 项目结构

```
AICL-KG/
├── config/              # 配置文件
├── data_utils/          # 数据处理模块
├── datasets/            # 数据集目录
├── models/              # 模型实现
├── trainer/             # 训练和评估模块
├── checkpoint/          # 模型检查点
├── log/                 # 训练日志
├── main.py              # 主入口
├── requirements.txt     # 依赖列表
└── README.md            # 本文件
```

## 常见问题

### Q: 如何查看训练进度？

A: 训练过程会在终端显示进度条，同时日志文件会记录详细信息。日志保存在 `log/aiclkg/` 目录下。

### Q: 训练时内存不足怎么办？

A: 可以减小 `batch_size` 或 `embedding_size`。例如：
```yaml
train:
  batch_size: 512  # 从 1024 减小到 512

model:
  embedding_size: 32  # 从 64 减小到 32
```

### Q: CPU 训练太慢怎么办？

A: AICLKG 是一个大型模型，强烈建议使用 GPU 训练。如果必须使用 CPU，可以：
- 减小 batch_size
- 减小 embedding_size
- 减少训练 epoch 数

### Q: 如何使用预训练模型？

A: 在配置文件中添加：
```yaml
train:
  pretrain_path: checkpoint/aiclkg/aiclkg-mooc-1234567890.pth
```

### Q: 支持分布式训练吗？

A: 当前版本不支持，未来版本会考虑添加。

### Q: 如何调整超参数？

A: 编辑 `config/modelconf/aiclkg.yml` 文件，修改相应的参数值。主要超参数包括：
- `lr`: 学习率
- `embedding_size`: 嵌入维度
- `layer_num`: LightGCN 层数
- `kg_layer_num`: KG GNN 层数
- `num_intents`: 意图数量
- `cl_weight`: 对比学习权重

## 性能提示

### GPU vs CPU

- **GPU 训练**（推荐）: 每个 epoch 约 2-5 分钟
- **CPU 训练**: 每个 epoch 约 1-2 小时

### 内存需求

- **GPU**: 建议至少 8GB 显存
- **CPU**: 建议至少 16GB 内存

### 优化建议

1. 使用 `eval_at_one_forward: true` 加速评估（已默认启用）
2. 调整 `batch_size` 以充分利用 GPU
3. 使用早停机制避免过拟合

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{aiclkg2025,
  title={AICL-KG: Adversarial Intent Contrastive Learning for Knowledge Graph-enhanced Recommendation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 致谢

本项目基于 [SSLRec](https://github.com/HKUDS/SSLRec) 框架开发。感谢 SSLRec 团队提供的优秀基础框架。

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- Email: your.email@example.com
- GitHub Issues: [项目地址]

---

**注意**: 本代码仅供学术研究使用。
