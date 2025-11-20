"""
检查数据集规模，分析显存占用
"""
import numpy as np

# 读取知识图谱数据
kg_file = './datasets/kg/mooccube_kg/kg_final.txt'

try:
    kg_data = np.loadtxt(kg_file, dtype=np.int32)
    
    n_edges = len(kg_data)
    n_entities = max(kg_data[:, 0].max(), kg_data[:, 2].max()) + 1
    n_relations = kg_data[:, 1].max() + 1
    
    print("=" * 60)
    print("数据集规模分析")
    print("=" * 60)
    print(f"边数量 (n_edges): {n_edges:,}")
    print(f"实体数量 (n_entities): {n_entities:,}")
    print(f"关系数量 (n_relations): {n_relations:,}")
    print()
    
    # 计算显存占用
    print("=" * 60)
    print("显存占用估算 (单位: MB)")
    print("=" * 60)
    
    for emb_size in [128, 96, 64]:
        print(f"\n嵌入维度 = {emb_size}:")
        print("-" * 40)
        
        # 嵌入参数
        embed_params = (n_entities + n_relations) * emb_size * 4 / (1024**2)
        print(f"  嵌入参数: {embed_params:.1f} MB")
        
        # KG传播 (单层)
        kg_single_layer = (
            n_edges * emb_size * 4 * 3 +  # head, tail, rel
            n_edges * emb_size * 2 * 4     # concat
        ) / (1024**2)
        print(f"  KG传播 (单层): {kg_single_layer:.1f} MB")
        
        # KG传播 (2层×2视图)
        kg_total = kg_single_layer * 2 * 2
        print(f"  KG传播 (2层×2视图): {kg_total:.1f} MB")
        
        # Denoiser
        denoiser = n_edges * emb_size * 4 * 4 / (1024**2)
        print(f"  Denoiser: {denoiser:.1f} MB")
        
        # 总计
        total = embed_params + kg_total + denoiser
        print(f"  总计 (估算): {total:.1f} MB")
        print(f"  总计 (估算): {total/1024:.2f} GB")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print(f"你的数据集有 {n_edges:,} 条边")
    print(f"这就是为什么嵌入维度影响巨大的原因！")
    print(f"batch_size 只影响 ~1-2 MB")
    print(f"embedding_size 影响 ~{n_edges/1000:.0f}k × emb_size 的显存")
    
except FileNotFoundError:
    print("错误: 找不到数据集文件")
    print("请确保数据集在: ./datasets/kg/mooccube_kg/kg_final.txt")
except Exception as e:
    print(f"错误: {e}")
