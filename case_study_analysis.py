"""
案例分析脚本 - IACD 模型可解释性分析

展示：
1. 意图感知的边重要性（不同用户对同一边的注意力）
2. 去噪效果（高分边 vs 低分边）
3. 用户推荐路径可视化

所有输出保存在 case_study/ 目录下
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 创建输出目录
OUTPUT_DIR = './case_study'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 添加命令行参数以满足 configurator 的要求
sys.argv = ['case_study_analysis.py', '--model', 'iacd', '--dataset', 'mooc']

from config.configurator import configs
from data_utils.data_handler_kg import DataHandlerKG
from models.iacd import IACD

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path):
    """加载训练好的模型"""
    print("Loading data...")
    data_handler = DataHandlerKG()
    data_handler.load_data()
    
    print("Loading model...")
    model = IACD(data_handler).to(configs['device'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=configs['device']))
    model.eval()
    
    # 清理显存
    torch.cuda.empty_cache()
    
    return model, data_handler


def case1_intent_aware_attention(model, data_handler, n_users=3, n_edges=5):
    """
    案例1: 意图感知的边重要性
    
    展示不同用户对同一组边的注意力权重
    """
    print("\n" + "="*60)
    print("案例1: 意图感知的边重要性分析")
    print("="*60)
    
    # 选择一些用户（不同意图）
    user_ids = np.random.choice(model.n_users, n_users, replace=False)
    user_intents = model.user_embed[user_ids].detach()  # [n_users, emb_size]
    
    # 选择一些边
    edge_ids = np.random.choice(model.edge_index.shape[1], n_edges, replace=False)
    
    # 获取边的信息
    head_ids = model.edge_index[0, edge_ids]
    tail_ids = model.edge_index[1, edge_ids]
    rel_ids = model.edge_type[edge_ids]
    
    head_emb = model.entity_embed[head_ids].detach()
    tail_emb = model.entity_embed[tail_ids].detach()
    rel_emb = model.relation_embed[rel_ids].detach()
    
    # 计算边的语义表示
    edge_semantic = (head_emb + rel_emb + tail_emb) / 3.0  # [n_edges, emb_size]
    
    # 计算相似度和注意力权重
    similarity = torch.matmul(edge_semantic, user_intents.T)  # [n_edges, n_users]
    similarity = similarity / torch.sqrt(torch.tensor(model.embedding_size, dtype=torch.float32))
    
    gate_weights = torch.sigmoid(similarity)  # [n_edges, n_users]
    gate_sum = gate_weights.sum(dim=-1, keepdim=True) + 1e-8
    attn_weights = (gate_weights / gate_sum).cpu().numpy()  # [n_edges, n_users]
    
    # 计算边的重要性分数
    with torch.no_grad():
        student_scores = []
        for i in range(n_users):
            user_intent_single = user_intents[i:i+1]  # [1, emb_size]
            score, _ = model.collaborative_denoiser(
                head_emb, rel_emb, tail_emb, user_intent_single
            )
            student_scores.append(score.cpu().numpy())
        student_scores = np.array(student_scores).T  # [n_edges, n_users]
    
    # 打印结果
    print(f"\n选择了 {n_users} 个用户和 {n_edges} 条边")
    print("\n注意力权重矩阵 (行=边, 列=用户):")
    print("值越大表示该用户对该边越关注\n")
    
    # 创建表格
    print(f"{'边ID':<8}", end="")
    for u in range(n_users):
        print(f"用户{u+1:<6}", end="")
    print()
    print("-" * (8 + 10 * n_users))
    
    for e in range(n_edges):
        print(f"边{edge_ids[e]:<6}", end="")
        for u in range(n_users):
            print(f"{attn_weights[e, u]:.4f}   ", end="")
        print()
    
    print("\n边重要性分数矩阵 (行=边, 列=用户):")
    print("值越大表示该边对该用户越重要\n")
    
    print(f"{'边ID':<8}", end="")
    for u in range(n_users):
        print(f"用户{u+1:<6}", end="")
    print()
    print("-" * (8 + 10 * n_users))
    
    for e in range(n_edges):
        print(f"边{edge_ids[e]:<6}", end="")
        for u in range(n_users):
            print(f"{student_scores[e, u]:.4f}   ", end="")
        print()
    
    # 可视化：热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 注意力权重热力图
    sns.heatmap(attn_weights, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=[f'User{i+1}' for i in range(n_users)],
                yticklabels=[f'Edge{edge_ids[i]}' for i in range(n_edges)],
                ax=ax1, cbar_kws={'label': 'Attention Weight'})
    ax1.set_title('Intent-Aware Attention Weights')
    ax1.set_xlabel('Users')
    ax1.set_ylabel('Edges')
    
    # 重要性分数热力图
    sns.heatmap(student_scores, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=[f'User{i+1}' for i in range(n_users)],
                yticklabels=[f'Edge{edge_ids[i]}' for i in range(n_edges)],
                ax=ax2, cbar_kws={'label': 'Importance Score'})
    ax2.set_title('Edge Importance Scores')
    ax2.set_xlabel('Users')
    ax2.set_ylabel('Edges')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'case1_intent_aware_attention.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: {save_path}")
    
    return attn_weights, student_scores


def case2_denoising_effect(model, data_handler, top_k=10):
    """
    案例2: 去噪效果分析
    
    展示模型识别的重要边和噪声边
    """
    print("\n" + "="*60)
    print("案例2: 去噪效果分析")
    print("="*60)
    
    # 采样用户意图（避免显存爆炸）
    # 从所有用户中随机采样256个用户
    n_sample_users = min(256, model.n_users)
    sampled_user_ids = np.random.choice(model.n_users, n_sample_users, replace=False)
    sampled_user_intents = model.user_embed[sampled_user_ids].detach()
    
    print(f"使用 {n_sample_users} 个采样用户计算边重要性")
    
    # 获取所有边的信息
    head_emb = model.entity_embed[model.edge_index[0]].detach()
    tail_emb = model.entity_embed[model.edge_index[1]].detach()
    rel_emb = model.relation_embed[model.edge_type].detach()
    
    # 计算所有边的重要性分数
    with torch.no_grad():
        student_scores, _ = model.collaborative_denoiser(
            head_emb, rel_emb, tail_emb, sampled_user_intents
        )
        student_scores = student_scores.cpu().numpy()
    
    # 找出 Top-K 重要边和噪声边
    top_indices = np.argsort(student_scores)[-top_k:][::-1]  # 降序
    bottom_indices = np.argsort(student_scores)[:top_k]
    
    print(f"\nTop-{top_k} 重要边 (高分边):")
    print(f"{'排名':<6}{'边ID':<10}{'头实体':<10}{'关系':<10}{'尾实体':<10}{'分数':<10}")
    print("-" * 60)
    for rank, idx in enumerate(top_indices, 1):
        head_id = model.edge_index[0, idx].item()
        tail_id = model.edge_index[1, idx].item()
        rel_id = model.edge_type[idx].item()
        score = student_scores[idx]
        print(f"{rank:<6}{idx:<10}{head_id:<10}{rel_id:<10}{tail_id:<10}{score:.4f}")
    
    print(f"\nTop-{top_k} 噪声边 (低分边):")
    print(f"{'排名':<6}{'边ID':<10}{'头实体':<10}{'关系':<10}{'尾实体':<10}{'分数':<10}")
    print("-" * 60)
    for rank, idx in enumerate(bottom_indices, 1):
        head_id = model.edge_index[0, idx].item()
        tail_id = model.edge_index[1, idx].item()
        rel_id = model.edge_type[idx].item()
        score = student_scores[idx]
        print(f"{rank:<6}{idx:<10}{head_id:<10}{rel_id:<10}{tail_id:<10}{score:.4f}")
    
    # 统计分析
    print(f"\n统计信息:")
    print(f"总边数: {len(student_scores)}")
    print(f"平均分数: {student_scores.mean():.4f}")
    print(f"标准差: {student_scores.std():.4f}")
    print(f"最高分: {student_scores.max():.4f}")
    print(f"最低分: {student_scores.min():.4f}")
    print(f"中位数: {np.median(student_scores):.4f}")
    
    # 可视化：分数分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 分数分布直方图（使用对数刻度更好地展示分布）
    ax1.hist(student_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(student_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {student_scores.mean():.3f}')
    ax1.axvline(np.median(student_scores), color='green', linestyle='--',
                label=f'Median: {np.median(student_scores):.3f}')
    ax1.set_xlabel('Edge Importance Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Edge Importance Scores')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Top-K 和 Bottom-K 对比（使用对数刻度）
    x_pos = np.arange(top_k)
    
    # 为了更好的可视化，使用对数刻度
    ax2.bar(x_pos - 0.2, student_scores[top_indices], width=0.4, 
            label='Top-K Important Edges', color='green', alpha=0.7)
    ax2.bar(x_pos + 0.2, student_scores[bottom_indices], width=0.4,
            label='Top-K Noisy Edges', color='red', alpha=0.7)
    
    # 使用对数刻度（如果分数差异很大）
    score_range = student_scores.max() - student_scores.min()
    if score_range > 0.9:  # 如果分数范围很大（接近0-1）
        ax2.set_yscale('log')
        ax2.set_ylabel('Importance Score (log scale)')
        print("\n注意: 第二个图使用对数刻度以更好地展示极端值")
    else:
        ax2.set_ylabel('Importance Score')
    
    ax2.set_xlabel('Rank')
    ax2.set_title(f'Top-{top_k} Important vs Noisy Edges')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{i+1}' for i in range(top_k)])
    ax2.legend()
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'case2_denoising_effect.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: {save_path}")
    
    # 额外可视化：展示不同分数区间的边分布
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 将分数分成多个区间
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, bin_edges = np.histogram(student_scores, bins=bins)
    
    # 绘制柱状图
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    colors = plt.cm.RdYlGn(bin_centers)  # 红色(低分) -> 黄色(中分) -> 绿色(高分)
    
    bars = ax.bar(range(len(hist)), hist, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Score Range')
    ax.set_ylabel('Number of Edges')
    ax.set_title('Edge Distribution Across Score Ranges')
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)], 
                        rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, hist)):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}\n({count/len(student_scores)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path2 = os.path.join(OUTPUT_DIR, 'case2_score_distribution.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✅ 额外可视化已保存: {save_path2}")
    
    return top_indices, bottom_indices, student_scores


def case3_user_recommendation_path(model, data_handler, user_id=None, top_k=5):
    """
    案例3: 用户推荐路径可视化
    
    展示一个用户的推荐课程及相关的重要边
    """
    print("\n" + "="*60)
    print("案例3: 用户推荐路径分析")
    print("="*60)
    
    # 随机选择一个用户（如果未指定）
    if user_id is None:
        user_id = np.random.randint(0, model.n_users)
    
    print(f"\n分析用户 ID: {user_id}")
    
    # 获取用户的推荐分数
    with torch.no_grad():
        user_emb, item_emb = model.generate()
        user_scores = torch.matmul(user_emb[user_id], item_emb.T).cpu().numpy()
    
    # 获取 Top-K 推荐
    top_items = np.argsort(user_scores)[-top_k:][::-1]
    
    print(f"\nTop-{top_k} 推荐课程:")
    print(f"{'排名':<6}{'课程ID':<10}{'推荐分数':<12}")
    print("-" * 30)
    for rank, item_id in enumerate(top_items, 1):
        score = user_scores[item_id]
        print(f"{rank:<6}{item_id:<10}{score:.4f}")
    
    # 找出与这些课程相关的重要边
    print(f"\n与推荐课程相关的重要边:")
    
    user_intent = model.user_embed[user_id:user_id+1].detach()
    
    for item_id in top_items[:3]:  # 只展示前3个课程的边
        # 找出与该课程相关的边
        item_edges_mask = (model.edge_index[0] == item_id) | (model.edge_index[1] == item_id)
        item_edge_indices = torch.where(item_edges_mask)[0]
        
        if len(item_edge_indices) == 0:
            continue
        
        # 计算这些边的重要性
        head_emb = model.entity_embed[model.edge_index[0, item_edge_indices]].detach()
        tail_emb = model.entity_embed[model.edge_index[1, item_edge_indices]].detach()
        rel_emb = model.relation_embed[model.edge_type[item_edge_indices]].detach()
        
        with torch.no_grad():
            edge_scores, _ = model.collaborative_denoiser(
                head_emb, rel_emb, tail_emb, user_intent
            )
            edge_scores = edge_scores.cpu().numpy()
        
        # 找出最重要的边
        if len(edge_scores) > 0:
            top_edge_idx = np.argmax(edge_scores)
            global_edge_idx = item_edge_indices[top_edge_idx].item()
            
            head_id = model.edge_index[0, global_edge_idx].item()
            tail_id = model.edge_index[1, global_edge_idx].item()
            rel_id = model.edge_type[global_edge_idx].item()
            
            print(f"\n课程 {item_id} 的最重要边:")
            print(f"  {head_id} --[关系{rel_id}]--> {tail_id}")
            print(f"  重要性分数: {edge_scores[top_edge_idx]:.4f}")
    
    print("\n✅ 案例3分析完成")
    
    return top_items, user_scores


def main():
    """主函数"""
    print("IACD 模型案例分析")
    print("="*60)
    
    # 创建日志文件
    log_file = os.path.join(OUTPUT_DIR, 'analysis_results.txt')
    
    # 重定向输出到文件和终端
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    logger = Logger(log_file)
    sys.stdout = logger
    
    # 加载模型
    checkpoint_path = './checkpoint/iacd/iacd-mooc-1763423464.pth'
    model, data_handler = load_model(checkpoint_path)
    
    print(f"\n模型信息:")
    print(f"  用户数: {model.n_users}")
    print(f"  物品数: {model.n_items}")
    print(f"  实体数: {model.n_entities}")
    print(f"  边数: {model.edge_index.shape[1]}")
    
    # 案例1: 意图感知
    case1_intent_aware_attention(model, data_handler, n_users=3, n_edges=8)
    torch.cuda.empty_cache()  # 清理显存
    
    # 案例2: 去噪效果
    case2_denoising_effect(model, data_handler, top_k=10)
    torch.cuda.empty_cache()  # 清理显存
    
    # 案例3: 推荐路径
    case3_user_recommendation_path(model, data_handler, top_k=5)
    torch.cuda.empty_cache()  # 清理显存
    
    print("\n" + "="*60)
    print("✅ 所有案例分析完成！")
    print(f"✅ 所有结果已保存到: {OUTPUT_DIR}/")
    print("="*60)
    
    # 关闭日志文件
    logger.close()
    sys.stdout = logger.terminal


if __name__ == '__main__':
    main()
