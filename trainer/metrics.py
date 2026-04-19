import torch
import numpy as np
from config.configurator import configs


class Metric(object):
    def __init__(self):
        self.metrics = configs['test']['metrics']
        self.k = configs['test']['k']

    def recall(self, test_data, r, k):
        """Recall@K"""
        right_pred = r[:, :k].sum(1)
        recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
        recall = np.sum(right_pred / recall_n)
        return recall

    def ndcg(self, test_data, r, k):
        """NDCG@K"""
        assert len(r) == len(test_data)
        pred_data = r[:, :k]

        test_matrix = np.zeros((len(pred_data), k))
        for i, items in enumerate(test_data):
            length = k if k <= len(items) else len(items)
            test_matrix[i, :length] = 1
        max_r = test_matrix
        idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
        dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg, axis=1)
        idcg[idcg == 0.] = 1.
        ndcg = dcg / idcg
        ndcg[np.isnan(ndcg)] = 0.
        return np.sum(ndcg)

    def hr(self, r, k):
        """Hit Rate@K"""
        pred_data = r[:, :k]
        hit = pred_data.sum(1)
        hit = (hit > 0).astype(float)
        return np.sum(hit)

    def get_label(self, test_data, pred_data):
        """Get binary labels for predictions"""
        r = []
        for i in range(len(test_data)):
            ground_true = test_data[i]
            predict_topk = pred_data[i]
            pred = list(map(lambda x: x in ground_true, predict_topk))
            pred = np.array(pred).astype("float")
            r.append(pred)
        return np.array(r).astype('float')

    def eval_batch(self, data, topks):
        """Evaluate a batch of predictions"""
        sorted_items = data[0].numpy()
        ground_true = data[1]
        r = self.get_label(ground_true, sorted_items)

        result = {}
        for metric in self.metrics:
            result[metric] = []

        for k in topks:
            for metric in result:
                if metric == 'recall':
                    result[metric].append(self.recall(ground_true, r, k))
                elif metric == 'ndcg':
                    result[metric].append(self.ndcg(ground_true, r, k))
                elif metric == 'hr':
                    result[metric].append(self.hr(r, k))

        for metric in result:
            result[metric] = np.array(result[metric])

        return result

    def _mask_history_pos(self, batch_rate, test_user, test_dataloader):
        """Mask training items from predictions"""
        if not hasattr(test_dataloader.dataset, 'user_history_lists'):
            return batch_rate
        for i, user_idx in enumerate(test_user):
            pos_list = test_dataloader.dataset.user_history_lists[user_idx]
            batch_rate[i, pos_list] = -1e8
        return batch_rate
    
    def eval_at_one_forward(self, model, test_dataloader):
        """Evaluate model with single forward pass (efficient for GNN models)
        
        支持 IACD 的个性化评分：每个用户根据自己的意图分布动态合成 item embedding
        """
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        batch_ratings = []
        ground_truths = []
        test_user_count = 0
        test_user_num = len(test_dataloader.dataset.test_users)

        # Generate all embeddings at once
        with torch.no_grad():
            gen_output = model.generate()
            
            # 检查是否是 IACD 的四元组返回（个性化评分）
            if isinstance(gen_output, tuple) and len(gen_output) == 4:
                user_emb, item_struc_emb, kg_emb_stack, user_intent_attn = gen_output
                use_personalized = True
            else:
                # 兼容其他模型的二元组返回
                user_emb, item_emb = gen_output
                use_personalized = False

        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            test_user = tem[0].numpy().tolist()
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            
            # Predict
            batch_u = batch_data[0]
            
            with torch.no_grad():
                if use_personalized:
                    # IACD 个性化评分
                    u_emb_batch = user_emb[batch_u]  # [B, D]
                    attn_batch = user_intent_attn[batch_u]  # [B, K]
                    
                    # 结构分数：user @ item_struc.T
                    score_struc = torch.matmul(u_emb_batch, item_struc_emb.T)  # [B, N]
                    
                    # KG 个性化分数：对每个意图分支分别打分再聚合
                    # kg_emb_stack: [K, N, D]
                    score_kg = torch.stack([torch.matmul(u_emb_batch, kg_emb.T) 
                                           for kg_emb in kg_emb_stack], dim=1)  # [B, K, N]
                    score_personalized = (attn_batch.unsqueeze(-1) * score_kg).sum(dim=1)  # [B, N]
                    
                    batch_pred = score_struc + score_personalized
                else:
                    # 标准评分
                    batch_u_emb, all_i_emb = user_emb[batch_u], item_emb
                    batch_pred = model.rating(batch_u_emb, all_i_emb)
                    
            test_user_count += batch_pred.shape[0]
            
            # Filter out history items
            batch_pred = self._mask_history_pos(
                batch_pred, test_user, test_dataloader)
            _, batch_rate = torch.topk(batch_pred, k=max(self.k))
            batch_ratings.append(batch_rate.cpu())
            
            # Ground truth
            ground_truth = []
            for user_idx in test_user:
                ground_truth.append(
                    list(test_dataloader.dataset.user_pos_lists[user_idx]))
            ground_truths.append(ground_truth)
        
        assert test_user_count == test_user_num

        # Calculate metrics
        data_pair = zip(batch_ratings, ground_truths)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.eval_batch(_data, self.k))
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / test_user_num

        return result

    def eval(self, model, test_dataloader):
        """Main evaluation function"""
        # Use efficient evaluation for GNN models
        if 'eval_at_one_forward' in configs['test'] and configs['test']['eval_at_one_forward']:
            return self.eval_at_one_forward(model, test_dataloader)
        
        # Standard evaluation (not used for IACD)
        raise NotImplementedError("Standard eval not implemented. Use eval_at_one_forward=true in config.")
