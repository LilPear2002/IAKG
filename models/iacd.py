import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_scatter import scatter_sum, scatter_softmax
from logging import getLogger
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class CollaborativeDenoiser(nn.Module):

    def __init__(self, embedding_size, hidden_size=128):
        super(CollaborativeDenoiser, self).__init__()

        self.embedding_size = embedding_size

        # Student Network: learns to predict edge importance
        self.student_network = nn.Sequential(
            nn.Linear(embedding_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Teacher Network: provides soft labels based on user intent and interactions
        self.teacher_network = nn.Sequential(
            nn.Linear(embedding_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, head_emb, rel_emb, tail_emb, user_intent_emb):
        """
        Full Chunking Implementation to prevent OOM
        """
        # Handle different input dimensions
        if user_intent_emb.dim() == 1:
            user_intent_emb = user_intent_emb.unsqueeze(0)  # [1, emb_size]

        n_edges = head_emb.shape[0]

        # Step 1: Compute edge semantic representation (Global is fine, low memory)
        edge_semantic = (head_emb + rel_emb + tail_emb) / 3.0  # [n_edges, emb_size]

        # [Optimization] Chunk Size (Adjust based on GPU memory)
        # 20000 is a safe number for 24G VRAM with AMP
        chunk_size = 100000

        student_score_list = []
        teacher_score_list = []

        # Step 2: Loop through chunks
        for i in range(0, n_edges, chunk_size):
            # Slicing data for current chunk
            semantic_chunk = edge_semantic[i: i + chunk_size]
            head_chunk = head_emb[i: i + chunk_size]
            rel_chunk = rel_emb[i: i + chunk_size]
            tail_chunk = tail_emb[i: i + chunk_size]

            # === Batch-Aware Attention (Chunked) ===
            # similarity: [chunk_size, batch_size]
            similarity = torch.matmul(semantic_chunk, user_intent_emb.T)
            similarity = similarity / torch.sqrt(
                torch.tensor(self.embedding_size, dtype=torch.float32, device=similarity.device))

            # Gating mechanism
            gate_weights = torch.sigmoid(similarity)

            # Normalize
            gate_sum = gate_weights.sum(dim=-1, keepdim=True) + 1e-8
            attn_weights = gate_weights / gate_sum

            # Aggregate Context: [chunk_size, emb_size]
            chunk_context = torch.matmul(attn_weights, user_intent_emb)

            # === Construct Edge Repr (Chunked) ===
            # [chunk_size, 4 * emb_size]
            edge_repr_chunk = torch.cat([head_chunk, rel_chunk, tail_chunk, chunk_context], dim=-1)

            # === Network Forward (Chunked) ===
            # Move MLP calculation inside loop to save memory
            s_score = self.student_network(edge_repr_chunk).squeeze(-1)
            t_score = self.teacher_network(edge_repr_chunk).squeeze(-1)

            student_score_list.append(s_score)
            teacher_score_list.append(t_score)

        # Step 3: Concatenate results
        student_score = torch.cat(student_score_list, dim=0)
        teacher_score = torch.cat(teacher_score_list, dim=0)

        return student_score, teacher_score


class RGAT(nn.Module):

    def __init__(self, embedding_size, n_layers=2, mess_dropout_rate=0.1):
        super(RGAT, self).__init__()
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.mess_dropout_rate = mess_dropout_rate

        # Learnable attention parameters
        self.W = nn.Parameter(torch.empty(size=(embedding_size, embedding_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * embedding_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Relation-aware fusion layer
        self.fc = nn.Linear(2 * embedding_size, embedding_size)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def aggregate(self, entity_emb, relation_emb, edge_index, edge_type, edge_weight=None):
        head, tail = edge_index

        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        rel_emb = relation_emb[edge_type]

        # Concatenate head and tail embeddings
        a_input = torch.cat([head_emb, tail_emb], dim=-1)

        # Relation-aware attention: multiply with relation embedding
        e_input = torch.multiply(self.fc(a_input), rel_emb).sum(-1)

        # Apply leaky relu
        attn_score = self.leakyrelu(e_input)

        # Softmax normalization per head node
        attn_score = scatter_softmax(attn_score, head, dim=0, dim_size=entity_emb.shape[0])

        # Apply edge weight from collaborative denoiser if provided
        if edge_weight is not None:
            attn_score = attn_score * edge_weight.view(-1)

        # Aggregate neighbor embeddings
        agg_emb = tail_emb * attn_score.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])

        return agg_emb

    def forward(self, entity_emb, relation_emb, edge_index, edge_type, edge_weight=None, mess_dropout=True):
        entity_res_emb = entity_emb

        for _ in range(self.n_layers):
            entity_emb = self.aggregate(entity_emb, relation_emb, edge_index, edge_type, edge_weight)

            # Apply dropout during training (Message/Node Dropout)
            if mess_dropout and self.training:
                entity_emb = self.dropout(entity_emb)

            # Normalize embeddings
            entity_emb = F.normalize(entity_emb)

            # Residual connection
            entity_res_emb = entity_res_emb + entity_emb

        return entity_res_emb


class IACD(BaseModel):

    def __init__(self, data_handler):
        super(IACD, self).__init__(data_handler)

        self.logger = getLogger()

        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.n_entities = configs['data']['entity_num']
        self.n_relations = configs['data']['relation_num']

        self.embedding_size = configs['model']['embedding_size']
        self.layer_num = configs['model']['layer_num']
        self.kg_layer_num = configs['model']['kg_layer_num']

        self.reg_weight = configs['model']['reg_weight']
        self.cl_weight = configs['model']['cl_weight']
        self.distill_weight = configs['model']['distill_weight']
        self.teacher_weight = configs['model']['teacher_weight']

        self.temperature = configs['model']['temperature']

        self.ui_mat = data_handler.ui_mat
        self.ui_graph = self._get_norm_adj_mat(data_handler.ui_mat).to(configs['device'])
        self.kg_edges = data_handler.kg_edges
        self.kg_dict = data_handler.kg_dict

        kg_edges_array = np.array(self.kg_edges)
        self.edge_index = torch.LongTensor([kg_edges_array[:, 0], kg_edges_array[:, 1]]).to(configs['device'])
        self.edge_type = torch.LongTensor(kg_edges_array[:, 2]).to(configs['device'])

        self.user_embed = nn.Parameter(torch.empty(self.n_users, self.embedding_size))
        self.entity_embed = nn.Parameter(torch.empty(self.n_entities, self.embedding_size))
        self.relation_embed = nn.Parameter(torch.empty(self.n_relations, self.embedding_size))

        nn.init.xavier_uniform_(self.user_embed)
        nn.init.xavier_uniform_(self.entity_embed)
        nn.init.xavier_uniform_(self.relation_embed)

        self.collaborative_denoiser = CollaborativeDenoiser(self.embedding_size)
        self.kg_gnn = RGAT(self.embedding_size, self.kg_layer_num, mess_dropout_rate=0.1)

        self.item_to_edges = self._build_item_to_edges_mapping()

        self.logger.info(f"IACD initialized: emb_size={self.embedding_size}, "
                         f"layer_num={self.layer_num}, kg_layer_num={self.kg_layer_num}, "
                         f"distill_weight={self.distill_weight}, teacher_weight={self.teacher_weight}")

    def _build_item_to_edges_mapping(self):
        item_to_edges = {}
        for edge_idx in range(self.edge_index.shape[1]):
            head = self.edge_index[0, edge_idx].item()
            tail = self.edge_index[1, edge_idx].item()

            if head < self.n_items:
                if head not in item_to_edges:
                    item_to_edges[head] = []
                item_to_edges[head].append(edge_idx)

            if tail < self.n_items:
                if tail not in item_to_edges:
                    item_to_edges[tail] = []
                item_to_edges[tail].append(edge_idx)

        if len(item_to_edges) > 0:
            max_edges_per_item = max(len(edges) for edges in item_to_edges.values())
        else:
            max_edges_per_item = 1

        item_edge_matrix = torch.full((self.n_items, max_edges_per_item), -1, dtype=torch.long)

        for item_id, edge_list in item_to_edges.items():
            item_edge_matrix[item_id, :len(edge_list)] = torch.tensor(edge_list)

        return item_edge_matrix.to(configs['device'])

    def _get_norm_adj_mat(self, ui_mat):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = ui_mat
        inter_M_t = ui_mat.transpose()

        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def _lightgcn_propagate(self, graph, embeds):
        return torch.spmm(graph, embeds)

    def lightgcn_forward(self, graph, kg_emb=None):
        if kg_emb is not None:
            item_init_emb = self.entity_embed[:self.n_items] + kg_emb[:self.n_items]
        else:
            item_init_emb = self.entity_embed[:self.n_items]

        all_emb = torch.cat([self.user_embed, item_init_emb], dim=0)
        emb_list = [all_emb]

        for _ in range(self.layer_num):
            all_emb = self._lightgcn_propagate(graph, all_emb)
            emb_list.append(all_emb)

        all_emb = torch.stack(emb_list, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])

        return user_emb, item_emb

    def contrastive_loss(self, view1_emb, view2_emb, nodes):
        view1_emb = F.normalize(view1_emb, dim=-1)
        view2_emb = F.normalize(view2_emb, dim=-1)

        view1_nodes = view1_emb[nodes]
        view2_nodes = view2_emb[nodes]

        pos_score = (view1_nodes * view2_nodes).sum(dim=-1) / self.temperature

        neg_score = torch.matmul(view1_nodes, view2_emb.T) / self.temperature

        loss = -torch.log(
            torch.exp(pos_score) / torch.exp(neg_score).sum(dim=-1)
        ).mean()

        return loss

    def get_edge_labels(self, pos_items, neg_items):
        n_edges = self.edge_index.shape[1]
        edge_labels = torch.zeros(n_edges).to(self.edge_index.device)
        pos_item_mask = torch.zeros(n_edges, dtype=torch.bool).to(self.edge_index.device)
        neg_item_mask = torch.zeros(n_edges, dtype=torch.bool).to(self.edge_index.device)

        for item in pos_items:
            if item < self.n_items:
                edge_indices = self.item_to_edges[item]
                valid_mask = edge_indices >= 0
                valid_edges = edge_indices[valid_mask]
                if len(valid_edges) > 0:
                    pos_item_mask[valid_edges] = True
                    edge_labels[valid_edges] = 1.0

        for item in neg_items:
            if item < self.n_items:
                edge_indices = self.item_to_edges[item]
                valid_mask = edge_indices >= 0
                valid_edges = edge_indices[valid_mask]
                if len(valid_edges) > 0:
                    neg_item_mask[valid_edges] = True

        return edge_labels, pos_item_mask, neg_item_mask

    def distillation_loss(self, student_score, teacher_score, pos_items, neg_items):
        edge_labels, pos_mask, neg_mask = self.get_edge_labels(pos_items, neg_items)

        # Teacher Loss
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            teacher_loss = F.binary_cross_entropy(
                teacher_score[pos_mask | neg_mask],
                edge_labels[pos_mask | neg_mask]
            )
        else:
            teacher_loss = torch.tensor(0.0).to(student_score.device)

        # Student Loss Part 1: MSE (Distill)
        student_loss_distill = F.mse_loss(student_score, teacher_score.detach())

        # Student Loss Part 2: Alignment
        if (pos_mask | neg_mask).sum() > 0:
            student_loss_align = F.binary_cross_entropy(
                student_score[pos_mask | neg_mask],
                edge_labels[pos_mask | neg_mask]
            )
            student_loss = student_loss_distill + 0.5 * student_loss_align
        else:
            student_loss = student_loss_distill

        return student_loss, teacher_loss

    def forward(self, users_batch):
        user_intent_emb = self.user_embed[users_batch].detach()

        head_emb = self.entity_embed[self.edge_index[0]].detach()
        tail_emb = self.entity_embed[self.edge_index[1]].detach()
        rel_emb = self.relation_embed[self.edge_type].detach()

        # Collaborative denoising (Chunked internal)
        student_score, teacher_score = self.collaborative_denoiser(
            head_emb, rel_emb, tail_emb, user_intent_emb
        )

        kg_emb_main = self.kg_gnn(
            self.entity_embed,
            self.relation_embed,
            self.edge_index,
            self.edge_type,
            edge_weight=student_score,
            mess_dropout=False
        )

        student_score_view1 = F.dropout(student_score, p=0.1, training=self.training)
        kg_emb_view1 = self.kg_gnn(
            self.entity_embed,
            self.relation_embed,
            self.edge_index,
            self.edge_type,
            edge_weight=student_score_view1,
            mess_dropout=True
        )

        student_score_view2 = F.dropout(student_score, p=0.1, training=self.training)
        kg_emb_view2 = self.kg_gnn(
            self.entity_embed,
            self.relation_embed,
            self.edge_index,
            self.edge_type,
            edge_weight=student_score_view2,
            mess_dropout=True
        )

        # Recommendation Task
        user_emb_cf, item_emb_cf = self.lightgcn_forward(self.ui_graph, kg_emb=kg_emb_main)

        item_emb = item_emb_cf

        return user_emb_cf, item_emb, kg_emb_view1, kg_emb_view2, student_score, teacher_score

    def cal_loss(self, batch_data):
        users, pos_items, neg_items = batch_data

        user_emb, item_emb, kg_emb_view1, kg_emb_view2, student_score, teacher_score = self.forward(users)

        user_emb_batch = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        # 1. BPR Loss
        bpr_loss = cal_bpr_loss(user_emb_batch, pos_item_emb, neg_item_emb) / user_emb_batch.shape[0]

        # 2. Contrastive Loss
        cl_nodes = torch.cat([pos_items, neg_items], dim=0)
        cl_loss = self.contrastive_loss(kg_emb_view1, kg_emb_view2, cl_nodes)

        # 3. Distillation Loss
        student_loss, teacher_loss = self.distillation_loss(
            student_score, teacher_score, pos_items, neg_items)

        # 4. Regularization Loss
        reg_loss = reg_pick_embeds([self.user_embed[users],
                                    self.entity_embed[pos_items],
                                    self.entity_embed[neg_items]]) / user_emb_batch.shape[0]
        reg_loss = self.reg_weight * reg_loss

        total_loss = bpr_loss + \
                     self.cl_weight * cl_loss + \
                     self.distill_weight * student_loss + \
                     self.teacher_weight * teacher_loss + \
                     reg_loss

        loss_dict = {
            'bpr_loss': bpr_loss.item(),
            'cl_loss': cl_loss.item(),
            'student_loss': student_loss.item(),
            'teacher_loss': teacher_loss.item(),
            'reg_loss': reg_loss.item()
        }

        return total_loss, loss_dict

    def generate(self):
        """Generate embeddings for evaluation (using Global User Intent)"""
        head_emb = self.entity_embed[self.edge_index[0]].detach()
        tail_emb = self.entity_embed[self.edge_index[1]].detach()
        rel_emb = self.relation_embed[self.edge_type].detach()

        # [Optimization] Use Global Mean for Inference
        global_user_intent = self.user_embed.mean(dim=0, keepdim=True).detach()

        with torch.no_grad():
            student_score, _ = self.collaborative_denoiser(
                head_emb, rel_emb, tail_emb, global_user_intent
            )

            kg_emb = self.kg_gnn(
                self.entity_embed,
                self.relation_embed,
                self.edge_index,
                self.edge_type,
                edge_weight=student_score,
                mess_dropout=False
            )

        user_emb_cf, item_emb_cf = self.lightgcn_forward(self.ui_graph, kg_emb=kg_emb)

        return user_emb_cf, item_emb_cf

    def rating(self, u_emb, i_emb):
        return torch.matmul(u_emb, i_emb.t())

    def full_predict(self, batch_data):
        users, train_mask = batch_data

        # [Optimization] Use Global Mean for Inference Consistency
        global_user_intent = self.user_embed.mean(dim=0, keepdim=True).detach()

        head_emb = self.entity_embed[self.edge_index[0]]
        tail_emb = self.entity_embed[self.edge_index[1]]
        rel_emb = self.relation_embed[self.edge_type]

        with torch.no_grad():
            student_score, _ = self.collaborative_denoiser(
                head_emb, rel_emb, tail_emb, global_user_intent
            )

            kg_emb = self.kg_gnn(
                self.entity_embed,
                self.relation_embed,
                self.edge_index,
                self.edge_type,
                edge_weight=student_score,
                mess_dropout=False
            )

        user_emb_cf, item_emb_cf = self.lightgcn_forward(self.ui_graph, kg_emb=kg_emb)

        user_emb_batch = user_emb_cf[users]

        full_preds = user_emb_batch @ item_emb_cf.T

        full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds