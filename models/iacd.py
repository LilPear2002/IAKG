import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_scatter import scatter_sum, scatter_softmax
from logging import getLogger
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds


class IntentGuidedDenoiser(nn.Module):

    def __init__(self, embedding_size):
        super(IntentGuidedDenoiser, self).__init__()
        self.embedding_size = embedding_size
        self.edge_proj = nn.Linear(3 * embedding_size, embedding_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, head_emb, rel_emb, tail_emb, user_intent_emb):
        if user_intent_emb.dim() == 1:
            user_intent_emb = user_intent_emb.unsqueeze(0)

        edge_semantic = self.edge_proj(torch.cat([head_emb, rel_emb, tail_emb], dim=-1))  # [n_edges, emb_size]

        intent_sim = torch.matmul(edge_semantic, user_intent_emb.T)  # [n_edges, n_intent]
        intent_sim = intent_sim / (self.temperature.abs() + 0.1)  # 可学习温度

        edge_weight = torch.sigmoid(intent_sim.max(dim=-1)[0])  # [n_edges]
        
        return edge_weight


class RGAT(nn.Module):

    def __init__(self, embedding_size, n_layers=2, mess_dropout_rate=0.2):
        super(RGAT, self).__init__()
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.mess_dropout_rate = mess_dropout_rate

        self.fc = nn.Linear(2 * embedding_size, embedding_size)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def aggregate(self, entity_emb, relation_emb, edge_index, edge_type, edge_weight=None):
        head, tail = edge_index

        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        rel_emb = relation_emb[edge_type]

        a_input = torch.cat([head_emb, tail_emb], dim=-1)
        e_input = torch.multiply(self.fc(a_input), rel_emb).sum(-1)
        attn_score = self.leakyrelu(e_input)
        attn_score = scatter_softmax(attn_score, head, dim=0, dim_size=entity_emb.shape[0])

        if edge_weight is not None:
            attn_score = attn_score * edge_weight.view(-1)

        agg_emb = tail_emb * attn_score.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])

        return agg_emb

    def forward(self, entity_emb, relation_emb, edge_index, edge_type, edge_weight=None, mess_dropout=True):
        entity_res_emb = entity_emb

        for _ in range(self.n_layers):
            entity_emb = self.aggregate(entity_emb, relation_emb, edge_index, edge_type, edge_weight)

            if mess_dropout and self.training:
                entity_emb = self.dropout(entity_emb)

            entity_emb = F.normalize(entity_emb)
            entity_res_emb = entity_res_emb + entity_emb

        return entity_res_emb


class IACD(nn.Module):

    def __init__(self, data_handler):
        super(IACD, self).__init__()

        self.logger = getLogger()

        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.n_entities = configs['data']['entity_num']
        self.n_relations = configs['data']['relation_num']

        self.embedding_size = configs['model']['embedding_size']
        self.layer_num = configs['model']['layer_num']
        self.kg_layer_num = configs['model']['kg_layer_num']
        self.intent_size = configs['model']['intent_size']

        self.reg_weight = configs['model']['reg_weight']
        self.cl_weight = configs['model']['cl_weight']
        self.temperature = configs['model']['temperature']

        self.ui_mat = data_handler.ui_mat
        self.ui_graph = self._get_norm_adj_mat(data_handler.ui_mat).to(configs['device'])
        self.kg_edges = data_handler.kg_edges
        self.kg_dict = data_handler.kg_dict

        # Prepare KG edge index and types
        kg_edges_array = np.array(self.kg_edges)
        self.edge_index = torch.from_numpy(kg_edges_array[:, :2].transpose()).long().to(configs['device'])
        self.edge_type = torch.LongTensor(kg_edges_array[:, 2]).to(configs['device'])

        # Initialize embeddings
        self.user_embed = nn.Parameter(torch.empty(self.n_users, self.embedding_size))
        self.entity_embed = nn.Parameter(torch.empty(self.n_entities, self.embedding_size))
        self.relation_embed = nn.Parameter(torch.empty(self.n_relations, self.embedding_size))
        self.user_intent = nn.Parameter(torch.empty(self.embedding_size, self.intent_size))

        nn.init.xavier_uniform_(self.user_embed)
        nn.init.xavier_uniform_(self.entity_embed)
        nn.init.xavier_uniform_(self.relation_embed)
        nn.init.xavier_uniform_(self.user_intent)

        # Initialize modules
        self.intent_denoiser = IntentGuidedDenoiser(self.embedding_size)
        self.kg_gnn = RGAT(self.embedding_size, self.kg_layer_num, mess_dropout_rate=0.2)

        self.logger.info("IACD initialized")

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

        # Symmetric normalization
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

    def lightgcn_forward(self, graph):
        """LightGCN propagation for collaborative filtering"""
        item_init_emb = self.entity_embed[:self.n_items]
        all_emb = torch.cat([self.user_embed, item_init_emb], dim=0)
        emb_list = [all_emb]

        for _ in range(self.layer_num):
            all_emb = self._lightgcn_propagate(graph, all_emb)
            emb_list.append(all_emb)

        all_emb = torch.stack(emb_list, dim=1).mean(dim=1)
        user_struc_emb, item_struc_emb = torch.split(all_emb, [self.n_users, self.n_items])

        return user_struc_emb, item_struc_emb

    def contrastive_loss(self, view1_emb, view2_emb, nodes):
        """InfoNCE contrastive loss"""
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

    def forward(self, users_batch):
        # LightGCN for collaborative filtering
        user_struc_emb_all, item_struc_emb_all = self.lightgcn_forward(self.ui_graph)
        user_struc_batch = user_struc_emb_all[users_batch]
        
        # Intent-aware user representation
        user_intent_attn = torch.softmax(user_struc_batch @ self.user_intent, dim=1)  # [batch, intent_size]
        user_individual_intent = user_intent_attn @ self.user_intent.T

        final_user_emb = user_struc_batch + user_individual_intent

        head_emb = self.entity_embed[self.edge_index[0]]
        tail_emb = self.entity_embed[self.edge_index[1]]
        rel_emb = self.relation_embed[self.edge_type]

        intent_prototypes = self.user_intent.T  # [intent_size, emb_size]
        kg_emb_list = []
        edge_weight_list = []
        
        for proto in intent_prototypes:
            proto = proto.unsqueeze(0)
            edge_weight = self.intent_denoiser(head_emb, rel_emb, tail_emb, proto)
            edge_weight_list.append(edge_weight)
            
            kg_emb = self.kg_gnn(
                self.entity_embed,
                self.relation_embed,
                self.edge_index,
                self.edge_type,
                edge_weight=edge_weight,
                mess_dropout=False
            )
            kg_emb_list.append(kg_emb[:self.n_items])
        
        kg_emb_stack = torch.stack(kg_emb_list, dim=0)  # [intent_size, n_items, emb_size]

        # user_intent_attn: [batch, intent_size]
        # kg_emb_stack: [intent_size, n_items, emb_size]
        batch_item_kg_emb = torch.einsum('uk,kid->uid', user_intent_attn, kg_emb_stack)  # [batch, n_items, emb_size]
        final_item_emb_batch = item_struc_emb_all.unsqueeze(0) + batch_item_kg_emb  # [batch, n_items, emb_size]

        avg_edge_weight = torch.stack(edge_weight_list, dim=0).mean(dim=0)
        
        kg_emb_view1 = self.kg_gnn(
            self.entity_embed, self.relation_embed, self.edge_index, self.edge_type,
            edge_weight=F.dropout(avg_edge_weight, p=0.2, training=self.training),
            mess_dropout=True
        )
        kg_emb_view2 = self.kg_gnn(
            self.entity_embed, self.relation_embed, self.edge_index, self.edge_type,
            edge_weight=F.dropout(avg_edge_weight, p=0.2, training=self.training),
            mess_dropout=True
        )
        
        return final_user_emb, final_item_emb_batch, kg_emb_view1, kg_emb_view2

    def cal_loss(self, batch_data):
        users, pos_items, neg_items = batch_data
        batch_size = users.shape[0]

        final_user_emb, final_item_emb_batch, kg_emb_view1, kg_emb_view2 = self.forward(users)

        batch_idx = torch.arange(batch_size, device=users.device)
        pos_item_emb = final_item_emb_batch[batch_idx, pos_items]
        neg_item_emb = final_item_emb_batch[batch_idx, neg_items]

        # BPR loss
        bpr_loss = cal_bpr_loss(final_user_emb, pos_item_emb, neg_item_emb) / batch_size

        # Contrastive loss
        cl_nodes = torch.cat([pos_items, neg_items], dim=0)
        cl_loss = self.contrastive_loss(kg_emb_view1, kg_emb_view2, cl_nodes)

        # Regularization loss
        reg_loss = reg_pick_embeds([
            self.user_embed[users],
            self.entity_embed[pos_items],
            self.entity_embed[neg_items],
            self.user_intent
        ]) / batch_size
        reg_loss = self.reg_weight * reg_loss

        total_loss = bpr_loss + self.cl_weight * cl_loss + reg_loss

        loss_dict = {
            'bpr_loss': bpr_loss.item(),
            'cl_loss': cl_loss.item(),
            'reg_loss': reg_loss.item()
        }

        return total_loss, loss_dict

    def generate(self):
        return self.full_predict_eval()

    def rating(self, u_emb, i_emb):
        return torch.matmul(u_emb, i_emb.t())

    def full_predict_eval(self):
        user_struc_emb_all, item_struc_emb_all = self.lightgcn_forward(self.ui_graph)

        user_intent_attn = torch.softmax(user_struc_emb_all @ self.user_intent, dim=1)  # [n_users, intent_size]
        user_individual_intent = user_intent_attn @ self.user_intent.T
        final_user_emb = user_struc_emb_all + user_individual_intent

        head_emb = self.entity_embed[self.edge_index[0]]
        tail_emb = self.entity_embed[self.edge_index[1]]
        rel_emb = self.relation_embed[self.edge_type]

        with torch.no_grad():
            intent_prototypes = self.user_intent.T
            kg_emb_list = []
            
            for proto in intent_prototypes:
                proto = proto.unsqueeze(0)
                edge_weight = self.intent_denoiser(head_emb, rel_emb, tail_emb, proto)
                kg_emb = self.kg_gnn(self.entity_embed, self.relation_embed, self.edge_index,
                                     self.edge_type, edge_weight=edge_weight, mess_dropout=False)
                kg_emb_list.append(kg_emb[:self.n_items])
            
            kg_emb_stack = torch.stack(kg_emb_list, dim=0)  # [intent_size, n_items, emb_size]

        return final_user_emb, item_struc_emb_all, kg_emb_stack, user_intent_attn