"""
Data handler for knowledge graph-enhanced recommendation
"""

import torch
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from os import path
from collections import defaultdict
from tqdm import tqdm
from .datasets_kg import KGTrainDataset, KGTestDataset


class DataHandlerKG:
    def __init__(self) -> None:
        if configs['data']['name'] == 'mind':
            predir = './datasets/kg/mind_kg/'
        elif configs['data']['name'] == 'alibaba-fashion':
            predir = './datasets/kg/alibaba-fashion_kg/'
        elif configs['data']['name'] == 'last-fm':
            predir = './datasets/kg/last-fm_kg/'
        elif configs['data']['name'] == 'mooc':
            predir = './datasets/kg/mooccube_kg/'
        else:
            raise ValueError(f"Unknown dataset: {configs['data']['name']}")

        configs['data']['dir'] = predir
        self.trn_file = path.join(predir, 'train.txt')
        self.val_file = path.join(predir, 'test.txt')
        self.tst_file = path.join(predir, 'test.txt') 
        self.kg_file = path.join(predir, 'kg_final.txt')   
        self.train_user_dict = defaultdict(list)
        self.test_user_dict = defaultdict(list)

    def _read_cf(self, file_name):
        """Read collaborative filtering data from file"""
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
        return np.array(inter_mat)
    
    def _collect_ui_dict(self, train_data, test_data):
        """Collect user-item interaction dictionaries"""
        n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
        n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1
        configs['data']['user_num'] = n_users
        configs['data']['item_num'] = n_items

        for u_id, i_id in train_data:
            self.train_user_dict[int(u_id)].append(int(i_id))
        for u_id, i_id in test_data:
            self.test_user_dict[int(u_id)].append(int(i_id))

    def _read_triplets(self, file_name):
        """Read knowledge graph triplets from file"""
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
        n_nodes = n_entities + configs['data']['user_num']
        n_relations = max(triplets[:, 1]) + 1

        configs['data']['entity_num'] = n_entities
        configs['data']['node_num'] = n_nodes
        configs['data']['relation_num'] = n_relations
        configs['data']['triplet_num'] = len(triplets)

        return triplets

    def _build_graphs(self, train_data, triplets):
        """Build knowledge graph and user-item interaction graph"""
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        # u, i
        ui_edges = list()

        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(train_data, ascii=True):
            ui_edges.append([u_id, i_id])

        print("Begin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            # h,t,r
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))

        return kg_edges, ui_edges, kg_dict

    def _build_ui_mat(self, ui_edges):
        """Build user-item interaction sparse matrix"""
        n_users = configs['data']['user_num']
        n_items = configs['data']['item_num']
        cf_edges = np.array(ui_edges)
        vals = [1.] * len(cf_edges)
        mat = sp.coo_matrix((vals, (cf_edges[:, 0], cf_edges[:, 1])), shape=(n_users, n_items))
        return mat

    def load_data(self):
        """Load all data including CF data and KG data"""
        train_cf = self._read_cf(self.trn_file)
        test_cf = self._read_cf(self.tst_file)
        self._collect_ui_dict(train_cf, test_cf)
        kg_triplets = self._read_triplets(self.kg_file)
        self.kg_edges, self.ui_edges, self.kg_dict = self._build_graphs(train_cf, kg_triplets)
        self.ui_mat = self._build_ui_mat(self.ui_edges)

        test_data = KGTestDataset(self.test_user_dict, self.train_user_dict)
        self.test_dataloader = data.DataLoader(
            test_data, 
            batch_size=configs['test']['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        train_data = KGTrainDataset(train_cf, self.train_user_dict)
        self.train_dataloader = data.DataLoader(
            train_data, 
            batch_size=configs['train']['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
