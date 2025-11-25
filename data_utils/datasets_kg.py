"""
Dataset classes for knowledge graph-enhanced recommendation
"""

import torch.utils.data as data
from config.configurator import configs
import numpy as np


class KGTrainDataset(data.Dataset):
    """Training dataset for KG-enhanced recommendation"""
    
    def __init__(self, train_cf_pairs, train_user_dict):
        self.train_cf_pairs = train_cf_pairs
        self.train_user_dict = train_user_dict

    def sample_negs(self):
        """Sample negative items for each positive user-item pair"""
        self.negs = np.zeros(len(self.train_cf_pairs), dtype=np.int32)
        for i in range(len(self.train_cf_pairs)):
            u = self.train_cf_pairs[i][0]
            while True:
                neg_i = np.random.randint(configs['data']['item_num'])
                if neg_i not in self.train_user_dict[u]:
                    break
            self.negs[i] = neg_i

    def __len__(self):
        return len(self.train_cf_pairs)

    def __getitem__(self, idx):
        # Return (user, pos_item, neg_item)
        return self.train_cf_pairs[idx][0], self.train_cf_pairs[idx][1], self.negs[idx]


class KGTestDataset(data.Dataset):
    """Test dataset for KG-enhanced recommendation"""
    
    def __init__(self, test_user_dict, train_user_dict):
        self.user_pos_lists = test_user_dict
        self.test_users = np.array(list(test_user_dict.keys()))
        self.user_history_lists = train_user_dict

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        return self.test_users[idx]
