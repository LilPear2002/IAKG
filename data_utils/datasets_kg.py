import torch.utils.data as data
from config.configurator import configs
import numpy as np
import random
import torch


class KGTrainDataset(data.Dataset):
    """Training dataset for knowledge graph-enhanced recommendation"""
    
    def __init__(self, train_cf_pairs, train_user_dict) -> None:
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
    """Test dataset for knowledge graph-enhanced recommendation"""
    
    def __init__(self, test_user_dict, train_user_dict) -> None:
        self.user_pos_lists = test_user_dict
        self.test_users = np.array(list(test_user_dict.keys()))
        self.user_history_lists = train_user_dict

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        return self.test_users[idx]


def generate_kg_batch(kg_dict, batch_size, highest_neg_idx):
    """Generate a batch of KG triplets for training (if needed)"""
    exist_heads = list(kg_dict.keys())
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        pos_triples = kg_dict[h]
        n_pos_triples = len(pos_triples)
        
        # Sample one positive triple
        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples)
        relation = pos_triples[pos_triple_idx][0]
        pos_tail = pos_triples[pos_triple_idx][1]
        
        batch_relation.append(relation)
        batch_pos_tail.append(pos_tail)

        # Sample one negative tail
        while True:
            neg_tail = np.random.randint(low=0, high=highest_neg_idx)
            if (relation, neg_tail) not in pos_triples:
                break
        batch_neg_tail.append(neg_tail)

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail
