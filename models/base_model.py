import torch as t
from torch import nn
from config.configurator import configs


class BaseModel(nn.Module):
    """Base model class for all recommendation models"""
    
    def __init__(self, data_handler):
        super(BaseModel, self).__init__()

        # Basic hyperparameters
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

    def forward(self):
        """Forward propagation, should return embeddings"""
        pass

    def cal_loss(self, batch_data):
        """Calculate loss for training
        
        Args:
            batch_data (tuple): a batch of training samples already in cuda
        
        Return:
            loss (0-d torch.Tensor): the overall weighted loss
            losses (dict): dict for specific terms of losses for printing
        """
        pass
    
    def _mask_predict(self, full_preds, train_mask):
        """Mask training pairs in prediction"""
        return full_preds * (1 - train_mask) - 1e8 * train_mask
    
    def full_predict(self, batch_data):
        """Return all-rank predictions for evaluation
        
        Args:
            batch_data (tuple): data in a test batch, e.g. batch_users, train_mask
        
        Return:
            full_preds (torch.Tensor): a [test_batch_size * item_num] prediction tensor
        """
        pass
