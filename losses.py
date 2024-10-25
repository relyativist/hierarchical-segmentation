import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t
import math


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_levels=[2, 3, 7], level_weights=[0.2, 0.5, 1.0]):
        super().__init__()
        self.class_levels = class_levels
        self.level_weights = level_weights
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        total_loss = 0

        for pred, target, weight in zip(predictions, targets, self.level_weights):
            level_loss = self.ce_loss(pred, target)
            total_loss += weight * level_loss

        return total_loss
    

### Forked from
# https://github.com/lingorX/HieraSeg/blob/cc3c1cfbabe3cc2af620e0193a245822cca8a841/Pytorch/mmseg/models/losses/hiera_loss.py

class TreeMinLoss(nn.Module):
    """Implementation of Tree-Min Loss from the paper"""
    def __init__(self, hierarchy_levels=[2, 3, 7], gamma=2.0):
        super().__init__()
        self.hierarchy_levels = hierarchy_levels
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        total_loss = 0
        device = predictions[0].device
        
        # Process each hierarchy level
        for pred, target, num_classes in zip(predictions, targets, self.hierarchy_levels):
            batch_size = pred.size(0)
            
            # Convert to probabilities
            pred_probs = F.softmax(pred, dim=1)
            
            # Create one-hot encoded targets
            target_one_hot = F.one_hot(target, num_classes).float()
            target_one_hot = target_one_hot.permute(0, 3, 1, 2)
            
            # Calculate focal weights
            pt = torch.where(target_one_hot == 1, pred_probs, 1 - pred_probs)
            focal_weight = (1 - pt) ** self.gamma
            
            # Calculate loss with focal weighting
            log_probs = F.log_softmax(pred, dim=1)
            loss = -(target_one_hot * focal_weight * log_probs).sum(dim=1).mean()
            
            total_loss += loss
            
        return total_loss