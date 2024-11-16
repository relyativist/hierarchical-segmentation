import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_levels=[2, 3, 7], level_weights=[0.1, 0.4, 1.0]):
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
    def __init__(self, gamma=2.0):
        """
        Tree-Min Loss implementation with improved weighting and stability
        Args:
            gamma: Focal loss focusing parameter (default: 2.0)
        """
        super().__init__()
        self.gamma = gamma
        
        # Define hierarchy levels
        self.coarse_weight = 1.0    # 7 classes (detailed parts)
        self.middle_weight = 0.5    # 3 classes (body regions)
        self.fine_weight = 0.25     # 2 classes (foreground/background)
        
        # Define part groupings
        self.upper_body_parts = [1, 2, 3, 4]  # head, torso, upper arms
        self.lower_body_parts = [5, 6]        # legs
        self.background_idx = 0

    def compute_level_loss(self, pred, target, valid_mask, num_classes, eps=1e-6):
        """Helper function to compute BCE loss with focal weighting"""
        # Convert target to one-hot
        target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Compute focal weights
        pt = torch.where(target_onehot == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute BCE loss with focal weighting
        bce = -(target_onehot * focal_weight * torch.log(pred_prob + eps) + 
                (1 - target_onehot) * focal_weight * torch.log(1 - pred_prob + eps))
        
        # Apply valid mask and normalize
        valid_mask = valid_mask.float()
        num_valid = valid_mask.sum() + eps
        loss = (bce * valid_mask.unsqueeze(1)).sum() / (num_valid * num_classes)
        
        return loss

    def compute_hierarchy_loss(self, pred_parent, pred_children, children_indices, valid_mask, eps=1e-6):
        """Compute hierarchical consistency loss"""
        # Get probabilities
        parent_prob = torch.sigmoid(pred_parent)
        children_prob = torch.sigmoid(pred_children[:, children_indices])
        
        # Maximum probability of children should be <= parent probability
        max_children_prob = torch.max(children_prob, dim=1, keepdim=True)[0]
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(
            parent_prob,
            torch.maximum(parent_prob, max_children_prob),
            reduction='none'
        )
        
        # Apply valid mask and normalize
        valid_mask = valid_mask.float()
        num_valid = valid_mask.sum() + eps
        loss = (consistency_loss * valid_mask.unsqueeze(1)).sum() / num_valid
        
        return loss

    def forward(self, predictions, targets):
        """
        Forward pass
        Args:
            predictions: Tuple of (pred_fine, pred_middle, pred_coarse)
                pred_fine: [B, 2, H, W] - background/foreground
                pred_middle: [B, 3, H, W] - background/upper/lower
                pred_coarse: [B, 7, H, W] - detailed parts
            targets: List of target masks [target_fine, target_middle, target_coarse]
                Each target: [B, H, W] with class indices
        """
        pred_fine, pred_middle, pred_coarse = predictions
        target_fine, target_middle, target_coarse = targets
        
        # Create valid mask
        valid_mask = (target_fine != 255)
        
        # Compute basic level losses
        loss_coarse = self.compute_level_loss(
            pred_coarse, target_coarse, valid_mask, num_classes=7
        )
        
        loss_middle = self.compute_level_loss(
            pred_middle, target_middle, valid_mask, num_classes=3
        )
        
        loss_fine = self.compute_level_loss(
            pred_fine, target_fine, valid_mask, num_classes=2
        )
        
        # Compute hierarchical consistency losses
        # Upper body parts -> upper body region
        hier_loss_upper = self.compute_hierarchy_loss(
            pred_middle[:, 1:2],  # upper body logit
            pred_coarse,
            self.upper_body_parts,
            valid_mask
        )
        
        # Lower body parts -> lower body region
        hier_loss_lower = self.compute_hierarchy_loss(
            pred_middle[:, 2:3],  # lower body logit
            pred_coarse,
            self.lower_body_parts,
            valid_mask
        )
        
        # Body regions -> foreground
        hier_loss_body = self.compute_hierarchy_loss(
            pred_fine[:, 1:2],  # foreground logit
            pred_middle,
            [1, 2],  # upper and lower body indices
            valid_mask
        )
        
        # Background consistency
        bg_loss = F.mse_loss(
            torch.sigmoid(pred_fine[:, 0:1]),
            torch.sigmoid(pred_middle[:, 0:1]),
            reduction='none'
        )
        bg_loss = (bg_loss * valid_mask.unsqueeze(1).float()).mean()
        
        # Combine all losses with weights
        total_loss = (
            self.coarse_weight * loss_coarse +
            self.middle_weight * (loss_middle + 0.1 * (hier_loss_upper + hier_loss_lower)) +
            self.fine_weight * (loss_fine + 0.1 * hier_loss_body + 0.1 * bg_loss)
        )
        
        return total_loss