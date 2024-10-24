import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import torch
import os


def save_checkpoint(model, epoch, logdir, filename="model.pt", best_loss=0):
    state_dict = model.state_dict()

    save_dict = {
        "epoch": epoch,
        "best_loss": best_loss,
        "state_dict": state_dict
    }

    filename = os.path.join(logdir, filename)
    torch.save(save_dict, filename)

    print("Saving checkpoint", filename)
    return None


def visualize_batch_example(batch_data, batch_idx=0, save_path="./media"):
    """
    Visualize a single example from a batch of data
    
    Args:
        batch_data: tuple of (images, masks) from dataloader
            images: tensor of shape [batch, 3, height, width]
            masks: list of tensors, each of shape [batch, height, width]
        batch_idx: which item in the batch to visualize
        save_path: directory to save visualization
    """
    image = batch_data[0][batch_idx]  # [3, H, W]
    masks = [mask[batch_idx] for mask in batch_data[1]]  # list of [H, W]
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)  # Clip values to valid range
    
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    colormaps = [plt.cm.binary, plt.cm.viridis, plt.cm.viridis]
    label_mappings = [
        {0: 'background', 1: 'body'},
        {0: 'background', 1: 'upper body', 2: 'lower body'},
        {0: 'background', 1: 'low_hand', 2: 'torso', 3: 'low_leg', 
         4: 'head', 5: 'up_leg', 6: 'up_hand'}
    ]
    
    titles = ['Level 1 Mask', 'Level 2 Mask', 'Level 3 Mask']
    for i, (mask, title, cmap, labels) in enumerate(zip(masks, titles, colormaps, label_mappings)):
        ax = axs[(i+1)//2, (i+1)%2]
        
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        unique_values = np.unique(mask)
        color_map = {val: cmap(idx / max(3, len(unique_values))) 
                    for idx, val in enumerate(unique_values)}
        
        rgb_mask = np.apply_along_axis(
            lambda x: color_map[x[0]], 2, 
            mask.reshape(*mask.shape, 1)
        )
        
        ax.imshow(rgb_mask)
        ax.set_title(title)
        ax.axis('off')
        
        legend_elements = [
            Patch(facecolor=color_map[k], edgecolor='black', 
                  label=f'{k}: {labels[k]}') for k in unique_values
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'batch_sample_{batch_idx}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()


def calculate_miou(confusion_matrix):
    """
    Calculate mean IoU from confusion matrix, excluding background class (index 0)
    """
    confusion_matrix = confusion_matrix.cpu().numpy()
    intersection = np.diag(confusion_matrix)[1:]  # Exclude background
    union = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))[1:]
    
    valid_mask = union > 0
    iou = np.zeros_like(union, dtype=float)
    iou[valid_mask] = intersection[valid_mask] / union[valid_mask]
    
    return np.mean(iou[valid_mask]) if np.any(valid_mask) else 0.0
