import os
import torch
import numpy as np
from dataset import setup_eval_dataloader
import albumentations as A
from models.hieraseg import HierarchicalSegmentationModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse


def visualize_predictions(image, gt_masks, pred_masks, save_path, idx):
    """
    Visualize ground truth and predicted masks for all levels
    """

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    
    colormaps = [plt.cm.binary, plt.cm.viridis, plt.cm.viridis]
    label_mappings = [
        {0: 'background', 1: 'body'},
        {0: 'background', 1: 'upper body', 2: 'lower body'},
        {0: 'background', 1: 'low_hand', 2: 'torso', 3: 'low_leg', 
         4: 'head', 5: 'up_leg', 6: 'up_hand'}
    ]
    
    if isinstance(image, torch.Tensor):
        image = image.cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
    
    for i in range(3):
        axs[i, 0].imshow(image)
        axs[i, 0].set_title(f'Original Image (Level {i+1})')
        axs[i, 0].axis('off')
    
    # Plot masks for each level
    for level, (gt_mask, pred_mask, cmap, labels) in enumerate(zip(gt_masks, pred_masks, colormaps, label_mappings)):
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
            
        unique_values = np.unique(gt_mask)
        color_map = {val: cmap(idx / max(3, len(unique_values))) 
                    for idx, val in enumerate(unique_values)}
        
        rgb_mask = np.apply_along_axis(
            lambda x: color_map[x[0]], 2, 
            gt_mask.reshape(*gt_mask.shape, 1)
        )
        axs[level, 1].imshow(rgb_mask)
        axs[level, 1].set_title(f'Ground Truth (Level {level+1})')
        axs[level, 1].axis('off')
        
        legend_elements = [
            Patch(facecolor=color_map[k], edgecolor='black', 
                  label=f'{k}: {labels[k]}') for k in unique_values
        ]
        axs[level, 1].legend(
            handles=legend_elements, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )
        
        unique_values = np.unique(pred_mask)
        color_map = {val: cmap(idx / max(3, len(unique_values))) 
                    for idx, val in enumerate(unique_values)}
        
        rgb_mask = np.apply_along_axis(
            lambda x: color_map[x[0]], 2, 
            pred_mask.reshape(*pred_mask.shape, 1)
        )
        axs[level, 2].imshow(rgb_mask)
        axs[level, 2].set_title(f'Prediction (Level {level+1})')
        axs[level, 2].axis('off')
        
        legend_elements = [
            Patch(facecolor=color_map[k], edgecolor='black', 
                  label=f'{k}: {labels[k]}') for k in unique_values
        ]
        axs[level, 2].legend(
            handles=legend_elements, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'prediction_{idx}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def calculate_iou(pred_mask, gt_mask, n_classes):
    """
    Calculate IoU for each class
    """
    ious = []
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    
    for cls in range(1, n_classes):
        pred_inds = pred_mask == cls
        gt_inds = gt_mask == cls
        
        intersection = (pred_inds & gt_inds).sum()
        union = (pred_inds | gt_inds).sum()
        
        if union == 0:
            ious.append(float('nan'))  # Class not present in ground truth
        else:
            ious.append(intersection / union)
    
    return ious

def evaluate(model, test_loader, device, visualize=False, num_viz=0, viz_dir=None):
    """
    Evaluate model on test set and optionally visualize predictions
    """
    model.eval()
    
    level1_ious, level2_ious, level3_ious = [], [], []
    viz_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch[0].to(device)
            gt_masks = [mask.to(device) for mask in batch[1]]
            
            # Get predictions
            pred_level1, pred_level2, pred_level3 = model(images)
            
            pred_level1 = torch.argmax(pred_level1, dim=1)
            pred_level2 = torch.argmax(pred_level2, dim=1)
            pred_level3 = torch.argmax(pred_level3, dim=1)
            
            for i in range(len(images)):
                level1_iou = calculate_iou(pred_level1[i], gt_masks[0][i], n_classes=2)
                level2_iou = calculate_iou(pred_level2[i], gt_masks[1][i], n_classes=3)
                level3_iou = calculate_iou(pred_level3[i], gt_masks[2][i], n_classes=7)
                
                level1_ious.extend(level1_iou)
                level2_ious.extend(level2_iou)
                level3_ious.extend(level3_iou)
                
                if visualize and viz_count < num_viz:
                    visualize_predictions(
                        images[i].cpu(),  # Move to CPU before visualization
                        [gt_mask[i].cpu() for gt_mask in gt_masks],  # Move to CPU
                        [pred_level1[i].cpu(), pred_level2[i].cpu(), pred_level3[i].cpu()],  # Move to CPU
                        viz_dir,
                        viz_count
                    )
                    viz_count += 1
    
    miou_level1 = np.nanmean(level1_ious)
    miou_level2 = np.nanmean(level2_ious)
    miou_level3 = np.nanmean(level3_ious)
    
    return {
        'mIoU_level1': miou_level1,
        'mIoU_level2': miou_level2, 
        'mIoU_level3': miou_level3
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate hierarchical segmentation model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--experiment_name', type=str, default='Test', help='Experiment name')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of predictions')
    parser.add_argument('--num_viz', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--viz_dir', type=str, default=None, 
                        help='Directory to save visualizations (default: logs/experiment_name/viz)')
    args = parser.parse_args()

    # Configuration
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'batch_size': args.batch_size,
        'num_workers': 4,
        'image_size': 224,
        'normalize_mean': (0.485, 0.456, 0.406),
        'normalize_std': (0.229, 0.224, 0.225)
    }
    
    if args.visualize:
        if args.viz_dir is None:
            args.viz_dir = os.path.join('logs', args.experiment_name, 'viz')
        os.makedirs(args.viz_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {args.viz_dir}")

    test_loader = setup_eval_dataloader()
    
    model = HierarchicalSegmentationModel().to(config['device'])
    
    # Load checkpoint
    checkpoint_path = os.path.join("logs", args.experiment_name, "model_best.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Evaluate
    metrics = evaluate(
        model, 
        test_loader, 
        config['device'],
        visualize=args.visualize,
        num_viz=args.num_viz,
        viz_dir=args.viz_dir
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"mIoU Level 1 (body): {metrics['mIoU_level1']:.4f}")
    print(f"mIoU Level 2 (upper/lower body): {metrics['mIoU_level2']:.4f}")
    print(f"mIoU Level 3 (detailed parts): {metrics['mIoU_level3']:.4f}")

if __name__ == "__main__":
    """
    # Just evaluate metrics
    python eval.py --experiment_name Test

    # Evaluate and visualize 5 predictions
    python eval.py --experiment_name Test --visualize --num_viz 5

    # Custom visualization directory
    python eval.py --experiment_name Test --visualize --num_viz 10 --viz_dir ./my_results
    """
    main()