import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os


def visualize_dataset_example(dataset, index=0):
    image, masks = dataset[index]
    class_to_idx = dataset.class_to_idx
    
    image = image.permute(1, 2, 0).numpy()
    masks = [mask.numpy() for mask in masks]

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    colormaps = [plt.cm.binary, plt.cm.viridis, plt.cm.viridis]
    label_mappings = [
        {0: 'background', 1: 'body'},
        {0: 'background', 1: 'upper body', 2: 'lower body'},
        {v: k for k, v in class_to_idx.items()}  # Invert class_to_idx for level 3
    ]
    
    titles = ['Level 1 Mask', 'Level 2 Mask', 'Level 3 Mask']
    for i, (mask, title, cmap, labels) in enumerate(zip(masks, titles, colormaps, label_mappings)):
        ax = axs[(i+1)//2, (i+1)%2]
        
        unique_values = np.unique(mask)
        color_map = {val: cmap(idx / len(unique_values)) for idx, val in enumerate(unique_values)}
        
        rgb_mask = np.apply_along_axis(lambda x: color_map[x[0]], 2, mask.reshape(*mask.shape, 1))
        ax.imshow(rgb_mask)
        ax.set_title(title)
        ax.axis('off')
        
        legend_elements = [Patch(facecolor=color_map[k], edgecolor='black', label=f'{k}: {labels[k]}')
                           for k in unique_values]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs("./media", exist_ok=True)
    plt.savefig(f"./media/sample-{index}.png")
