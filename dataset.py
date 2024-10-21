import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PascalPartDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(root_dir, f"{split}_id.txt"), "r") as f:
            self.image_ids = [line.strip() for line in f]
        
        self.class_levels = {
            "body": {
                "upper_body": [
                    "low_hand",
                    "up_hand",
                    "torso",
                    "head"],
                "lower_body": ["low_leg", "up_leg"]
            }
        }
        
        self.class_to_idx = self._load_class_mappings()

        self.upper_body_classes = set(self.class_to_idx[c] for c in self.class_levels['body']['upper_body'])
        self.lower_body_classes = set(self.class_to_idx[c] for c in self.class_levels['body']['lower_body'])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        
        img_id = self.image_ids[idx]

        img_path = os.path.join(self.root_dir, "JPEGImages", f"{img_id}.jpg")
        img = cv2.imread(img_path)
        
        mask_path = os.path.join(self.root_dir, "gt_masks", f"{img_id}.npy")
        mask = np.load(mask_path)
        
        #pdb.set_trace()
        level1_mask = (mask > 0).astype(np.uint8)
        level2_mask, level3_mask = self._create_level2_and_3_masks(mask)
        """
        masks = [
            np.expand_dims(level1_mask, axis=2),
            np.expand_dims(level2_mask, axis=2),
            np.expand_dims(level3_mask, axis=2)
        ]
        """
        masks = [
            level1_mask,
            level2_mask,
            level3_mask
        ]
        
        if self.transform:
            transformed = self.transform(image=img, masks=masks)
            img = transformed["image"]
            masks = transformed["masks"]
        
        return img, masks
    
    def _load_class_mappings(self):
        class_to_idx = {}
        with open(os.path.join(self.root_dir, "classes.txt"), "r") as f:
            
            for line in f:
                idx, name = line.strip().split(": ")
                class_to_idx[name] = int(idx)
            
        return class_to_idx

    
    def _create_level2_and_3_masks(self, mask):
        level2_mask = np.zeros_like(mask)
        level3_mask = np.zeros_like(mask)
        
        
        # Level 2 mask
        level2_mask[np.isin(mask, list(self.upper_body_classes))] = 1 
        level2_mask[np.isin(mask, list(self.lower_body_classes))] = 2  
        
        # Level 3 mask
        for class_name, idx in self.class_to_idx.items():
            if class_name != 'bg':
                level3_mask[mask == idx] = idx
        
        return level2_mask, level3_mask

def setup_dataset():
    train_transf = A.Compose([
            A.Resize(height=224, width=224, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ])
    dataset = PascalPartDataset(root_dir="/root/data", split="train", transform=train_transf)

    return dataset

