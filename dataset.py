import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PascalPartDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        split: str = "train", "val" or "holdout" - choose split to partition dataset
        """

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(root_dir, "train_id.txt"), "r") as f:
            train_val_ids = [line.strip() for line in f]
            
        with open(os.path.join(root_dir, "val_id.txt"), "r") as f:
            holdout_ids = [line.strip() for line in f]

        if split == "train":
            train_idcs = list(range(0, 2400))  # apprx 80 % of dataset for train and 20 for val
            self.image_ids = [train_val_ids[i] for i in train_idcs]
        elif split == "val":
            val_idcs = list(range(2400, 2826))
            self.image_ids = [train_val_ids[i] for i in val_idcs]
        elif split == "holdout":
            self.image_ids = holdout_ids
        else:
            raise ValueError(f"Invalid split {split}, choose train, val or holdout")


        
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

import pdb
def setup_dataloaders():
    height = 224
    width = 224

    train_transf = A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.GaussianBlur(blur_limit=3, p=1),
            ], p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )
    
    val_transf = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )
    #pdb.set_trace()
    train_ds = PascalPartDataset(root_dir="/root/data", split="train", transform=train_transf)
    val_ds = PascalPartDataset(root_dir="/root/data", split="val", transform=val_transf)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=32
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=32,
    )
    #pdb.set_trace()
    return train_loader, val_loader


def setup_eval_dataloader():
    height = 256
    width = 256

    val_transf = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )

    holdout_dataset = PascalPartDataset(
        root_dir="/root/data",
        split='holdout',
        transform=val_transf
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=1,
        shuffle=False,
    )

    return holdout_loader

