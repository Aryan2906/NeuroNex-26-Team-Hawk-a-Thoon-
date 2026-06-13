
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OASISFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        print(f"Dataset Initialized: Loaded {len(self.image_paths)} images from {len(self.classes)} classes.")
        print(f"Class Mapping: {self.class_to_idx}")