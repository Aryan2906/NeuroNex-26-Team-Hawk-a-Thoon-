import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OASISFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_size = image_size

        self.classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

        print(
            f"Dataset Initialized: Loaded {len(self.image_paths)} images from {len(self.classes)} classes."
        )
        print(f"Class Mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"could not read image at {img_path}")
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = img.astype(np.float32) / 255.0

        img_tensor = torch.tensor(img).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


if __name__ == "__main__":
    path = "./Data"

    if not os.path.exists(path):
        print(f"Error: directory {path} was not found. ")
    else:
        dataset = OASISFolderDataset(root_dir=path, image_size=(128, 128))
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    print("\nFetching the first batch to verify tensor shapes and values...")
    images, labels = next(iter(loader))

    print("\n--- Pipeline Verification Results ---")
    print(f"Batch Image Tensor Shape: {images.shape}")
    print(f"Batch Label Tensor Shape: {labels.shape}")
    print(f"Max Pixel Value (Must be <= 1.0): {images.max().item():.4f}")
    print(f"Min Pixel Value (Must be >= 0.0): {images.min().item():.4f}")
    print("Preprocessing Phase 1 is complete and ready for SNN Neuromorphic Encoding.")
