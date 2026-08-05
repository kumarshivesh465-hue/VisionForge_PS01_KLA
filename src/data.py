import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from typing import Tuple, Optional
import yaml


class SemiconDataset(Dataset):
    """Paired dataset for SemiCon AI Image Restoration."""
    
    def __init__(
        self,
        gt_dir: str,
        lr_dir: str,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        aug_config: dict = None
    ):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.split = split
        self.augment = augment and (split == "train")
        self.aug_config = aug_config or {}
        
        # Get all paired files
        self.gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.npy')])
        
        assert len(self.gt_files) == len(self.lr_files), "Mismatch in GT/LR counts"
        assert all(g == l for g, l in zip(self.gt_files, self.lr_files)), "Filename mismatch"
        
        # Train/val split
        random.seed(seed)
        indices = list(range(len(self.gt_files)))
        random.shuffle(indices)
        val_size = int(len(indices) * val_split)
        
        if split == "train":
            self.indices = indices[val_size:]
        else:
            self.indices = indices[:val_size]
        
        print(f"{split.upper()} dataset: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        gt_file = self.gt_files[file_idx]
        lr_file = self.lr_files[file_idx]
        
        # Load
        gt = np.load(os.path.join(self.gt_dir, gt_file)).astype(np.float32)  # (256, 256)
        lr = np.load(os.path.join(self.lr_dir, lr_file)).astype(np.float32)  # (128, 128)
        
        # Add channel dim
        gt = gt[None, ...]  # (1, 256, 256)
        lr = lr[None, ...]  # (1, 128, 128)
        
        # Augmentation (only on training)
        if self.augment:
            gt, lr = self._augment_pair(gt, lr)
        
        # Convert to tensor
        gt = torch.from_numpy(gt).float()
        lr = torch.from_numpy(lr).float()
        
        # LR already has values outside [0,1] - keep as is (model must handle this)
        # GT is in [0,1] - clamp for safety
        gt = torch.clamp(gt, 0.0, 1.0)
        
        return lr, gt, gt_file
    
    def _augment_pair(self, gt: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply synchronized augmentations to GT and LR pair."""
        # Geometric augmentations (must be synchronized)
        if random.random() < self.aug_config.get('hflip_prob', 0.5):
            gt = np.flip(gt, axis=2).copy()
            lr = np.flip(lr, axis=2).copy()
        
        if random.random() < self.aug_config.get('vflip_prob', 0.5):
            gt = np.flip(gt, axis=1).copy()
            lr = np.flip(lr, axis=1).copy()
        
        if random.random() < self.aug_config.get('rot90_prob', 0.5):
            k = random.randint(1, 3)
            gt = np.rot90(gt, k, axes=(1, 2)).copy()
            lr = np.rot90(lr, k, axes=(1, 2)).copy()
        
        # Photometric augmentations on GT only (LR already has real degradation)
        if random.random() < self.aug_config.get('brightness_prob', 0.2):
            factor = 1.0 + random.uniform(-self.aug_config.get('brightness_factor', 0.1),
                                           self.aug_config.get('brightness_factor', 0.1))
            gt = gt * factor
        
        if random.random() < self.aug_config.get('contrast_prob', 0.2):
            factor = 1.0 + random.uniform(-self.aug_config.get('contrast_factor', 0.1),
                                           self.aug_config.get('contrast_factor', 0.1))
            mean = gt.mean()
            gt = (gt - mean) * factor + mean
        
        gt = np.clip(gt, 0.0, 1.0)
        
        return gt, lr


class TestDataset(Dataset):
    """Test dataset (LR only, no GT)."""
    
    def __init__(self, lr_dir: str):
        self.lr_dir = lr_dir
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.npy')])
        print(f"TEST dataset: {len(self.lr_files)} samples")
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr_file = self.lr_files[idx]
        lr = np.load(os.path.join(self.lr_dir, lr_file)).astype(np.float32)
        lr = lr[None, ...]  # (1, 128, 128)
        lr = torch.from_numpy(lr).float()
        return lr, lr_file


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    train_ds = SemiconDataset(
        gt_dir=config['data']['train_gt_dir'],
        lr_dir=config['data']['train_lr_dir'],
        split="train",
        val_split=config['data']['val_split'],
        seed=config['training']['seed'],
        augment=True,
        aug_config=config['augmentation']
    )
    
    val_ds = SemiconDataset(
        gt_dir=config['data']['train_gt_dir'],
        lr_dir=config['data']['train_lr_dir'],
        split="val",
        val_split=config['data']['val_split'],
        seed=config['training']['seed'],
        augment=False
    )
    
    test_ds = TestDataset(
        lr_dir=config['data']['test_lr_dir']
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Test one batch
    lr, gt, fname = next(iter(train_loader))
    print(f"LR shape: {lr.shape}, range: [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"GT shape: {gt.shape}, range: [{gt.min():.4f}, {gt.max():.4f}]")
    print(f"Filename: {fname[0]}")