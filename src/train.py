#!/usr/bin/env python3
"""
Training Script for SemiCon AI Hackathon - KLA Image Restoration
Reproduces the training process from scratch.
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from data import get_dataloaders
from model import create_model
from losses import CombinedLoss


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'config': config
    }
    
    last_path = os.path.join(config['paths']['checkpoint_dir'], config['paths']['last_model_name'])
    torch.save(checkpoint, last_path)
    
    if is_best:
        best_path = os.path.join(config['paths']['checkpoint_dir'], config['paths']['best_model_name'])
        torch.save(checkpoint, best_path)
        print(f"  >>> Best model saved (metric: {best_metric:.4f})")


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    psnr_val = psnr_fn(target_np, pred_np, data_range=1.0)
    
    ssim_vals = []
    for i in range(pred_np.shape[0]):
        ssim_val = ssim_fn(target_np[i, 0], pred_np[i, 0], data_range=1.0)
        ssim_vals.append(ssim_val)
    ssim_val = np.mean(ssim_vals)
    
    return {'psnr': psnr_val, 'ssim': ssim_val}


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, epoch, config, writer):
    model.train()
    total_loss = 0
    loss_components = {}
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (lr, gt, _) in enumerate(pbar):
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if config['training']['mixed_precision']:
            with torch.cuda.amp.autocast():
                pred = model(lr)
                pred = torch.clamp(pred, 0.0, 1.0)
                loss, losses = loss_fn(pred, gt)
            
            scaler.scale(loss).backward()
            if config['training']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss, losses = loss_fn(pred, gt)
            loss.backward()
            if config['training']['grad_clip'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
        
        total_loss += loss.item()
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            loss_components[k] = loss_components.get(k, 0) + val
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        step = epoch * num_batches + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), step)
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            writer.add_scalar(f'train/{k}', val, step)
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    return avg_loss, avg_components


def validate(model, val_loader, loss_fn, device, epoch, writer):
    model.eval()
    total_loss = 0
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for lr, gt, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            lr = lr.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            
            pred = model(lr)
            pred = torch.clamp(pred, 0.0, 1.0)
            
            loss, _ = loss_fn(pred, gt)
            total_loss += loss.item()
            
            metrics = compute_metrics(pred, gt)
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/psnr', avg_psnr, epoch)
    writer.add_scalar('val/ssim', avg_ssim, epoch)
    
    print(f"  Val Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    return avg_loss, avg_psnr, avg_ssim


def main():
    config = load_config("config.yaml")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    set_seed(config['training']['seed'])
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    model = create_model(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    
    loss_fn = CombinedLoss(config['loss']).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']
    )
    
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] and device.type == 'cuda' else None
    
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    start_epoch = 0
    best_psnr = 0
    resume_path = os.path.join(config['paths']['checkpoint_dir'], config['paths']['last_model_name'])
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_metric']
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    print(f"\nStarting training from epoch {start_epoch} to {config['training']['epochs']}")
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, epoch, config, writer
        )
        
        val_loss, val_psnr, val_ssim = validate(model, val_loader, loss_fn, device, epoch, writer)
        
        scheduler.step()
        
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        for k, v in train_components.items():
            writer.add_scalar(f'train/epoch_{k}', v, epoch)
        
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
        save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, config, is_best)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val PSNR={val_psnr:.2f} | Time={epoch_time:.1f}s")
    
    writer.close()
    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()