#!/usr/bin/env python3
"""
Evaluation Script for SemiCon AI Hackathon - KLA Image Restoration
This script will be used by KLA's benchmarking team to evaluate submissions.
It MUST run without manual edits.

Usage:
    python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/outputs --model_path /path/to/model.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from model import SwinIR


def load_model(model_path: str, device: torch.device) -> SwinIR:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']['model']
    else:
        config = {
            'in_chans': 1,
            'embed_dim': 60,
            'depths': [6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6],
            'window_size': 8,
            'mlp_ratio': 2.0,
            'scale': 2,
            'img_range': 1.0,
            'upsampler': 'pixelshuffle'
        }
    
    model = SwinIR(
        img_size=128,
        patch_size=1,
        in_chans=config['in_chans'],
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=config['window_size'],
        mlp_ratio=config['mlp_ratio'],
        upscale=config['scale'],
        img_range=config['img_range'],
        upsampler=config['upsampler']
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def process_image(lr_path: str, model: SwinIR, device: torch.device, output_dir: str):
    """Process a single LR image and save restored output."""
    # Load LR image
    lr = np.load(lr_path).astype(np.float32)  # (128, 128)
    
    # Add batch and channel dims
    lr_tensor = torch.from_numpy(lr[None, None, ...]).float().to(device)  # (1, 1, 128, 128)
    
    # Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            pred = model(lr_tensor)
            pred = torch.clamp(pred, 0.0, 1.0)
    
    # Convert to numpy
    pred_np = pred.cpu().numpy()[0, 0]  # (256, 256)
    
    # Save
    filename = os.path.basename(lr_path)
    output_path = os.path.join(output_dir, filename)
    np.save(output_path, pred_np.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="SemiCon AI Hackathon - Image Restoration Evaluation")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to test NoisyLR directory (contains .npy files)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for restored images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda/cpu)')
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model checkpoint not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)
    print(f"Model loaded successfully")
    
    # Get test files
    test_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy')])
    print(f"Found {len(test_files)} test images")
    
    # Process all images
    print("Running inference...")
    for fname in tqdm(test_files, desc="Processing"):
        lr_path = os.path.join(args.input_dir, fname)
        process_image(lr_path, model, device, args.output_dir)
    
    print(f"\nDone! Restored images saved to: {args.output_dir}")
    
    # Verify outputs
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.npy')]
    print(f"Generated {len(output_files)} output files")
    
    # Quick sanity check on first output
    if output_files:
        sample = np.load(os.path.join(args.output_dir, output_files[0]))
        print(f"Sample output shape: {sample.shape}, range: [{sample.min():.4f}, {sample.max():.4f}]")
        assert sample.shape == (256, 256), f"Expected 256x256, got {sample.shape}"


if __name__ == "__main__":
    main()