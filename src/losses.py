import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import lpips


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant with epsilon for stability)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features (grayscale adapted)."""
    def __init__(self, layer_weights: dict = None, use_lpips: bool = False, lpips_net: str = 'alex'):
        super().__init__()
        self.use_lpips = use_lpips
        if use_lpips:
            self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False)
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        else:
            # VGG-based perceptual (requires 3-channel input)
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
            self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
            self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
            self.slice3 = nn.Sequential(*list(vgg[9:18])) # relu3_4
            self.slice4 = nn.Sequential(*list(vgg[18:27]))# relu4_4
            for param in self.parameters():
                param.requires_grad = False
            
            self.layer_weights = layer_weights or {
                'relu1_2': 0.1, 'relu2_2': 0.1, 'relu3_4': 1.0, 'relu4_4': 1.0
            }
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_lpips:
            # LPIPS expects [-1, 1] range, 3 channels
            pred_3ch = pred.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
            pred_norm = pred_3ch * 2 - 1
            target_norm = target_3ch * 2 - 1
            return self.lpips_fn(pred_norm, target_norm).mean()
        else:
            # VGG perceptual (convert grayscale to 3-channel)
            pred_3ch = pred.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
            pred_norm = (pred_3ch - self.mean) / self.std
            target_norm = (target_3ch - self.mean) / self.std
            
            pred_feats = []
            target_feats = []
            
            x = pred_norm
            for slice_module in [self.slice1, self.slice2, self.slice3, self.slice4]:
                x = slice_module(x)
                pred_feats.append(x)
            
            x = target_norm
            for slice_module in [self.slice1, self.slice2, self.slice3, self.slice4]:
                x = slice_module(x)
                target_feats.append(x)
            
            loss = 0
            for i, (pf, tf) in enumerate(zip(pred_feats, target_feats)):
                layer_name = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'][i]
                weight = self.layer_weights.get(layer_name, 1.0)
                loss += weight * F.l1_loss(pf, tf)
            return loss


class FrequencyLoss(nn.Module):
    """Frequency domain loss using FFT."""
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # Magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        if self.loss_type == 'l1':
            mag_loss = F.l1_loss(pred_mag, target_mag)
            phase_loss = F.l1_loss(pred_phase, target_phase)
        else:
            mag_loss = F.mse_loss(pred_mag, target_mag)
            phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel gradients."""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        
        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-8)
        
        return F.l1_loss(pred_grad, target_grad)


class CombinedLoss(nn.Module):
    """Combined loss for image restoration."""
    def __init__(self, config: dict):
        super().__init__()
        self.l1_weight = config.get('l1_weight', 1.0)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        self.frequency_weight = config.get('frequency_weight', 0.05)
        self.edge_weight = config.get('edge_weight', 0.0)
        self.gan_weight = config.get('gan_weight', 0.0)
        
        self.l1_loss = CharbonnierLoss()
        
        # Only create perceptual loss if weight > 0
        if self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(
                use_lpips=False  # Use VGG for now
            )
        else:
            self.perceptual_loss = None
        
        self.frequency_loss = FrequencyLoss()
        self.edge_loss = EdgeLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        losses = {}
        total_loss = 0
        
        # L1 / Charbonnier
        if self.l1_weight > 0:
            l1 = self.l1_loss(pred, target)
            losses['l1'] = l1
            total_loss += self.l1_weight * l1
        
        # Perceptual (VGG)
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            perc = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc
            total_loss += self.perceptual_weight * perc
        
        # Frequency
        if self.frequency_weight > 0:
            freq = self.frequency_loss(pred, target)
            losses['frequency'] = freq
            total_loss += self.frequency_weight * freq
        
        # Edge
        if self.edge_weight > 0:
            edge = self.edge_loss(pred, target)
            losses['edge'] = edge
            total_loss += self.edge_weight * edge
        
        losses['total'] = total_loss
        return total_loss, losses


if __name__ == "__main__":
    import yaml
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    loss_fn = CombinedLoss(config['loss'])
    
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    
    total, losses = loss_fn(pred, target)
    print(f"Total loss: {total.item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")