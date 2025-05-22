import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from tqdm import tqdm
from os import makedirs, path
from typing import Optional, Dict, Tuple, List
import lightning.pytorch as pl
from lidiff.utils.collations import feats_to_coord
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from einops import rearrange
import math


class ImprovedDiffusionPoints(pl.LightningModule):
    """Improved diffusion model for 3D point cloud completion with modern techniques."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Model configuration
        self.model_config = self.hparams['model']
        self.diff_config = self.hparams['diff']
        self.train_config = self.hparams['train']
        self.data_config = self.hparams['data']
        
        # Initialize scheduler with modern diffusers library
        self.noise_scheduler = self._init_scheduler()
        
        # Model architecture with improved channel configuration
        in_ch_full = self.model_config.get('in_channels_full', 3)
        in_ch_part = self.model_config.get('in_channels_part', 4)  # xyz + intensity
        out_dim = self.model_config['out_dim']
        
        # Enhanced encoder for partial point clouds
        self.partial_enc = minknet.MinkGlobalEnc(
            in_channels=in_ch_part, 
            out_channels=out_dim
        )
        
        # Main diffusion U-Net with timestep conditioning
        self.model = minknet.MinkUNetDiff(
            in_channels=in_ch_full, 
            out_channels=out_dim
        )
        
        # Learnable positional encoding for better spatial awareness
        self.time_embed = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        
        # Metrics
        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(
            self.data_config['resolution'],
            2 * self.data_config['resolution'],
            100
        )
        
        # Classifier-free guidance scale
        self.guidance_scale = self.train_config.get('uncond_w', 6.0)
        self.uncond_prob = self.train_config.get('uncond_prob', 0.1)
        
        # Memory optimization flag
        self.enable_memory_efficient_attention = True
        
    def _init_scheduler(self) -> SchedulerMixin:
        """Initialize modern diffusion scheduler."""
        scheduler_type = self.diff_config.get('scheduler_type', 'ddpm')
        
        scheduler_args = {
            'num_train_timesteps': self.diff_config['t_steps'],
            'beta_start': self.diff_config['beta_start'],
            'beta_end': self.diff_config['beta_end'],
            'beta_schedule': self.diff_config.get('beta_func', 'linear'),
        }
        
        if scheduler_type == 'ddpm':
            return DDPMScheduler(**scheduler_args)
        elif scheduler_type == 'ddim':
            return DDIMScheduler(**scheduler_args)
        elif scheduler_type == 'dpm':
            return DPMSolverMultistepScheduler(
                **scheduler_args,
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
            )
        elif scheduler_type == 'euler':
            return EulerDiscreteScheduler(**scheduler_args)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def points_to_tensor(
        self, 
        coords: torch.Tensor, 
        mean: torch.Tensor, 
        std: torch.Tensor, 
        intensity: Optional[torch.Tensor] = None
    ) -> ME.TensorField:
        """Convert point coordinates to MinkowskiEngine tensor with proper memory management."""
        batch_size = coords.shape[0]
        
        # Prepare coordinates
        coords_list = list(coords)
        quant_coords = ME.utils.batched_coordinates(
            coords_list, dtype=torch.float32, device=self.device
        )
        
        # Quantize coordinates
        quant = quant_coords.clone()
        quant[:, 1:] = feats_to_coord(
            quant_coords[:, 1:], 
            self.data_config['resolution'], 
            mean, 
            std
        )
        
        # Prepare features
        coords_flat = torch.cat(coords_list, dim=0)
        if intensity is not None:
            intens_flat = torch.cat(list(intensity), dim=0)
            feats = torch.cat([coords_flat, intens_flat.unsqueeze(-1)], dim=1)
        else:
            feats = coords_flat
            
        # Create tensor field without immediate cache clearing
        x_t = ME.TensorField(
            features=feats,
            coordinates=quant,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        
        return x_t
    
    def forward(
        self, 
        x_full: ME.TensorField, 
        x_full_sparse: ME.SparseTensor, 
        x_part: ME.TensorField, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with improved memory efficiency."""
        # Encode partial point cloud
        part_feat = self.partial_enc(x_part)
        
        # Add timestep embedding
        t_emb = self.get_timestep_embedding(t.float(), self.model_config['out_dim'])
        t_emb = self.time_embed(t_emb.unsqueeze(-1)).squeeze(-1)
        
        # Forward through main model
        out = self.model(x_full, x_full_sparse, part_feat, t)
        
        return out.reshape(t.shape[0], -1, 3)
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Improved training step with variance-preserving noise schedule."""
        # Sample timesteps
        batch_size = batch['pcd_full'].shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to point clouds
        noise = torch.randn_like(batch['pcd_full'])
        noisy_points = self.noise_scheduler.add_noise(
            batch['pcd_full'], noise, timesteps
        )
        
        # Create tensor fields
        x_full = self.points_to_tensor(noisy_points, batch['mean'], batch['std'])
        
        # Classifier-free guidance dropout
        if torch.rand(1) > self.uncond_prob:
            x_part = self.points_to_tensor(
                batch['pcd_part'], batch['mean'], batch['std'], batch['intensity_part']
            )
        else:
            # Unconditional training
            x_part = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), 
                torch.zeros_like(batch['mean']), 
                torch.zeros_like(batch['std']),
                torch.zeros_like(batch['intensity_part'])
            )
        
        # Predict noise
        noise_pred = self.forward(x_full, x_full.sparse(), x_part, timesteps)
        
        # Compute losses
        loss_mse = F.mse_loss(noise_pred, noise)
        
        # Additional regularization losses
        loss_mean = noise_pred.mean().pow(2)
        loss_std = (noise_pred.std() - 1.0).pow(2)
        
        # Total loss with adaptive weighting
        reg_weight = self.diff_config.get('reg_weight', 5.0)
        loss = loss_mse + reg_weight * (loss_mean + loss_std)
        
        # Logging
        self.log('train/loss_mse', loss_mse, prog_bar=True)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        x_init: torch.Tensor,
        x_cond: ME.TensorField,
        batch_mean: torch.Tensor,
        batch_std: torch.Tensor,
        num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Improved sampling with classifier-free guidance."""
        if num_inference_steps is None:
            num_inference_steps = self.diff_config.get('s_steps', 50)
            
        # Set timesteps for sampling
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # Initialize from noise
        x_t = x_init + torch.randn_like(x_init)
        
        # Create unconditional input
        x_uncond = self.points_to_tensor(
            torch.zeros_like(x_init),
            torch.zeros_like(batch_mean),
            torch.zeros_like(batch_std)
        )
        
        # Sampling loop
        for t in tqdm(self.noise_scheduler.timesteps):
            # Create tensor field for current state
            x_t_field = self.points_to_tensor(x_t, batch_mean, batch_std)
            x_t_sparse = x_t_field.sparse()
            
            # Classifier-free guidance
            noise_cond = self.forward(x_t_field, x_t_sparse, x_cond, t.unsqueeze(0))
            noise_uncond = self.forward(x_t_field, x_t_sparse, x_uncond, t.unsqueeze(0))
            
            # Apply guidance
            noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            
            # Denoise step
            x_t = self.noise_scheduler.step(
                noise_pred.squeeze(0), t, x_t.squeeze(0)
            ).prev_sample.unsqueeze(0)
            
            # Clear intermediate tensors
            del x_t_field, x_t_sparse
            
        return x_t
    
    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, float]:
        """Validation with improved metrics."""
        if batch_idx > 0:  # Only validate on first batch
            return
            
        self.eval()
        
        # Generate completions
        x_init = batch['pcd_part'].repeat(1, 10, 1)
        x_cond = self.points_to_tensor(
            batch['pcd_part'], batch['mean'], batch['std'], batch['intensity_part']
        )
        
        completed = self.sample(x_init, x_cond, batch['mean'], batch['std'])
        completed = completed.reshape(batch['pcd_full'].shape[0], -1, 3)
        
        # Compute metrics
        for i in range(len(batch['pcd_full'])):
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(
                completed[i].cpu().numpy()
            )
            
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(
                batch['pcd_full'][i].cpu().numpy()
            )
            
            self.chamfer_distance.update(pcd_gt, pcd_pred)
            self.precision_recall.update(pcd_gt, pcd_pred)
        
        # Log metrics
        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        
        self.log('val/cd_mean', cd_mean, prog_bar=True)
        self.log('val/cd_std', cd_std)
        self.log('val/precision', pr)
        self.log('val/recall', re)
        self.log('val/fscore', f1, prog_bar=True)
        
        return {
            'val/cd_mean': cd_mean,
            'val/cd_std': cd_std,
            'val/precision': pr,
            'val/recall': re,
            'val/fscore': f1
        }
    
    def configure_optimizers(self):
        """Optimizer configuration with modern techniques."""
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config['lr'],
            betas=(0.9, 0.999),
            weight_decay=self.train_config.get('weight_decay', 1e-4)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/fscore",
                "interval": "epoch",
                "frequency": 1
            }
        }