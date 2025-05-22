"""Utilities for improved diffusion model training and sampling."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Union, Tuple
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def get_scheduler(config: Dict) -> SchedulerMixin:
    """Create a scheduler from configuration."""
    scheduler_type = config.get('scheduler_type', 'ddpm').lower()
    
    # Common arguments for all schedulers
    common_args = {
        'num_train_timesteps': config['t_steps'],
        'beta_start': config['beta_start'],
        'beta_end': config['beta_end'],
        'beta_schedule': config.get('beta_func', 'linear'),
        'prediction_type': config.get('prediction_type', 'epsilon'),
        'clip_sample': config.get('clip_sample', True),
    }
    
    if 'clip_sample_range' in config:
        common_args['clip_sample_range'] = config['clip_sample_range']
    
    # Create scheduler based on type
    if scheduler_type == 'ddpm':
        scheduler = DDPMScheduler(**common_args)
        
    elif scheduler_type == 'ddim':
        scheduler = DDIMScheduler(
            **common_args,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
    elif scheduler_type == 'dpm' or scheduler_type == 'dpmsolver':
        scheduler = DPMSolverMultistepScheduler(
            **common_args,
            algorithm_type=config.get('dpm_solver_algorithm', 'sde-dpmsolver++'),
            solver_order=config.get('dpm_solver_order', 2),
            use_karras_sigmas=config.get('use_karras_sigmas', False),
        )
        
    elif scheduler_type == 'euler':
        scheduler = EulerDiscreteScheduler(
            **common_args,
            use_karras_sigmas=config.get('use_karras_sigmas', False),
        )
        
    elif scheduler_type == 'euler_ancestral':
        scheduler = EulerAncestralDiscreteScheduler(**common_args)
        
    elif scheduler_type == 'heun':
        scheduler = HeunDiscreteScheduler(
            **common_args,
            use_karras_sigmas=config.get('use_karras_sigmas', False),
        )
        
    elif scheduler_type == 'kdpm2':
        scheduler = KDPM2DiscreteScheduler(
            **common_args,
            use_karras_sigmas=config.get('use_karras_sigmas', False),
        )
        
    elif scheduler_type == 'pndm':
        scheduler = PNDMScheduler(
            **common_args,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class NoiseScheduler:
    """Wrapper for diffusion noise schedulers with additional utilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scheduler = get_scheduler(config)
        self.num_train_timesteps = config['t_steps']
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = 'cpu'):
        """Set the number of inference timesteps."""
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor
    ) -> torch.Tensor:
        """Add noise to samples."""
        return self.scheduler.add_noise(original_samples, noise, timesteps)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ):
        """Perform one denoising step."""
        return self.scheduler.step(
            model_output, timestep, sample, return_dict=return_dict, **kwargs
        )
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor
    ) -> torch.Tensor:
        """Get velocity for v-prediction parameterization."""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(sample.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def scale_model_input(self, sample: torch.Tensor, timestep: int) -> torch.Tensor:
        """Scale the denoising model input (for some schedulers)."""
        if hasattr(self.scheduler, 'scale_model_input'):
            return self.scheduler.scale_model_input(sample, timestep)
        return sample


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings with optional learned components."""
    
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 1.0,
        learned: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        
        if learned:
            self.linear = nn.Linear(dim, dim)
        else:
            self.linear = None
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = self.dim // 2
        exponent = -np.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        
        # Scale embeddings
        emb = 2 * np.pi * emb
        
        # Concatenate sin and cos
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Zero pad if necessary
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant', value=0)
        
        # Apply learned linear transformation if specified
        if self.linear is not None:
            emb = self.linear(emb)
        
        return emb


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values from a 1D tensor for given timesteps and reshape for broadcasting."""
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def variance_scaling_loss_weight(timesteps: torch.Tensor, scheduler: SchedulerMixin) -> torch.Tensor:
    """Compute variance-based loss weights for different timesteps."""
    # Get variance schedule
    alphas_cumprod = scheduler.alphas_cumprod
    
    # Compute SNR (signal-to-noise ratio)
    snr = alphas_cumprod / (1 - alphas_cumprod)
    
    # Different weighting schemes
    # Min-SNR weighting
    min_snr_gamma = 5.0
    weight = torch.minimum(snr, torch.ones_like(snr) * min_snr_gamma) / snr
    
    return extract_into_tensor(weight, timesteps, timesteps.shape)


def compute_snr(scheduler: SchedulerMixin, timesteps: torch.Tensor) -> torch.Tensor:
    """Compute signal-to-noise ratio for given timesteps."""
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    
    alpha_t = extract_into_tensor(sqrt_alphas_cumprod, timesteps, timesteps.shape)
    sigma_t = extract_into_tensor(sqrt_one_minus_alphas_cumprod, timesteps, timesteps.shape)
    
    snr = (alpha_t / sigma_t) ** 2
    return snr