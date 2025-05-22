import click
from pathlib import Path
import yaml
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    EarlyStopping,
    RichProgressBar,
    RichModelSummary
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import MinkowskiEngine as ME

import lidiff.datasets.datasets as datasets
from lidiff.models.models_improved import ImprovedDiffusionPoints


def set_deterministic(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


@click.command()
@click.option('--config', '-c', type=str, 
              help='Path to config file (.yaml)',
              default='lidiff/config/config.yaml')
@click.option('--weights', '-w', type=str,
              help='Path to pretrained weights (.ckpt)',
              default=None)
@click.option('--checkpoint', '-ckpt', type=str,
              help='Path to checkpoint file (.ckpt) to resume training',
              default=None)
@click.option('--test', '-t', is_flag=True, 
              help='Run in test mode')
@click.option('--seed', '-s', type=int, default=42,
              help='Random seed for reproducibility')
@click.option('--wandb', is_flag=True,
              help='Enable Weights & Biases logging')
@click.option('--name', '-n', type=str, default=None,
              help='Experiment name (defaults to config id)')
def main(config, weights, checkpoint, test, seed, wandb, name):
    """Improved training script with modern PyTorch Lightning features."""
    
    # Set deterministic behavior
    set_deterministic(seed)
    
    # Load configuration
    cfg = yaml.safe_load(open(config))
    
    # Update config with environment variables if available
    if 'TRAIN_DATABASE' in os.environ:
        cfg['data']['data_dir'] = os.environ['TRAIN_DATABASE']
    
    # Set experiment name
    exp_name = name or cfg['experiment']['id']
    
    # Initialize model
    if weights and test:
        # Load checkpoint config for testing
        ckpt_dir = Path(weights).parent
        hparams_path = ckpt_dir / 'hparams.yaml'
        if hparams_path.exists():
            ckpt_cfg = yaml.safe_load(open(hparams_path))
            # Override test-specific parameters
            ckpt_cfg['train']['batch_size'] = cfg['train'].get('batch_size', 1)
            ckpt_cfg['train']['num_workers'] = cfg['train'].get('num_workers', 4)
            ckpt_cfg['diff']['s_steps'] = cfg['diff'].get('s_steps', 50)
            cfg = ckpt_cfg
    
    # Create model
    if weights:
        model = ImprovedDiffusionPoints.load_from_checkpoint(weights, config=cfg)
    else:
        model = ImprovedDiffusionPoints(cfg)
    
    # Initialize data module
    data_module = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    
    # Setup callbacks
    callbacks = []
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Model checkpointing with improved settings
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{exp_name}',
        filename='{epoch:02d}-{val_fscore:.3f}',
        monitor='val/fscore',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if not test:
        callbacks.append(
            EarlyStopping(
                monitor='val/fscore',
                mode='max',
                patience=10,
                verbose=True,
            )
        )
    
    # Rich progress bar for better visualization
    callbacks.append(RichProgressBar())
    callbacks.append(RichModelSummary(max_depth=2))
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir='experiments',
        name=exp_name,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger (optional)
    if wandb:
        wandb_logger = WandbLogger(
            project='lidiff',
            name=exp_name,
            save_dir='experiments',
            log_model=True,
        )
        loggers.append(wandb_logger)
    
    # Configure trainer with modern features
    trainer_kwargs = {
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 'auto',
        'logger': loggers,
        'callbacks': callbacks,
        'max_epochs': cfg['train']['max_epoch'],
        'check_val_every_n_epoch': 5,
        'log_every_n_steps': 50,
        'gradient_clip_val': 1.0,
        'gradient_clip_algorithm': 'norm',
        'deterministic': True,
        'precision': '16-mixed',  # Automatic mixed precision
        'accumulate_grad_batches': cfg['train'].get('accumulate_grad_batches', 1),
    }
    
    # Multi-GPU training
    if torch.cuda.device_count() > 1:
        # Convert model to use SyncBatchNorm
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        
        # Use DDP strategy with optimizations
        trainer_kwargs['strategy'] = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    
    # Add profiler for performance analysis
    if cfg['train'].get('profile', False):
        trainer_kwargs['profiler'] = 'advanced'
    
    # Create trainer
    trainer = Trainer(**trainer_kwargs)
    
    # Run training or testing
    if test:
        print("Running in TEST mode...")
        trainer.test(model, data_module, ckpt_path=checkpoint)
    else:
        print(f"Starting training experiment: {exp_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optionally validate before training
        if cfg['train'].get('validate_before_training', False):
            trainer.validate(model, data_module)
        
        # Train the model
        trainer.fit(
            model, 
            data_module,
            ckpt_path=checkpoint  # Resume from checkpoint if provided
        )
        
        # Run final test
        if cfg['train'].get('test_after_training', True):
            trainer.test(model, data_module)


if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
    main()