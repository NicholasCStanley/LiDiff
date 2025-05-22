import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from typing import Optional, Dict
from pathlib import Path

from lidiff.datasets.dataloader.SemanticKITTIImproved import ImprovedSemanticKITTIDataset
from lidiff.utils.collations import SparseSegmentCollation


class ImprovedKITTIDataModule(pl.LightningDataModule):
    """Improved Lightning DataModule for SemanticKITTI dataset."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.data_config = config['data']
        self.train_config = config['train']
        
        # Dataset parameters
        self.data_dir = Path(self.data_config['data_dir'])
        self.resolution = self.data_config['resolution']
        self.num_points = self.data_config['num_points']
        self.max_range = self.data_config['max_range']
        
        # Training parameters
        self.batch_size = self.train_config['batch_size']
        self.num_workers = self.train_config.get('num_workers', 4)
        
        # Augmentation config
        self.augmentation_config = self.data_config.get('augmentation', {})
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Collation function
        self.collate_fn = SparseSegmentCollation(mode='diffusion')
        
    def prepare_data(self):
        """Download or prepare data. Called only on 1 GPU."""
        # Check if data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist!")
            
        # Check for required sequences
        sequences_path = self.data_dir / 'dataset' / 'sequences'
        if not sequences_path.exists():
            raise ValueError(f"Sequences directory {sequences_path} does not exist!")
            
        # Verify at least one sequence exists
        required_sequences = (
            self.data_config.get('train', []) + 
            self.data_config.get('validation', []) + 
            self.data_config.get('test', [])
        )
        
        for seq in required_sequences[:1]:  # Check at least one
            seq_path = sequences_path / seq
            if not seq_path.exists():
                raise ValueError(f"Sequence {seq} not found at {seq_path}")
                
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = ImprovedSemanticKITTIDataset(
                data_dir=str(self.data_dir),
                sequences=self.data_config['train'],
                split='train',
                resolution=self.resolution,
                num_points=self.num_points,
                max_range=self.max_range,
                dataset_norm=self.data_config.get('dataset_norm', False),
                std_axis_norm=self.data_config.get('std_axis_norm', False),
                augmentation_config=self.augmentation_config,
                cache_maps=True
            )
            
            # Validation dataset
            self.val_dataset = ImprovedSemanticKITTIDataset(
                data_dir=str(self.data_dir),
                sequences=self.data_config['validation'],
                split='val',
                resolution=self.resolution,
                num_points=self.num_points,
                max_range=self.max_range,
                dataset_norm=self.data_config.get('dataset_norm', False),
                std_axis_norm=self.data_config.get('std_axis_norm', False),
                augmentation_config=None,  # No augmentation for validation
                cache_maps=True
            )
            
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = ImprovedSemanticKITTIDataset(
                data_dir=str(self.data_dir),
                sequences=self.data_config['test'],
                split='test',
                resolution=self.resolution,
                num_points=self.num_points,
                max_range=self.max_range,
                dataset_norm=self.data_config.get('dataset_norm', False),
                std_axis_norm=self.data_config.get('std_axis_norm', False),
                augmentation_config=None,  # No augmentation for test
                cache_maps=False  # No maps for test sequences
            )
    
    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # Use batch size 1 for validation
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Use batch size 1 for testing
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        # Clear cached maps to free memory
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            self.train_dataset.sequence_maps.clear()
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            self.val_dataset.sequence_maps.clear()


# Updated dataloader registry
dataloaders = {
    'KITTI': ImprovedKITTIDataModule,
    'KITTIImproved': ImprovedKITTIDataModule,
}