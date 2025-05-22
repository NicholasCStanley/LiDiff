import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
from pathlib import Path
from natsort import natsorted
import open3d as o3d
from typing import Dict, List, Tuple, Optional
import warnings

from lidiff.utils.pcd_preprocess import load_poses
from lidiff.utils.pcd_transforms import (
    rotate_point_cloud,
    rotate_perturbation_point_cloud,
    random_scale_point_cloud,
    random_flip_point_cloud,
    jitter_point_cloud
)
from lidiff.utils.data_map import learning_map

warnings.filterwarnings('ignore')


class ImprovedSemanticKITTIDataset(Dataset):
    """Improved SemanticKITTI dataset with better memory management and features."""
    
    def __init__(
        self,
        data_dir: str,
        sequences: List[str],
        split: str = 'train',
        resolution: float = 0.05,
        num_points: int = 180000,
        max_range: float = 50.0,
        dataset_norm: bool = False,
        std_axis_norm: bool = False,
        augmentation_config: Optional[Dict] = None,
        cache_maps: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.sequences = sequences
        self.split = split
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range
        self.num_points_partial = num_points // 10  # 1/10 of full cloud
        
        # Augmentation settings
        self.augmentation_config = augmentation_config or {}
        self.enable_augmentation = split == 'train' and any(self.augmentation_config.values())
        
        # Normalization settings
        self.dataset_norm = dataset_norm
        self.std_axis_norm = std_axis_norm
        
        # Load dataset statistics if normalization is enabled
        self.data_stats = self._load_data_stats()
        
        # Cache settings
        self.cache_maps = cache_maps
        self.sequence_maps = {}
        
        # Build file list
        self.scan_files = []
        self.poses = []
        self._build_file_list()
        
        print(f'Initialized {split} dataset with {len(self.scan_files)} scans')
        
    def _load_data_stats(self) -> Dict[str, Optional[torch.Tensor]]:
        """Load precomputed dataset statistics for normalization."""
        stats = {'mean': None, 'std': None}
        
        if not self.dataset_norm:
            return stats
            
        stats_file = f'utils/data_stats_range_{int(self.max_range)}m.yml'
        if not os.path.isfile(stats_file):
            print(f"Warning: Stats file {stats_file} not found. Normalization disabled.")
            return stats
            
        with open(stats_file, 'r') as f:
            data = yaml.safe_load(f)
            
        mean = np.array([data['mean_axis']['x'], data['mean_axis']['y'], data['mean_axis']['z']])
        
        if self.std_axis_norm:
            std = np.array([data['std_axis']['x'], data['std_axis']['y'], data['std_axis']['z']])
        else:
            std = np.array([data['std']] * 3)
            
        stats['mean'] = torch.tensor(mean, dtype=torch.float32)
        stats['std'] = torch.tensor(std, dtype=torch.float32)
        
        return stats
    
    def _build_file_list(self):
        """Build list of scan files and corresponding poses."""
        for seq in self.sequences:
            seq_path = self.data_dir / 'dataset' / 'sequences' / seq
            velodyne_path = seq_path / 'velodyne'
            
            if not velodyne_path.exists():
                print(f"Warning: Sequence {seq} not found at {seq_path}")
                continue
                
            # Load poses
            calib_file = seq_path / 'calib.txt'
            poses_file = seq_path / 'poses.txt'
            
            if poses_file.exists():
                poses = load_poses(str(calib_file), str(poses_file))
            else:
                # Identity poses for test sequences
                num_scans = len(list(velodyne_path.glob('*.bin')))
                poses = [np.eye(4) for _ in range(num_scans)]
            
            # Load or generate map
            if self.split != 'test' and self.cache_maps:
                map_file = seq_path / 'map_clean.npy'
                if map_file.exists():
                    self.sequence_maps[seq] = np.load(str(map_file))
                else:
                    print(f"Warning: Map file not found for sequence {seq}")
                    self.sequence_maps[seq] = None
            
            # Add scan files
            scan_files = natsorted(list(velodyne_path.glob('*.bin')))
            for i, scan_file in enumerate(scan_files):
                if i < len(poses):
                    self.scan_files.append(scan_file)
                    self.poses.append(poses[i])
    
    def _load_scan(self, scan_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load scan from binary file."""
        scan = np.fromfile(str(scan_path), dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]
        intensity = scan[:, 3]
        return points, intensity
    
    def _filter_static_points(
        self, 
        points: np.ndarray, 
        intensity: np.ndarray,
        label_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out dynamic objects using semantic labels."""
        if label_path is None or not label_path.exists():
            return points, intensity
            
        labels = np.fromfile(str(label_path), dtype=np.uint32).reshape(-1)
        labels = labels & 0xFFFF  # Extract semantic label
        
        # Keep only static objects (exclude moving objects like cars, people, etc.)
        static_mask = (labels < 252) & (labels > 1)
        
        return points[static_mask], intensity[static_mask]
    
    def _apply_range_filter(
        self, 
        points: np.ndarray, 
        intensity: np.ndarray,
        min_range: float = 3.5,
        min_height: float = -4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply range and height filtering."""
        distances = np.linalg.norm(points, axis=1)
        mask = (distances < self.max_range) & (distances > min_range) & (points[:, 2] > min_height)
        return points[mask], intensity[mask]
    
    def _generate_full_cloud(
        self,
        sequence: str,
        pose: np.ndarray,
        partial_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate full point cloud from map."""
        if self.split == 'test' or sequence not in self.sequence_maps:
            # For test split, use partial cloud as full cloud
            return partial_points.copy(), np.zeros(len(partial_points), dtype=np.float32)
            
        map_points = self.sequence_maps[sequence]
        if map_points is None:
            return partial_points.copy(), np.zeros(len(partial_points), dtype=np.float32)
        
        # Transform map to current frame
        translation = pose[:-1, -1]
        distances = np.linalg.norm(map_points - translation, axis=1)
        nearby_points = map_points[distances < self.max_range]
        
        # Convert to homogeneous coordinates
        homog = np.hstack([nearby_points, np.ones((len(nearby_points), 1))])
        
        # Transform to current frame
        full_points = (homog @ np.linalg.inv(pose).T)[:, :3]
        
        # Filter by height
        full_points = full_points[full_points[:, 2] > -4.0]
        
        # No intensity for map points
        full_intensity = np.zeros(len(full_points), dtype=np.float32)
        
        return full_points, full_intensity
    
    def _augment_points(
        self,
        points: np.ndarray,
        intensity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        if not self.enable_augmentation:
            return points, intensity
            
        # Stack for joint transformation
        combined = np.hstack([points, intensity[:, np.newaxis]])
        combined = combined[np.newaxis, ...]  # Add batch dimension
        
        # Apply augmentations
        if self.augmentation_config.get('rotation', True):
            combined[:, :, :3] = rotate_point_cloud(combined[:, :, :3])
            combined[:, :, :3] = rotate_perturbation_point_cloud(combined[:, :, :3])
            
        if self.augmentation_config.get('scaling', True):
            combined[:, :, :3] = random_scale_point_cloud(combined[:, :, :3])
            
        if self.augmentation_config.get('flip', True):
            combined[:, :, :3] = random_flip_point_cloud(combined[:, :, :3])
            
        if self.augmentation_config.get('jitter', 0) > 0:
            noise = self.augmentation_config['jitter']
            combined[:, :, :3] = jitter_point_cloud(combined[:, :, :3], sigma=noise)
        
        # Unpack
        combined = combined[0]  # Remove batch dimension
        return combined[:, :3], combined[:, 3]
    
    def _sample_points(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
        num_samples: int,
        use_fps: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points to fixed size."""
        current_size = len(points)
        
        if current_size >= num_samples:
            # Downsample
            if use_fps and num_samples < current_size // 2:
                # Use FPS for significant downsampling
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd_down = pcd.farthest_point_down_sample(num_samples)
                
                # Get corresponding intensities
                tree = o3d.geometry.KDTreeFlann(pcd)
                sampled_intensity = []
                for p in np.asarray(pcd_down.points):
                    [_, idx, _] = tree.search_knn_vector_3d(p, 1)
                    sampled_intensity.append(intensity[idx[0]])
                    
                return np.asarray(pcd_down.points), np.array(sampled_intensity)
            else:
                # Random sampling
                indices = np.random.choice(current_size, num_samples, replace=False)
                return points[indices], intensity[indices]
        else:
            # Upsample by repetition
            repeat_times = int(np.ceil(num_samples / current_size))
            indices = np.tile(np.arange(current_size), repeat_times)[:num_samples]
            return points[indices], intensity[indices]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Get dataset item."""
        # Load scan data
        scan_path = self.scan_files[index]
        sequence = scan_path.parent.parent.name
        pose = self.poses[index]
        
        # Load partial cloud
        partial_points, partial_intensity = self._load_scan(scan_path)
        
        # Filter static points for training
        if self.split != 'test':
            label_path = scan_path.parent.parent / 'labels' / scan_path.name.replace('.bin', '.label')
            partial_points, partial_intensity = self._filter_static_points(
                partial_points, partial_intensity, label_path
            )
        
        # Apply range filtering
        partial_points, partial_intensity = self._apply_range_filter(
            partial_points, partial_intensity
        )
        
        # Generate full cloud
        full_points, full_intensity = self._generate_full_cloud(
            sequence, pose, partial_points
        )
        
        # Apply augmentation (to both clouds together for consistency)
        if self.enable_augmentation:
            # Concatenate for joint transformation
            all_points = np.vstack([full_points, partial_points])
            all_intensity = np.hstack([full_intensity, partial_intensity])
            
            # Augment
            all_points, all_intensity = self._augment_points(all_points, all_intensity)
            
            # Split back
            n_full = len(full_points)
            full_points = all_points[:n_full]
            full_intensity = all_intensity[:n_full]
            partial_points = all_points[n_full:]
            partial_intensity = all_intensity[n_full:]
        
        # Sample to fixed sizes
        full_points, full_intensity = self._sample_points(
            full_points, full_intensity, self.num_points, use_fps=False
        )
        partial_points, partial_intensity = self._sample_points(
            partial_points, partial_intensity, self.num_points_partial, use_fps=True
        )
        
        # Convert to tensors
        full_points = torch.from_numpy(full_points).float()
        full_intensity = torch.from_numpy(full_intensity).float()
        partial_points = torch.from_numpy(partial_points).float()
        partial_intensity = torch.from_numpy(partial_intensity).float()
        
        # Compute or use precomputed statistics
        if self.data_stats['mean'] is not None:
            mean = self.data_stats['mean']
            std = self.data_stats['std']
        else:
            mean = full_points.mean(dim=0)
            std = full_points.std(dim=0)
        
        return (
            full_points,
            full_intensity,
            mean,
            std,
            partial_points,
            partial_intensity,
            str(scan_path)
        )
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.scan_files)