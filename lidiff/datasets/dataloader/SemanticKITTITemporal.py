import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import point_set_to_sparse
from natsort import natsorted
import os
import numpy as np
import yaml

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, dataset_norm=False, std_axis_norm=False):
        super().__init__()
        self.data_dir = data_dir

        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range

        self.split = split
        self.seqs = seqs
        self.cache_maps = {}

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        if os.path.isfile(f'utils/data_stats_range_{int(self.max_range)}m.yml') and dataset_norm:
            stats = yaml.safe_load(open(f'utils/data_stats_range_{int(self.max_range)}m.yml'))
            data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
            if std_axis_norm:
                data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
            else:
                data_std = np.array([stats['std'], stats['std'], stats['std']])
            self.data_stats = {
                'mean': torch.tensor(data_mean),
                'std': torch.tensor(data_std)
            }

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []
        self.seq_poses = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            poses = load_poses(os.path.join(point_seq_path, 'calib.txt'), os.path.join(point_seq_path, 'poses.txt'))
            p_full = np.load(f'{point_seq_path}/map_clean.npy') if self.split != 'test' else np.array([[1,0,0],[0,1,0],[0,0,1]])
            self.cache_maps[seq] = p_full
 
            for file_num in range(0, len(point_seq_bin)):
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))
                self.seq_poses.append(poses[file_num])

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        seq_num = self.points_datapath[index].split('/')[-3]
        fname = self.points_datapath[index].split('/')[-1].split('.')[0]

        # Load raw points and intensities
        raw = np.fromfile(self.points_datapath[index], dtype=np.float32).reshape(-1,4)
        p_part = raw[:, :3].copy()
        i_part = raw[:, 3].copy()
        
        # Filter out moving points by label
        if self.split != 'test':
            label_file = self.points_datapath[index].replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32).reshape(-1)
            l_set = l_set & 0xFFFF
            static_idx = (l_set < 252) & (l_set > 1)
            p_part = p_part[static_idx]
            i_part = i_part[static_idx]
        # Range and height filtering
        dist_part = np.linalg.norm(p_part, axis=1)
        mask = (dist_part < self.max_range) & (dist_part > 3.5) & (p_part[:,2] > -4.)
        p_part = p_part[mask]
        i_part = i_part[mask]
        pose = self.seq_poses[index]

        p_map = self.cache_maps[seq_num]

        # Generate full point cloud (map) and corresponding intensities
        if self.split != 'test':
            trans = pose[:-1,-1]
            dist_full = np.linalg.norm(p_map - trans, axis=1)
            p_full = p_map[dist_full < self.max_range]
            # Transform to current frame
            homog = np.concatenate((p_full, np.ones((len(p_full),1))), axis=1)
            p_full = (homog @ np.linalg.inv(pose).T)[:,:3]
            p_full = p_full[p_full[:,2] > -4.]
            # No intensity available for map points: set to zero
            i_full = np.zeros((p_full.shape[0],), dtype=np.float32)
        else:
            p_full = p_part.copy()
            i_full = i_part.copy()

        # Data augmentation on training split (apply to coords only, preserve intensity)
        if self.split == 'train':
            coords_concat = np.concatenate((p_full, p_part), axis=0)
            intens_concat = np.concatenate((i_full, i_part), axis=0)
            # stack coords and intensity
            p_concat = np.concatenate((coords_concat, intens_concat[:,None]), axis=1)
            p_concat = self.transforms(p_concat)
            # split back
            num_full = p_full.shape[0]
            p_full = p_concat[:num_full, :3]
            i_full = p_concat[:num_full, 3]
            p_part = p_concat[num_full:, :3]
            i_part = p_concat[num_full:, 3]

        # patial pcd has 1/10 of the complete pcd size
        n_part = int(self.num_points / 10.)

        # Prepare samples to fixed size and include intensities
        return point_set_to_sparse(
            np.concatenate((p_full, i_full[:,None]), axis=1),
            np.concatenate((p_part, i_part[:,None]), axis=1),
            self.num_points,
            n_part,
            self.resolution,
            self.points_datapath[index],
            p_mean=self.data_stats['mean'],
            p_std=self.data_stats['std'],
        )

    def __len__(self):
        return self.nr_data

##################################################################################################
