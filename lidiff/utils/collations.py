import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

def feats_to_coord(p_feats, resolution, mean, std):
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def normalize_pcd(points, mean, std):
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse_refine(p_full, p_part, n_full, n_part, resolution, filename):
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    #if mode == 'diffusion':
    #p_full = p_full[torch.randperm(p_full.shape[0])]
    #p_part = p_part[torch.randperm(p_part.shape[0])]
    #elif mode == 'refine':
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])   

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])

    #p_feats = ME.utils.batched_coordinates([p_feats], dtype=torch.float32)[:2000]
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse(p_full_with_intensity, p_part_with_intensity, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    # Split coordinates and intensity
    p_full = p_full_with_intensity[:, :3]
    i_full = p_full_with_intensity[:, 3]
    p_part = p_part_with_intensity[:, :3]
    i_part = p_part_with_intensity[:, 3]
    
    # Process partial point cloud
    concat_part = int(np.ceil(n_part / p_part.shape[0]))
    if concat_part > 1:
        idx = np.tile(np.arange(p_part.shape[0]), concat_part)[:n_part]
        p_part = p_part[idx]
        i_part = i_part[idx]
    
    # Farthest point sampling for partial cloud
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    sampled_indices = np.array(pcd_part.points)
    # Find closest points to get corresponding intensities
    tree = o3d.geometry.KDTreeFlann(pcd_part)
    i_part_sampled = []
    for i in range(len(sampled_indices)):
        [_, idx, _] = tree.search_knn_vector_3d(sampled_indices[i], 1)
        i_part_sampled.append(i_part[idx[0]])
    p_part = torch.tensor(sampled_indices, dtype=torch.float32)
    i_part = torch.tensor(np.array(i_part_sampled), dtype=torch.float32)
    
    # Filter full cloud by viewpoint
    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint]
    i_full = i_full[in_viewpoint]
    
    # Ensure we have enough points
    if p_full.shape[0] < n_full:
        concat_full = int(np.ceil(n_full / p_full.shape[0]))
        idx = np.tile(np.arange(p_full.shape[0]), concat_full)[:n_full]
        p_full = p_full[idx]
        i_full = i_full[idx]
    else:
        # Random sampling
        idx = torch.randperm(p_full.shape[0])[:n_full]
        p_full = p_full[idx]
        i_full = i_full[idx]
    
    p_full = torch.tensor(p_full, dtype=torch.float32)
    i_full = torch.tensor(i_full, dtype=torch.float32)
    
    # Compute or use provided statistics (on coordinates only)
    p_mean = p_full.mean(axis=0) if p_mean is None else torch.tensor(p_mean, dtype=torch.float32)
    p_std = p_full.std(axis=0) if p_std is None else torch.tensor(p_std, dtype=torch.float32)
    
    return [p_full, i_full, p_mean, p_std, p_part, i_part, filename]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseSegmentCollation:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # Transpose the batch: list of lists to list of tuples
        batch = list(zip(*data))
        pcd_full = torch.stack(batch[0]).float()
        intensity_full = torch.stack(batch[1]).float()
        mean = torch.stack(batch[2]).float()
        std = torch.stack(batch[3]).float()
        pcd_part = torch.stack(batch[4]).float()
        intensity_part = torch.stack(batch[5]).float()
        filenames = batch[6]
        return {
            'pcd_full': pcd_full,
            'intensity_full': intensity_full,
            'mean': mean,
            'std': std,
            'pcd_part': pcd_part,
            'intensity_part': intensity_part,
            'filename': filenames,
        }
