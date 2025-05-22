import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from lidiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs, path

try:
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning import LightningDataModule
except ImportError:
    import lightning.pytorch as pl
    LightningModule = pl.LightningModule
    LightningDataModule = pl.LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        # configure separate input channels for full (xyz) and partial (xyz+intensity)
        in_ch_full = self.hparams['model'].get('in_channels_full', 3)
        in_ch_part = self.hparams['model'].get('in_channels_part', in_ch_full)
        out_dim = self.hparams['model']['out_dim']
        self.partial_enc = minknet.MinkGlobalEnc(in_channels=in_ch_part, out_channels=out_dim)
        self.model = minknet.MinkUNetDiff(in_channels=in_ch_full, out_channels=out_dim)

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']

    def scheduler_to_cuda(self):
        # Move scheduler tensors to CUDA if available
        if hasattr(self.dpm_scheduler, 'timesteps'):
            self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        if hasattr(self.dpm_scheduler, 'betas'):
            self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        if hasattr(self.dpm_scheduler, 'alphas'):
            self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        if hasattr(self.dpm_scheduler, 'alphas_cumprod'):
            self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        if hasattr(self.dpm_scheduler, 'alpha_t'):
            self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        if hasattr(self.dpm_scheduler, 'sigma_t'):
            self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        if hasattr(self.dpm_scheduler, 'lambda_t'):
            self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        if hasattr(self.dpm_scheduler, 'sigmas'):
            self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].cuda() * noise

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < self.hparams['data']['max_range']

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, gt_pts, x_mean, x_std):
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # Reset partial point clouds to avoid memory accumulation in MinkowskiEngine
            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            # Clear cache periodically to prevent memory issues
            if t % 10 == 0:
                torch.cuda.empty_cache()

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.model(x_full, x_full_sparse, part_feat, t)
        # Avoid frequent cache clearing - let PyTorch handle memory
        # torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def points_to_tensor(self, coords, mean, std, intensity=None):
        """Build a MinkowskiEngine TensorField from point coordinates and optional intensity."""
        # coords: tensor of shape (B, N, 3); intensity: tensor of shape (B, N, 1) or None
        # Quantize spatial coordinates
        coords_list = [c for c in coords]
        quant_coords = ME.utils.batched_coordinates(coords_list, dtype=torch.float32, device=self.device)
        quant = quant_coords.clone()
        quant[:,1:] = feats_to_coord(quant_coords[:,1:], self.hparams['data']['resolution'], mean, std)
        # Prepare features
        coords_flat = torch.cat(coords_list, dim=0)
        if intensity is not None:
            intens_list = [i for i in intensity]
            intens_flat = torch.cat(intens_list, dim=0)
            feats = torch.cat([coords_flat, intens_flat.unsqueeze(-1)], dim=1)
        else:
            feats = coords_flat
        x_t = ME.TensorField(
            features=feats,
            coordinates=quant,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        # Avoid frequent cache clearing - let PyTorch handle memory
        # torch.cuda.empty_cache()
        return x_t

    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        # torch.cuda.empty_cache()  # Let PyTorch handle memory automatically
        noise = torch.randn(batch['pcd_full'].shape, device=self.device)
        
        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()
        # sample q at step t
        # we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)

        # replace the original points with the noise sampled
        # Tensorize noisy full cloud (xyz only)
        x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

        # for classifier-free guidance switch between conditional and unconditional training
        # Partial conditioning with optional intensity feature
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1:
            x_part = self.points_to_tensor(
                batch['pcd_part'], batch['mean'], batch['std'], batch['intensity_part']
            )
        else:
            zeros_int = torch.zeros_like(batch['intensity_part'])
            x_part = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), batch['mean'], batch['std'], zeros_int
            )

        denoise_t = self.forward(x_full, x_full.sparse(), x_part, t)
        loss_mse = self.p_losses(denoise_t, noise)
        loss_mean = (denoise_t.mean())**2
        loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)

        std_noise = (denoise_t - noise)**2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/var', std_noise.var())
        self.log('train/std', std_noise.std())
        # torch.cuda.empty_cache()  # Let PyTorch handle memory automatically

        return loss

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            # Initialize with partial scan and preserve intensity
            x_init = batch['pcd_part'].repeat(1,10,1)
            i_init = batch['intensity_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'], i_init)
            x_part = self.points_to_tensor(
                batch['pcd_part'], batch['mean'], batch['std'], batch['intensity_part']
            )
            zeros_int = torch.zeros_like(batch['intensity_part'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), batch['mean'], batch['std'], zeros_int
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        # torch.cuda.empty_cache()  # Let PyTorch handle memory automatically

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # Initialize from partial scan with intensity
            x_init = batch['pcd_part'].repeat(1,10,1)
            i_init = batch['intensity_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'], i_init)
            x_part = self.points_to_tensor(
                batch['pcd_part'], batch['mean'], batch['std'], batch['intensity_part']
            )
            zeros_int = torch.zeros_like(batch['intensity_part'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), batch['mean'], batch['std'], zeros_int
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        # torch.cuda.empty_cache()  # Let PyTorch handle memory automatically

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]

#######################################
# Modules
#######################################
