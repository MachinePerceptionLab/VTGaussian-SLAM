import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import (keyframe_selection_overlap_visbased, keyframe_selection_overlap_visbased_earliest_dynamic_new_topkbase,
                                     keyframe_selection_overlap, find_earliest_keyframe)
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette, 
    transform_to_frame, l1_loss_v1, matrix_to_quaternion, l1_loss_v1_mask
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from visual_odometer import VisualOdometer

import inspect
import copy


device = torch.device('cuda:0')



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective", factor=1.005):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX + 0.5)/FX
    yy = (y_grid - CY + 0.5)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)
    depth_z = depth_z * factor

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2 
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
   
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld



def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
  
    avg_mean3_sq_dist = torch.mean(mean3_sq_dist)
    ############################################################################################################
    # use the average mean3_sq_dist for all the gaussians
    use_one_scale_for_each_frame = False
    if use_one_scale_for_each_frame == True:
        mean3_sq_dist = avg_mean3_sq_dist

    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)) 
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]

    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)



def initialize_params_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, random_select=False, num_points=None, mask_variation=None):
    """
    Initialize parameters for the first timestep
    """

    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    densify_on_color_mask = densify_dataset is not None
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    if densify_on_color_mask:
        init_pt_cld_ori, mean3_sq_dist_ori = get_pointcloud(color, depth, intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)
        

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    
    if densify_on_color_mask:
        mask_variation = cv2.resize(mask_variation, (color.shape[2], color.shape[1]), interpolation=cv2.INTER_NEAREST)
        mask_variation = mask_variation.astype(np.bool_)
        mask_variation = torch.tensor(mask_variation.reshape(-1)).cuda()
        valid_depth_mask = (depth > 0)
        valid_depth_mask = valid_depth_mask.reshape(-1)
        mask = valid_depth_mask & mask_variation
        mask_variation = mask_variation.cpu()
        valid_depth_mask = valid_depth_mask.cpu()
        torch.cuda.empty_cache()
        init_pt_cld_densify, mean3_sq_dist_densify = get_pointcloud(color, depth, densify_intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)
        # concatenate the two point clouds
        init_pt_cld = torch.cat((init_pt_cld_ori, init_pt_cld_densify), dim=0)
        mean3_sq_dist = torch.cat((mean3_sq_dist_ori, mean3_sq_dist_densify), dim=0)

        
    else:
        init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)
                                                               

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    # Initaialize a params set
    params_ls = []
    variables_ls = []
    params_ls.append(params)
    variables_ls.append(variables)

    color = color.cpu()
    depth = depth.cpu()
    mask = mask.cpu()
    init_pt_cld = init_pt_cld.cpu()
    mean3_sq_dist = mean3_sq_dist.cpu()
    if densify_on_color_mask:
        init_pt_cld_densify = init_pt_cld_densify.cpu()
        mean3_sq_dist_densify = mean3_sq_dist_densify.cpu()
        init_pt_cld_ori = init_pt_cld_ori.cpu()
        mean3_sq_dist_ori = mean3_sq_dist_ori.cpu()
    torch.cuda.empty_cache()

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam, params_ls, variables_ls
    else:
        return params, variables, intrinsics, w2c, cam, params_ls, variables_ls


def initialize_params_base_timestep(dataset, num_frames, time_idx, w2c, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, mask_variation=None):
    """
    Initialize parameters for baseframe timestep
    """                          
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, _ = dataset[time_idx]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    ############################################$$$$$$$$$$$$$$$$############################################
    densify_on_color_mask = True
    if densify_on_color_mask:
        mask = (depth > 0) # Mask out invalid depth values
        mask = mask.reshape(-1)
        init_pt_cld_ori, mean3_sq_dist_ori = get_pointcloud(color, depth, intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)
    else:
        mask = (depth > 0)
        mask = mask.reshape(-1)
        
    color = color.cpu()
    depth = depth.cpu()
    torch.cuda.empty_cache()

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[time_idx]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
    else:
        densify_intrinsics = intrinsics

    
    if densify_on_color_mask:
        mask_variation = cv2.resize(mask_variation, (color.shape[2], color.shape[1]), interpolation=cv2.INTER_NEAREST)
        mask_variation = mask_variation.astype(np.bool_)
        mask_variation = torch.tensor(mask_variation.reshape(-1)).cuda()
        valid_depth_mask = (depth > 0)
        valid_depth_mask = valid_depth_mask.reshape(-1)
        mask = valid_depth_mask & mask_variation
        mask_variation = mask_variation.cpu()
        valid_depth_mask = valid_depth_mask.cpu()
        torch.cuda.empty_cache()
        init_pt_cld_densify, mean3_sq_dist_densify = get_pointcloud(color, depth, densify_intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)
        # concatenate the two point clouds
        init_pt_cld = torch.cat((init_pt_cld_ori, init_pt_cld_densify), dim=0)
        mean3_sq_dist = torch.cat((mean3_sq_dist_ori, mean3_sq_dist_densify), dim=0)
        
    else:
        init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)                                                          

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))


    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    color = color.cpu()
    depth = depth.cpu()
    mask = mask.cpu()
    init_pt_cld = init_pt_cld.cpu()
    mean3_sq_dist = mean3_sq_dist.cpu()
    init_pt_cld_densify = init_pt_cld_densify.cpu()
    mean3_sq_dist_densify = mean3_sq_dist_densify.cpu()
    init_pt_cld_ori = init_pt_cld_ori.cpu()
    mean3_sq_dist_ori = mean3_sq_dist_ori.cpu()
    torch.cuda.empty_cache()


    return params, variables


def get_vis_mask(overlap_w2c, pts, intrinsics, overlap_gtdepth, vis_mask_thres, height, width):
    """
    Get the visible mask
    """
    est_w2c = overlap_w2c
    # Transform the 3D pointcloud to the keyframe's camera space
    pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    transformed_pts = (est_w2c @ pts4.T).T[:, :3]
    # Project the 3D pointcloud to the keyframe's image space
    points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_z = points_2d[:, 2:] + 1e-5
    points_2d = points_2d / points_z
    projected_pts = points_2d[:, :2]


    # Filter out the points that are invisible based on the depth
    curr_gt_depth = overlap_gtdepth.to(projected_pts.device).reshape(1, 1, height, width)
    vgrid = projected_pts.reshape(1, 1, -1, 2)
    # normalize to [-1, 1]
    vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
    vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
    depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
    depth_sample = depth_sample.reshape(-1)

    mask_visible = torch.abs(depth_sample - points_z[:, 0]) < vis_mask_thres * torch.min(depth_sample, points_z[:, 0])
    mask_visible = mask_visible.reshape(overlap_gtdepth.shape[1:])

    return mask_visible


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, 
             tracking_iteration=None, additional_mask=None, dataset_name=None, 
             presence_sil_mask_mse_ls=None, sil_thres_ls=None, far_depth_filter_thres=None, vis_mask_thres=0.05,
             curr_w2c=None, overlap_w2c=None, overlap_gtdepth=None, overlap_last_w2c=None, overlap_last_gtdepth=None, overlap_mid_w2c=None, overlap_mid_gtdepth=None):
    """
    Compute loss for mapping and tracking
    """
    # Initialize Loss Dictionary
    losses = {}

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.tensor(v).cuda().float().contiguous()
        else:
            params[k] = v.cuda().float().contiguous()
    
    for k, v in variables.items():
        if not isinstance(v, torch.Tensor):
            variables[k] = torch.tensor(v).cuda().float().contiguous()
        else:
            variables[k] = v.cuda().float().contiguous()

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_gaussians)

    
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification


    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]

   
    
    if dataset_name == 'replica':
        if tracking and use_sil_for_loss:
            if tracking_iteration == 0:
                time_begin = time.time()
                presence_sil_mask_0 = (silhouette > 0.990) & (curr_data['depth'] > 0)
                presence_sil_mask_3 = (silhouette > 0.993) & (curr_data['depth'] > 0)
                presence_sil_mask_5 = (silhouette > 0.995) & (curr_data['depth'] > 0)
                presence_sil_mask_7 = (silhouette > 0.997) & (curr_data['depth'] > 0)
                presence_sil_mask_9 = (silhouette > 0.999) & (curr_data['depth'] > 0)
                color_presence_sil_mask_0 = torch.tile(presence_sil_mask_0, (3, 1, 1))
                color_presence_sil_mask_3 = torch.tile(presence_sil_mask_3, (3, 1, 1))
                color_presence_sil_mask_5 = torch.tile(presence_sil_mask_5, (3, 1, 1))
                color_presence_sil_mask_7 = torch.tile(presence_sil_mask_7, (3, 1, 1))
                color_presence_sil_mask_9 = torch.tile(presence_sil_mask_9, (3, 1, 1))
                color_presence_sil_mask_0 = color_presence_sil_mask_0.detach()
                color_presence_sil_mask_3 = color_presence_sil_mask_3.detach()
                color_presence_sil_mask_5 = color_presence_sil_mask_5.detach()
                color_presence_sil_mask_7 = color_presence_sil_mask_7.detach()
                color_presence_sil_mask_9 = color_presence_sil_mask_9.detach()

                mse_0 = torch.mean((curr_data['im'] - im)[color_presence_sil_mask_0]**2)
                mse_3 = torch.mean((curr_data['im'] - im)[color_presence_sil_mask_3]**2)
                mse_5 = torch.mean((curr_data['im'] - im)[color_presence_sil_mask_5]**2)
                mse_7 = torch.mean((curr_data['im'] - im)[color_presence_sil_mask_7]**2)
                mse_9 = torch.mean((curr_data['im'] - im)[color_presence_sil_mask_9]**2)



                # compare the mse of different presence_sil_mask
                mse_ls = [mse_0.item(), mse_3.item(), mse_5.item(), mse_7.item(), mse_9.item()]
                silhouette_ls = [0.990, 0.993, 0.995, 0.997, 0.999]

                min_mse = min(mse_ls)
                min_mse_idx = mse_ls.index(min_mse)
                presence_sil_mask_mse_ls.append(min_mse)
                sil_thres_ls.append(silhouette_ls[min_mse_idx])
                presence_sil_mask = (silhouette > silhouette_ls[min_mse_idx])
            else:
                presence_sil_mask = (silhouette > sil_thres_ls[-1])


    elif dataset_name == 'tum' or dataset_name == 'scannet' or dataset_name == 'scannetpp':
        presence_sil_mask = (silhouette > sil_thres)
        

            

    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 50*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    if tracking and overlap_w2c is not None and dataset_name != 'replica':
        def get_pointcloud_forvismask(depth, intrinsics, w2c, sampled_indices):
            CX = intrinsics[0][2]
            CY = intrinsics[1][2]
            FX = intrinsics[0][0]
            FY = intrinsics[1][1]

            # Compute indices of sampled pixels
            xx = (sampled_indices[:, 1] - CX)/FX
            yy = (sampled_indices[:, 0] - CY)/FY
            depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

            # Initialize point cloud
            pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
            pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]

            return pts
        
        gt_depth = curr_data['depth']
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] >= 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        sampled_indices = valid_depth_indices


        # Back Project the selected pixels to 3D Pointcloud
        intrinsics = curr_data['intrinsics']
        pts = get_pointcloud_forvismask(gt_depth, intrinsics, curr_w2c, sampled_indices)
        # print('pts:', pts.shape)
        
        if dataset_name == 'replica':
            pass

        elif dataset_name == 'tum':
            mask_visible = get_vis_mask(overlap_w2c, pts, intrinsics, overlap_gtdepth, vis_mask_thres, height, width)

        elif dataset_name == 'scannet' or dataset_name == 'scannetpp':
            # overlap vis first
            mask_visible_first = get_vis_mask(overlap_w2c, pts, intrinsics, overlap_gtdepth, vis_mask_thres, height, width)
            # overlap mid 
            mask_visible_mid = get_vis_mask(overlap_mid_w2c, pts, intrinsics, overlap_mid_gtdepth, vis_mask_thres, height, width)
            # overlap last
            mask_visible_last = get_vis_mask(overlap_last_w2c, pts, intrinsics, overlap_last_gtdepth, vis_mask_thres, height, width)
            mask_visible = mask_visible_first | mask_visible_mid | mask_visible_last


        mask = mask & mask_visible

    if tracking and far_depth_filter_thres is not None and dataset_name != 'replica' and dataset_name != 'scannetpp':
        far_depth_filter_mask = curr_data['depth'] < far_depth_filter_thres
        mask = mask & far_depth_filter_mask
        

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        if additional_mask is None:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        elif additional_mask is not None:
            additional_mask = 10 * additional_mask.cuda().float() + 0.8 * torch.ones_like(additional_mask).cuda().float()
            losses['im'] = l1_loss_v1_mask(im, curr_data['im'], additional_mask) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

 

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        if dataset_name == 'replica':
            fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        else:
            fig, ax = plt.subplots(2, 7, figsize=(21, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(curr_data['im'].permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(curr_data['depth'][0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        if overlap_w2c is not None and dataset_name != 'replica':
            ax[0, 4].imshow(mask_visible.detach().cpu(), cmap="gray")
            ax[0, 4].set_title("Visible Mask")

            ax[0, 5].imshow(mask_visible_first.detach().cpu(), cmap="gray")
            ax[0, 5].set_title("Visible Mask First")
            ax[1, 5].imshow(mask_visible_mid.detach().cpu(), cmap="gray")
            ax[1, 5].set_title("Visible Mask Mid")
            ax[0, 6].imshow(mask_visible_last.detach().cpu(), cmap="gray")
            ax[0, 6].set_title("Visible Mask Last")


        if far_depth_filter_thres is not None and dataset_name != 'replica' and dataset_name != 'scannetpp':
            ax[1, 4].imshow(far_depth_filter_mask[0].detach().cpu(), cmap="gray")
            ax[1, 4].set_title("Far Depth Filter Mask")
        # Turn off axis
        if dataset_name == 'replica':
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis('off')
        else:
            for i in range(2):
                for j in range(7):
                    ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Frame{iter_time_idx:04d}_Tracking Iteration: {tracking_iteration}", fontsize=16) # , silthres: {sil_thres_ls[-1]}
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"frame{iter_time_idx:04d}_{tracking_iteration:03d}.png"), bbox_inches='tight')
        plt.close()
  

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    if presence_sil_mask_mse_ls is not None:
        return loss, variables, weighted_losses, presence_sil_mask_mse_ls, sil_thres_ls
    
    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    """
    Initialize params for newly added Gaussians
    """
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")


    avg_mean3_sq_dist = torch.mean(mean3_sq_dist)
    # use the average mean3_sq_dist for all the gaussians
    use_one_scale_for_each_frame = False #True
    if use_one_scale_for_each_frame == True:
        mean3_sq_dist = avg_mean3_sq_dist

    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params



def add_new_gaussians_base_frame(params, variables, ori_curr_data, densify_curr_data, sil_thres,
                      time_idx, mean_sq_dist_method, gaussian_distribution, config, mask_variation=None):
    """
    Add new Gaussians to the non-base frame
    """
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params[k] = v.cuda()
    for k, v in variables.items():
        if isinstance(v, torch.Tensor):
            variables[k] = v.cuda()
     # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, ori_curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=ori_curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = ori_curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask

    if torch.sum(non_presence_mask) > 0:

        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (ori_curr_data['depth'][0, :, :] > 0)
        ori_non_presence_mask = non_presence_mask & valid_depth_mask
        ori_non_presence_mask = ori_non_presence_mask.reshape(-1)
            
        new_pt_cld, mean3_sq_dist = get_pointcloud(ori_curr_data['im'], ori_curr_data['depth'], ori_curr_data['intrinsics'], 
                                    curr_w2c, mask=ori_non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        

        valid_depth_mask = (densify_curr_data['depth'][0, :, :] > 0)
        valid_depth_mask = valid_depth_mask.reshape(-1)


        mask_variation = cv2.resize(mask_variation, (densify_curr_data['im'].shape[2], densify_curr_data['im'].shape[1]), interpolation=cv2.INTER_NEAREST)
        mask_variation = mask_variation.astype(np.bool_)
        mask_variation = torch.tensor(mask_variation.reshape(-1)).cuda()

        non_presence_mask = non_presence_mask.cpu().numpy()
        non_presence_mask = non_presence_mask.astype(np.uint8)
        dense_non_presence_mask = cv2.resize(non_presence_mask, (densify_curr_data['im'].shape[2], densify_curr_data['im'].shape[1]), interpolation=cv2.INTER_NEAREST)
        dense_non_presence_mask = dense_non_presence_mask.astype(np.bool_)
        dense_non_presence_mask = torch.tensor(dense_non_presence_mask.reshape(-1)).cuda()
        dense_non_presence_mask = valid_depth_mask & mask_variation & dense_non_presence_mask
        valid_depth_mask = valid_depth_mask.cpu()
        mask_variation = mask_variation.cpu()
        torch.cuda.empty_cache()
        new_pt_cld_dense, mean3_sq_dist_dense = get_pointcloud(densify_curr_data['im'], densify_curr_data['depth'], densify_curr_data['intrinsics'], 
                                    curr_w2c, mask=dense_non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)

        new_pt_cld = torch.cat((new_pt_cld, new_pt_cld_dense), dim=0)
        mean3_sq_dist = torch.cat((mean3_sq_dist, mean3_sq_dist_dense), dim=0)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        add_number = new_params['means3D'].shape[0]

        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    else:
        add_number = 0

    return params, variables, add_number

    

def initialize_camera_pose(params, curr_time_idx, forward_prop, multiavg=False, odometer_rel=None):
    """
    Initialize the camera pose for the current frame based on the previous frame's pose.
    """
    with torch.no_grad():
        if curr_time_idx > 1 and odometer_rel is not None and forward_prop:
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            pre_w2c1 = torch.eye(4).cuda().float()
            pre_w2c1[:3, :3] = build_rotation(prev_rot1)
            pre_w2c1[:3, 3] = prev_tran1
            pre_c2w1 = torch.inverse(pre_w2c1)

            odometer_rel = odometer_rel.cuda()
            init_c2w = pre_c2w1 @ odometer_rel
            init_w2c = torch.inverse(init_c2w)
            init_rot = init_w2c[:3, :3].unsqueeze(0).detach()
            init_quat = matrix_to_quaternion(init_rot)
            init_tran = init_w2c[:3, 3]
            params['cam_unnorm_rots'][..., curr_time_idx] = init_quat.detach()
            params['cam_trans'][..., curr_time_idx] = init_tran.detach()
        elif curr_time_idx > 1 and odometer_rel is None and forward_prop:
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            pre_w2c1 = torch.eye(4).cuda().float()
            pre_w2c1[:3, :3] = build_rotation(prev_rot1)
            pre_w2c1[:3, 3] = prev_tran1
            pre_w2c2 = torch.eye(4).cuda().float()
            pre_w2c2[:3, :3] = build_rotation(prev_rot2)
            pre_w2c2[:3, 3] = prev_tran2
            pre_c2w1 = torch.inverse(pre_w2c1)
            pre_c2w2 = torch.inverse(pre_w2c2)

            if multiavg and curr_time_idx > 3:
                prev_rot3 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-3].detach())
                prev_tran3 = params['cam_trans'][..., curr_time_idx-3].detach()
                pre_w2c3 = torch.eye(4).cuda().float()
                pre_w2c3[:3, :3] = build_rotation(prev_rot3)
                pre_w2c3[:3, 3] = prev_tran3
                pre_c2w3 = torch.inverse(pre_w2c3)

                init_c2w = ((pre_c2w2 @ torch.inverse(pre_c2w3) + pre_c2w1 @ torch.inverse(pre_c2w2)) / 2) @ pre_c2w1
                init_w2c = torch.inverse(init_c2w)
                init_rot = init_w2c[:3, :3].unsqueeze(0).detach()
                init_quat = matrix_to_quaternion(init_rot)
                init_tran = init_w2c[:3, 3]
                params['cam_unnorm_rots'][..., curr_time_idx] = init_quat.detach()
                params['cam_trans'][..., curr_time_idx] = init_tran.detach()

            else:
                init_c2w = pre_c2w1 @ torch.inverse(pre_c2w2) @ pre_c2w1
                init_w2c = torch.inverse(init_c2w)
                init_rot = init_w2c[:3, :3].unsqueeze(0).detach()
                init_quat = matrix_to_quaternion(init_rot)
                init_tran = init_w2c[:3, 3]
                params['cam_unnorm_rots'][..., curr_time_idx] = init_quat.detach()
                params['cam_trans'][..., curr_time_idx] = init_tran.detach()

        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params




def quantize_selected_time_idx(selected_time_idx, num_frames_each_base_frame):
    """
    Quantize the selected timestep indices to the base frame indices.
    """
    quantized_selected_time_idx = []
    for idx in selected_time_idx:
        base_frame_idx = int(idx/num_frames_each_base_frame)
        quantized_selected_time_idx.append(base_frame_idx)
    # remove duplicates
    quantized_selected_time_idx = list(set(quantized_selected_time_idx))
    return quantized_selected_time_idx


def concat_keyframes_params_base_frame(params_ls, variables_ls, selected_time_idx, num_frames_each_base_frame):
    """
    Concatenate the parameters and variables from the quantized selected base frame indices.
    """
    params = {}
    num_gs_per_base_frame = []
    quantized_selected_time_idx = quantize_selected_time_idx(selected_time_idx, num_frames_each_base_frame)

      

    for idx in quantized_selected_time_idx:
        num_gs_per_base_frame.append(params_ls[idx]['means3D'].shape[0])
        for k, v in params_ls[idx].items():
            v = v.cuda().float().contiguous()
            if (k not in params) and (k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']):
                params[k] = v
            elif k in params:
                params[k] = torch.cat((params[k], v), dim=0)

    params['cam_unnorm_rots'] = params_ls[quantized_selected_time_idx[-1]]['cam_unnorm_rots']
    params['cam_trans'] = params_ls[quantized_selected_time_idx[-1]]['cam_trans']

    
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {}
    for idx in quantized_selected_time_idx:
        for k, v in variables_ls[idx].items():
            v = v.cuda().float().contiguous()
            if (k not in variables) and (k in ['max_2D_radius', 'means2D_gradient_accum', 'denom', 'timestep']):
                variables[k] = v
            elif k in variables:
                variables[k] = torch.cat((variables[k], v), dim=0)
            
    variables['scene_radius'] = variables_ls[quantized_selected_time_idx[-1]]['scene_radius']

    return params, variables, num_gs_per_base_frame


def concat_global(cat_params, cat_variables, cat_num_gs_per_frame=None, global_params=None, global_variables=None):
    """
    Concatenate the global parameters (fixed) the concatenated parameters from the base frames.
    """
    for k, v in global_params.items():
        if isinstance(v, torch.Tensor):
            global_params[k] = v.cuda()
    for k, v in global_variables.items():
        if isinstance(v, torch.Tensor):
            global_variables[k] = v.cuda()
    params = {}
    if cat_num_gs_per_frame is not None:
        num_gs_per_frame = [global_params['means3D'].shape[0]] + cat_num_gs_per_frame
    for k, v in cat_params.items():
        v = v.cuda().float().contiguous()
        if k in global_params and k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']:
            params[k] = torch.cat((global_params[k], v), dim=0)

    params['cam_unnorm_rots'] = cat_params['cam_unnorm_rots']
    params['cam_trans'] = cat_params['cam_trans']
    

    variables = {}
    for k, v in cat_variables.items():
        v = v.cuda().float().contiguous()
        if k in global_variables and k in ['max_2D_radius', 'means2D_gradient_accum', 'denom', 'timestep']:
            variables[k] = torch.cat((global_variables[k], v), dim=0)
            
    variables['scene_radius'] = cat_variables['scene_radius']

    if cat_num_gs_per_frame is not None:
        return params, variables, num_gs_per_frame
    else:
        return params, variables


def update_params_ls(params_ls, selected_time_idx, cat_params, num_gs_per_frame, num_frames_each_base_frame):
    """
    Update the parameters in the list of parameters.
    """
    
    cat_params_splitted = {}
    for k, v in cat_params.items():
        if k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']: 
            cat_params_splitted[k] = torch.split(v, num_gs_per_frame, dim=0)
            

    quantized_selected_time_idx = quantize_selected_time_idx(selected_time_idx, num_frames_each_base_frame)
    i = 0
    for idx in quantized_selected_time_idx:
        for k, v in params_ls[idx].items():
            if k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']:
                params_ls[idx][k] = cat_params_splitted[k][i]
        i += 1

    for param in params_ls:
        for k, v in param.items():
            if k in ['cam_unnorm_rot', 'cam_trans']:
                param[k] = cat_params[k]
    return params_ls


def update_variables_ls(variables_ls, selected_time_idx, cat_variables, num_gs_per_frame, num_frames_each_base_frame):
    cat_variables_splitted = {}
    for k, v in cat_variables.items():
        if k in ['max_2D_radius', 'means2D_gradient_accum', 'denom', 'timestep']:
            cat_variables_splitted[k] = torch.split(v, num_gs_per_frame, dim=0)
    
    quantized_selected_time_idx = quantize_selected_time_idx(selected_time_idx, num_frames_each_base_frame)
    i = 0
    for idx in quantized_selected_time_idx:
        for k, v in variables_ls[idx].items():
            if k in ['max_2D_radius', 'means2D_gradient_accum', 'denom', 'timestep']:
                variables_ls[idx][k] = cat_variables_splitted[k][i]
        i += 1

    return variables_ls




def geometric_edge_mask(rgb_image: np.ndarray, dilate: bool = True, RGB: bool = False) -> np.ndarray:
    """ Computes an edge mask for an RGB image using geometric edges.
    Args:
        rgb_image: The RGB image.
        dilate: Whether to dilate the edges.
        RGB: Indicates if the image format is RGB (True) or BGR (False).
    Returns:
        An edge mask of the input image.
    """
    # Convert the image to grayscale as Canny edge detection requires a single channel image
    gray_image = cv2.cvtColor(
        rgb_image, cv2.COLOR_BGR2GRAY if not RGB else cv2.COLOR_RGB2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=200, apertureSize=3, L2gradient=True)
    # Define the structuring element for dilation, you can change the size for a thicker/thinner mask
    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def get_frustum_mask(w2c, intrinsics, points, H, W):
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(points.device)
    homo_points = torch.cat([points, ones], dim=1).reshape(
        -1, 4, 1).to(points.device).float()  # (N, 4)
    # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord_homo = w2c @ homo_points
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)

    uv = intrinsics.float() @ cam_cord.float()
    z = uv[:, -1:] + 1e-8
    uv = uv[:, :2] / z
    uv = uv.float()
    edge = 0
    cur_mask_seen = (uv[:, 0] < W - edge) & (
        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
    cur_mask_seen = cur_mask_seen & (z[:, :, 0] > 0)
    cur_mask_seen = cur_mask_seen.reshape(-1)

    return cur_mask_seen




def compute_point2plane_dist(dataset, latest_frame_id, curr_frame_id, latest_w2c, curr_w2c, config, frustum=True, latest_varmask=None, curr_varmask=None, iter=None, output_dir=None, method='sum'):
    color, depth, intrinsics, pose = dataset[latest_frame_id]
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    intrinsics = intrinsics[:3, :3]
    w2c0 = latest_w2c
    mask = (depth > 0)
    if latest_varmask is not None:
        mask = mask & latest_varmask
    mask = mask.reshape(-1)
    
    from kornia.geometry import depth_to_normals
    normal0 = depth_to_normals(depth.unsqueeze(0), intrinsics.unsqueeze(0)).squeeze()
    normal0 = normal0.permute(1, 2, 0).reshape(-1, 3)
    normal0 = normal0[mask].cpu().numpy()
    normal0 = trans_normal_c2w(normal0, w2c0)
    normal0_cu = torch.tensor(normal0).cuda()
    pt_cld = get_pointcloud(color, depth, intrinsics, w2c0, mask=mask, factor=1)
    pt0_cu = pt_cld.clone().detach()


    color, depth, intrinsics, pose = dataset[curr_frame_id]
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    intrinsics = intrinsics[:3, :3]
    w2c1 = curr_w2c
    mask = (depth > 0)
    if curr_varmask is not None:
        mask = mask & curr_varmask
    mask = mask.reshape(-1)

    if iter is not None:
        from kornia.geometry import depth_to_normals
        normal1 = depth_to_normals(depth.unsqueeze(0), intrinsics.unsqueeze(0)).squeeze()
        normal1 = normal1.permute(1, 2, 0).reshape(-1, 3)
        normal1 = normal1[mask].cpu().numpy()
        normal1 = trans_normal_c2w(normal1, w2c1)
        normal1_cu = torch.tensor(normal1).cuda()

    pt_cld = get_pointcloud(color, depth, intrinsics, w2c1, mask=mask, factor=1)
    pt1_cu = pt_cld.clone().detach()
    


    pt0_cu = pt0_cu[:, :3]
    pt1_cu = pt1_cu[:, :3]

    if frustum:
        frustum_mask_0 = get_frustum_mask(w2c1, intrinsics, pt0_cu, config['data']['desired_image_height'], config['data']['desired_image_width'])
        frustum_mask_1 = get_frustum_mask(w2c0, intrinsics, pt1_cu, config['data']['desired_image_height'], config['data']['desired_image_width'])

        frustum_pt0 = pt0_cu[frustum_mask_0]
        frustum_pt1 = pt1_cu[frustum_mask_1]
        frustum_normal0 = normal0_cu[frustum_mask_0]
    else:
        frustum_pt0 = pt0_cu
        frustum_pt1 = pt1_cu
        frustum_normal0 = normal0_cu

    import open3d as o3d
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(frustum_pt0[:, :3].cpu().numpy())
    target.normals = o3d.utility.Vector3dVector(frustum_normal0.cpu().numpy())
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(frustum_pt1[:, :3].cpu().numpy())
    threshold = 0.02
    trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    correspondence_set = evaluation.correspondence_set # source, target
    correspondence_set = np.array(correspondence_set)

    frustum_pt0_paired = frustum_pt0[correspondence_set[:, 1]]
    frustum_pt1_paired = frustum_pt1[correspondence_set[:, 0]]
    frustum_normal0_paired = frustum_normal0[correspondence_set[:, 1]]



    point2plane_dist = torch.mul(frustum_normal0_paired, frustum_pt1_paired - frustum_pt0_paired).sum(dim=1)
    if method == 'sum':
        point2plane_dist = ((point2plane_dist) ** 2).sum()
    elif method == 'max':
        point2plane_dist = torch.abs(point2plane_dist).max()
    elif method == 'max100':
        point2plane_dist = torch.abs(point2plane_dist).topk(100)[0].mean()
    return point2plane_dist



def trans_normal_c2w(normal, w2c):
    """
    Transform the normal vector from camera coordinates to world coordinates.
    """
    if type(normal) == np.ndarray:
        normal = torch.from_numpy(normal).cuda().float()
    normal = normal.reshape(-1, 3)
    normal_ones = torch.ones(normal.shape[0], 1).cuda().float()
    normal4 = torch.cat((normal, normal_ones), dim=1)
    c2w = torch.inverse(w2c)
    normal = (c2w @ normal4.T).T[:, :3]

    normal_start = torch.zeros_like(normal)
    normal_start_ones = torch.ones(normal_start.shape[0], 1).cuda().float()
    normal_start4 = torch.cat((normal_start, normal_start_ones), dim=1)
    normal_start = (c2w @ normal_start4.T).T[:, :3]

    normal_transformed = normal - normal_start
    normal_transformed = normal_transformed.cpu().numpy()

    return normal_transformed


















def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_dir_new = os.path.join(output_dir, "eval_new")
    os.makedirs(eval_dir_new, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    dataset_name = gradslam_data_cfg["dataset_name"]
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False

    print("Dataset Name:", dataset_name)

    ############################################################################################################
    random_select = False 
    use_one_scale_for_each_frame = False



    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)


    # number of frames for each base frame set
    num_frames_each_base_frame = config['baseframe_every']

    color0, depth0, _, _ = dataset[0]
    color0_np = color0.cpu().numpy()
    mask_variation = geometric_edge_mask(color0_np, dilate=True, RGB=True)


    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam, params_ls, variables_ls = initialize_params_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset, 
                                                                        gaussian_distribution=config['gaussian_distribution'],
                                                                        mask_variation=mask_variation)  
                                                               
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, params_ls, variables_ls = initialize_params_first_timestep(dataset, num_frames,
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'],
                                                                                        mask_variation=mask_variation)
    
    torch.cuda.empty_cache()
    
    # Initialize list to keep track of Keyframes
    baseframe_list = []
    tracking_baseframe_list = []
    baseframe_time_indices = []
    keyframe_list = []
    keyframe_time_indices = []

    baseframe_corr_list = []
    tracking_baseframe_corr_list = []
    earliest_baseframe_corr_list = []

    presence_sil_mask_mse_ls = []
    presence_sil_mask_mse_ls1 = []
    presence_sil_mask_mse_ls2 = []
    sil_thres_ls = []
    sil_thres_ls1 = []
    sil_thres_ls2 = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # post num of points
    post_num_pts = params_ls[0]['means3D'].shape[0]

    # numofgs
    num_gs_per_frame_ls = [params['means3D'].shape[0]]

    depth_mean_ls = []

    if dataset_name == 'scannetpp':
        ### FOR Scannet++ ###
        intrinsic_np = intrinsics.cpu().numpy()
        print("Intrinsic Matrix:", intrinsic_np)
  
        odometer = VisualOdometer(intrinsic_np, config["odometer_method"])
        frame_depth_loss = []
        frame_color_loss = []
        odometry_type = config["odometry_type"]
        help_camera_initialization = config["help_camera_initialization"]
        init_err_ratio = config["init_err_ratio"]

    
        
    # Load Checkpoint
    #############################
    # TODO
    if config['load_checkpoint']:
        pass
    else:
        checkpoint_time_idx = 0

    ################################################################################
    # Eval 
    ################################################################################
    eval_with_combined_base_frame = config['eval_mode']
    if eval_with_combined_base_frame:
        params_ls_load = np.load(os.path.join(output_dir, "params_ls.npy"), allow_pickle=True)
        with torch.no_grad():
            eval(dataset, params_ls_load, num_frames, eval_dir, num_gs_per_frame=None, sil_thres=config['mapping']['sil_thres'],
                        wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                        eval_every=config['eval_every'], baseframe_every=config['baseframe_every'], save_frames=True)


    ################################################################################
    # Mapping & Tracking  
    ################################################################################
    else:    
        # Iterate over Scan
        for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
            
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]

            # Generate the variation mask
            color_np = color.cpu().numpy()
            depth_np = depth.cpu().numpy()
            mask_variation = geometric_edge_mask(color_np, dilate=True, RGB=True)


            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            # Process RGB-D Data
            color = color.permute(2, 0, 1) / 255
            depth = depth.permute(2, 0, 1)
            gt_w2c_all_frames.append(gt_w2c)
            curr_gt_w2c = gt_w2c_all_frames
            # Optimize only current time step for tracking
            iter_time_idx = time_idx
            # Initialize Mapping Data for selected frame
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
            
            if dataset_name != 'replica':
                depth_mean = torch.mean(depth[depth > 0])
                depth_mean_ls.append(depth_mean)
                depth_mean_ls.sort()
                # far depth filter thres
                far_id = min(30, len(depth_mean_ls))
                far_depth_filter_thres = config['far_depth_factor'] * torch.tensor(depth_mean_ls[-far_id:]).mean()
            else:
                far_depth_filter_thres = None

            
            # Initialize Data for Tracking
            tracking_curr_data = curr_data

            # Optimization Iterations
            num_iters_mapping = config['mapping']['num_iters']

            # Initialize the camera pose for the current frame
            base_frame_idx = int(time_idx/config['baseframe_every'])
            idx_in_base_frame_set = time_idx % config['baseframe_every']


            if time_idx > 0:
                if dataset_name == 'scannetpp':
                    num_iters_tracking = config['tracking']['num_iters']
                    tracking_sil_thres = config['tracking']['sil_thres']
                    if 'sil_thres_base' not in config['tracking']:
                        config['tracking']['sil_thres_base'] = None
                    if idx_in_base_frame_set == 0 and config['tracking']['sil_thres_base'] is not None:
                        tracking_sil_thres = config['tracking']['sil_thres_base']

                    if (help_camera_initialization or odometry_type == "odometer") and odometer.last_rgbd is None:
                        last_color, last_depth, _, _ = dataset[time_idx - 1]
                        last_color = last_color.cpu().numpy().astype(np.uint8)
                        last_depth = last_depth.cpu().numpy()
                        odometer.update_last_rgbd(last_color, last_depth)

                    # initial check
                    iter = -1
                    if idx_in_base_frame_set == 0:
                        params_ls[base_frame_idx-1] = initialize_camera_pose(params_ls[base_frame_idx-1], time_idx, forward_prop=config['tracking']['forward_prop'])
                        _, _, losses, _, _ = get_loss(params_ls[base_frame_idx-1], tracking_curr_data, variables_ls[base_frame_idx-1], iter_time_idx, config['tracking']['loss_weights'],
                                                            config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                            config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                            plot_dir=eval_dir, visualize_tracking_loss=True,
                                                            tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls, far_depth_filter_thres=far_depth_filter_thres)
                        init_color_loss = losses['im'].item()
                        init_depth_loss = losses['depth'].item()
                    else:
                        params_ls[base_frame_idx] = initialize_camera_pose(params_ls[base_frame_idx], time_idx, forward_prop=config['tracking']['forward_prop'])
                        _, _, losses, _, _ = get_loss(params_ls[base_frame_idx-1], tracking_curr_data, variables_ls[base_frame_idx-1], iter_time_idx, config['tracking']['loss_weights'],
                                                            config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                            config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                            plot_dir=eval_dir, visualize_tracking_loss=True,
                                                            tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls, far_depth_filter_thres=far_depth_filter_thres)
                        init_color_loss = losses['im'].item()
                        init_depth_loss = losses['depth'].item()
                    
                    


                    if len(frame_color_loss) > 0 and (init_color_loss > init_err_ratio * np.median(frame_color_loss)
                                            or init_depth_loss > init_err_ratio * np.median(frame_depth_loss)):
                        num_iters_tracking = 2 * num_iters_tracking
                        print(f"Higher initial loss, increasing num_iters to {num_iters_tracking}")
                        if help_camera_initialization and odometry_type != "odometer":
                            last_color, last_depth, _, _ = dataset[time_idx - 1]
                            last_color = last_color.cpu().numpy().astype(np.float32)
                            last_depth = last_depth.cpu().numpy().astype(np.float32)
                            print('last color and depth shape:', last_color.shape, last_depth.shape)

    
                            odometer.update_last_rgbd(last_color, last_depth)
                            curr_color, curr_depth = color_np.astype(np.float32), depth_np.astype(np.float32)
                            print('curr color and depth shape:', curr_color.shape, curr_depth.shape)
                            odometer_rel = odometer.estimate_rel_pose(curr_color, curr_depth)
                            odometer_rel = torch.tensor(odometer_rel).float()

                            print(f"re-init with odometer for frame {time_idx}")
                        else:
                            odometer_rel = None
                    else:
                        odometer_rel = None
                else:
                    odometer_rel = None


                # selcet tracking overlap 
                if idx_in_base_frame_set == 0:
                    tracking_baseframe_list = copy.deepcopy(baseframe_list)
                    params_ls[base_frame_idx-1] = initialize_camera_pose(params_ls[base_frame_idx-1], time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                    candidate_cam_unnorm_rot = params_ls[base_frame_idx-1]['cam_unnorm_rots'][..., time_idx].detach()
                    candidate_cam_tran = params_ls[base_frame_idx-1]['cam_trans'][..., time_idx].detach()
                    candidate_cam_w2c = torch.eye(4).cuda().float()
                    candidate_cam_w2c[:3, :3] = build_rotation(candidate_cam_unnorm_rot)
                    candidate_cam_w2c[:3, 3] = candidate_cam_tran
   

                if config['tracking']['onlybase_overlap']: 
                    if idx_in_base_frame_set == 0:
                        # Select the most overlapping baseframes for Mapping
                        num_baseframes = config['mapping_window_size']-2
                        if base_frame_idx == 1:
                            if dataset_name == 'replica':
                                tracking_selected_baseframes = [0]
                                tracking_selected_baseframes_time_idx = [0]
                            else:
                                earliest_tracking_selected_baseframes_time_idx = [0]
                                earliest_tracking_selected_baseframes = [0]
                        else:
                            if dataset_name == 'replica':
                                tracking_selected_baseframes = keyframe_selection_overlap(depth, candidate_cam_w2c, intrinsics, tracking_baseframe_list, num_baseframes)
                                tracking_selected_baseframes_time_idx = [tracking_baseframe_list[frame_idx]['id'] for frame_idx in tracking_selected_baseframes]
                            elif dataset_name == 'scannetpp':
                                tracking_selected_baseframes = [base_frame_idx-1]
                                tracking_selected_baseframes_time_idx = [base_frame_idx-1]
                                earliest_tracking_selected_baseframes_time_idx = [base_frame_idx-1]
                                earliest_tracking_selected_baseframes = [base_frame_idx-1]
                            else:
                                ignore_curr_keyframe_id = int(config['baseframe_every'] / config['overlap_every'])
                                if base_frame_idx <= 2:
                                    earliest_tracking_selected_baseframes = keyframe_selection_overlap_visbased_earliest_dynamic_new_topkbase(depth, candidate_cam_w2c, intrinsics, tracking_baseframe_list[:(-ignore_curr_keyframe_id+1)], 
                                                                                                    num_baseframes, config, kf_depth_thresh=config['tracking']['kf_depth_thresh'], earliest_thres=config['tracking']['earliest_thres'],
                                                                                                    lower_earliest_thres_percent=config['tracking']['lower_earliest_thres_percent'], topk_base=None) #######################MODIFIED################################
                                else:
                                    earliest_tracking_selected_baseframes = keyframe_selection_overlap_visbased_earliest_dynamic_new_topkbase(depth, candidate_cam_w2c, intrinsics, tracking_baseframe_list[:(-ignore_curr_keyframe_id+1)], 
                                                                                                    num_baseframes, config, kf_depth_thresh=config['tracking']['kf_depth_thresh'], earliest_thres=config['tracking']['earliest_thres'],
                                                                                                    lower_earliest_thres_percent=config['tracking']['lower_earliest_thres_percent'], topk_base=config['tracking']['topk_base'])

                        if dataset_name == 'replica':
                            tracking_selected_time_idx = []
                        else:
                            earliest_tracking_selected_time_idx = []
                    
                        if dataset_name == 'replica':
                            # Add the most overlapping baseframe and last baseframe to the selected keyframes
                            tracking_selected_time_idx.append(tracking_selected_baseframes_time_idx[-1])
                            tracking_selected_time_idx.append((base_frame_idx-1)*config['baseframe_every'])
                            # Add current frame to the selected keyframes
                            tracking_selected_time_idx.append(time_idx)
                            # save the corresponding tracking_baseframe idx
                            tracking_baseframe_corr_list.append(tracking_selected_time_idx)
                            earliest_baseframe_corr_list.append(find_earliest_keyframe(tracking_baseframe_corr_list, depth, candidate_cam_w2c, intrinsics, tracking_baseframe_list, num_baseframes, config['tracking']['edge'], config['baseframe_every'], config['tracking']['keyframe_thresh']))

                        else:
    
                            earliest_tracking_selected_time_idx.append(time_idx)
                            earliest_tracking_selected_time_idx.append('selected_baseframes')
                            earliest_tracking_selected_time_idx.append(earliest_tracking_selected_baseframes)

                            earliest_baseframe_corr_list.append(earliest_tracking_selected_time_idx)
                    
                    else:
                        if dataset_name != 'replica':
                            earliest_tracking_selected_baseframes = []
                else:
                    pass
     

                if base_frame_idx == 0:
                    params_ls[base_frame_idx] = initialize_camera_pose(params_ls[base_frame_idx], time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
    
                elif base_frame_idx >= 1:
                    if config['tracking']['onlybase_overlap']:
                        if idx_in_base_frame_set == 0:
                            if dataset_name == 'replica':
                                # tracking_ls = [tracking_baseframe_corr_list[-1][0]] #overlap
                                tracking_ls = [earliest_baseframe_corr_list[-1][0]] #earliest
                            else:
                                if len(earliest_tracking_selected_baseframes) == 1:
                                    tracking_ls = [int(config['baseframe_every']*(earliest_tracking_selected_baseframes[0]))]
                                    print('curr_time_idx', time_idx, 'tracking_ls', tracking_ls)
                                elif len(earliest_tracking_selected_baseframes) == 2:
                                    tracking_ls0 = [int(config['baseframe_every']*(earliest_tracking_selected_baseframes[0]))]
                                    tracking_ls1 = [int(config['baseframe_every']*(earliest_tracking_selected_baseframes[1]))]
                                    print('curr_time_idx', time_idx, 'tracking_ls0', tracking_ls0, 'tracking_ls1', tracking_ls1)
                                elif len(earliest_tracking_selected_baseframes) == 3:
                                    tracking_ls0 = [int(config['baseframe_every']*(earliest_tracking_selected_baseframes[0]))]
                                    tracking_ls1 = [int(config['baseframe_every']*(earliest_tracking_selected_baseframes[1]))]
                                    tracking_ls2 = [int(config['baseframe_every']*(earliest_tracking_selected_baseframes[2]))]
                                    print('curr_time_idx', time_idx, 'tracking_ls0', tracking_ls0, 'tracking_ls1', tracking_ls1, 'tracking_ls2', tracking_ls2)

                        else:
                            tracking_ls = [int(config['baseframe_every']*(base_frame_idx))]
   
                    else:
                        continue

                    if dataset_name == 'replica':
                        tracking_cat_params, tracking_cat_variables, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls, config['baseframe_every'])
                        if idx_in_base_frame_set == 0:
                            tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                            tracking_cat_params['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                        else:
                            tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx]['cam_unnorm_rots'].clone().detach()
                            tracking_cat_params['cam_trans'] = params_ls[base_frame_idx]['cam_trans'].clone().detach()
                        tracking_cat_params = initialize_camera_pose(tracking_cat_params, time_idx, forward_prop=config['tracking']['forward_prop'])
                        
                    

                    else:
                        if idx_in_base_frame_set == 0:
                            if len(earliest_tracking_selected_baseframes) == 2:
                                tracking_cat_params, tracking_cat_variables, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls0, config['baseframe_every'])
                                tracking_cat_params1, tracking_cat_variables1, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls1, config['baseframe_every'])
                                tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                                tracking_cat_params['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                                tracking_cat_params1['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                                tracking_cat_params1['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                                tracking_cat_params = initialize_camera_pose(tracking_cat_params, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                                tracking_cat_params1 = initialize_camera_pose(tracking_cat_params1, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                            elif len(earliest_tracking_selected_baseframes) == 3:
                                tracking_cat_params, tracking_cat_variables, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls0, config['baseframe_every'])
                                tracking_cat_params1, tracking_cat_variables1, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls1, config['baseframe_every'])
                                tracking_cat_params2, tracking_cat_variables2, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls2, config['baseframe_every'])
                                tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                                tracking_cat_params['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                                tracking_cat_params1['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                                tracking_cat_params1['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                                tracking_cat_params2['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                                tracking_cat_params2['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                                tracking_cat_params = initialize_camera_pose(tracking_cat_params, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                                tracking_cat_params1 = initialize_camera_pose(tracking_cat_params1, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                                tracking_cat_params2 = initialize_camera_pose(tracking_cat_params2, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                            elif len(earliest_tracking_selected_baseframes) == 1:
                                tracking_cat_params, tracking_cat_variables, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls, config['baseframe_every'])
                                tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots'].clone().detach()
                                tracking_cat_params['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans'].clone().detach()
                                tracking_cat_params = initialize_camera_pose(tracking_cat_params, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)
                        
                        else:
                            tracking_cat_params, tracking_cat_variables, _ = concat_keyframes_params_base_frame(params_ls, variables_ls, tracking_ls, config['baseframe_every'])
                            tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx]['cam_unnorm_rots'].clone().detach()
                            tracking_cat_params['cam_trans'] = params_ls[base_frame_idx]['cam_trans'].clone().detach()
                            tracking_cat_params = initialize_camera_pose(tracking_cat_params, time_idx, forward_prop=config['tracking']['forward_prop'], odometer_rel=odometer_rel)



            ################################################################################
            # Tracking
            ################################################################################
            tracking_start_time = time.time()
            if time_idx > 0 and not config['tracking']['use_gt_poses']:
                # Reset Optimizer & Learning Rates for tracking
                if base_frame_idx == 0:
                    for k, v in params_ls[base_frame_idx].items():
                        if not isinstance(v, torch.Tensor):
                            params_ls[base_frame_idx][k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                        else:
                            params_ls[base_frame_idx][k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                    optimizer = initialize_optimizer(params_ls[base_frame_idx], config['tracking']['lrs'], tracking=True)
                    # Keep Track of Best Candidate Rotation & Translation
                    candidate_cam_unnorm_rot = params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx].detach().clone()
                    candidate_cam_tran = params_ls[base_frame_idx]['cam_trans'][..., time_idx].detach().clone()
                elif base_frame_idx >= 1:
                    if dataset_name == 'replica':
                        for k, v in tracking_cat_params.items():
                            if not isinstance(v, torch.Tensor):
                                tracking_cat_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                            else:
                                tracking_cat_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                        optimizer = initialize_optimizer(tracking_cat_params, config['tracking']['lrs'], tracking=True)
                        # Keep Track of Best Candidate Rotation & Translation
                        candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                
                
                    else:
                        if idx_in_base_frame_set == 0:
                            if len(earliest_tracking_selected_baseframes) == 2:
                                for k, v in tracking_cat_params.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                for k, v in tracking_cat_params1.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params1[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params1[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                optimizer = initialize_optimizer(tracking_cat_params, config['tracking']['lrs'], tracking=True)
                                optimizer1 = initialize_optimizer(tracking_cat_params1, config['tracking']['lrs'], tracking=True)
                                # Keep Track of Best Candidate Rotation & Translation
                                candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                                candidate_cam_unnorm_rot1 = tracking_cat_params1['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran1 = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                            elif len(earliest_tracking_selected_baseframes) == 3:
                                for k, v in tracking_cat_params.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                for k, v in tracking_cat_params1.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params1[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params1[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                for k, v in tracking_cat_params2.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params2[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params2[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                optimizer = initialize_optimizer(tracking_cat_params, config['tracking']['lrs'], tracking=True)
                                optimizer1 = initialize_optimizer(tracking_cat_params1, config['tracking']['lrs'], tracking=True)
                                optimizer2 = initialize_optimizer(tracking_cat_params2, config['tracking']['lrs'], tracking=True)
                                # Keep Track of Best Candidate Rotation & Translation
                                candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                                candidate_cam_unnorm_rot1 = tracking_cat_params1['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran1 = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                                candidate_cam_unnorm_rot2 = tracking_cat_params2['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran2 = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                            elif len(earliest_tracking_selected_baseframes) == 1:
                                for k, v in tracking_cat_params.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                optimizer = initialize_optimizer(tracking_cat_params, config['tracking']['lrs'], tracking=True)
                                # Keep Track of Best Candidate Rotation & Translation
                                candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                                candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                        else:
                            for k, v in tracking_cat_params.items():
                                if not isinstance(v, torch.Tensor):
                                    tracking_cat_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                else:
                                    tracking_cat_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                            
                            optimizer = initialize_optimizer(tracking_cat_params, config['tracking']['lrs'], tracking=True)
                            # Keep Track of Best Candidate Rotation & Translation
                            candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                            candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
        
                current_min_loss = float(1e20)
                # Tracking Optimization
                iter = 0
                do_continue_slam = False
                if dataset_name != 'scannetpp':
                    num_iters_tracking = config['tracking']['num_iters']
                    if 'base1_num_iters' not in config['tracking']:
                        config['tracking']['base1_num_iters'] = None
                    if base_frame_idx == 0 and config['tracking']['base1_num_iters'] is not None:
                        num_iters_tracking = config['tracking']['base1_num_iters']
                else:
                    pass


                tracking_sil_thres = config['tracking']['sil_thres']
                if 'sil_thres_base' not in config['tracking']:
                    config['tracking']['sil_thres_base'] = None
                if idx_in_base_frame_set == 0 and config['tracking']['sil_thres_base'] is not None:
                    tracking_sil_thres = config['tracking']['sil_thres_base']
                    print('tracking_sil_thres_base', tracking_sil_thres)

                

                
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")

                loss_ls = []
                loss_ls1 = []
                loss_ls2 = []
                

                while True:
                    iter_start_time = time.time()

                    if iter == num_iters_tracking-1:
                        visualize_tracking = config['tracking']['visualize_tracking_loss']
                    else:
                        visualize_tracking = False
                    # Loss for current frame
                    if base_frame_idx == 0:
                        loss, variables_ls[base_frame_idx], losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(params_ls[base_frame_idx], tracking_curr_data, variables_ls[base_frame_idx], iter_time_idx, config['tracking']['loss_weights'],
                                                        config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                        config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                        plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                        tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls, far_depth_filter_thres=far_depth_filter_thres)
                    elif base_frame_idx >= 1:
                        if dataset_name == 'replica':
                            loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                        config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                        config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                        plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                        tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls)
                        else:
                            if dataset_name == 'tum' or 'scannet' or 'scannetpp':
                                vis_mask_thres = config['tracking']['vis_mask_thres']

                            if idx_in_base_frame_set == 0:
                                if len(earliest_tracking_selected_baseframes) == 2:
                                    if iter <= 30:
                                        loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls)
                                        loss1, tracking_cat_variables1, losses1, presence_sil_mask_mse_ls1, sil_thres_ls1 = get_loss(tracking_cat_params1, tracking_curr_data, tracking_cat_variables1, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls1, sil_thres_ls=sil_thres_ls1)
                                        loss_ls.append(loss.item())
                                        loss_ls1.append(loss1.item())
                                    else:
                                        loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls,far_depth_filter_thres=far_depth_filter_thres, vis_mask_thres=vis_mask_thres,
                                                                curr_w2c=curr_w2c, overlap_w2c=overlap_w2c, overlap_gtdepth=overlap_gtdepth, overlap_last_w2c=overlap_last_w2c, overlap_last_gtdepth=overlap_last_gtdepth, overlap_mid_w2c=overlap_mid_w2c, overlap_mid_gtdepth=overlap_mid_gtdepth)
                                if len(earliest_tracking_selected_baseframes) == 3:
                                    if iter <= 30:
                                        loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls)
                                        loss1, tracking_cat_variables1, losses1, presence_sil_mask_mse_ls1, sil_thres_ls1 = get_loss(tracking_cat_params1, tracking_curr_data, tracking_cat_variables1, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls1, sil_thres_ls=sil_thres_ls1)
                                        loss2, tracking_cat_variables2, losses2, presence_sil_mask_mse_ls2, sil_thres_ls2 = get_loss(tracking_cat_params2, tracking_curr_data, tracking_cat_variables2, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls2, sil_thres_ls=sil_thres_ls2)
                                        loss_ls.append(loss.item())
                                        loss_ls1.append(loss1.item())
                                        loss_ls2.append(loss2.item())
                                    else:
                                        loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls,far_depth_filter_thres=far_depth_filter_thres, vis_mask_thres=vis_mask_thres,
                                                                curr_w2c=curr_w2c, overlap_w2c=overlap_w2c, overlap_gtdepth=overlap_gtdepth, overlap_last_w2c=overlap_last_w2c, overlap_last_gtdepth=overlap_last_gtdepth, overlap_mid_w2c=overlap_mid_w2c, overlap_mid_gtdepth=overlap_mid_gtdepth)
                                if len(earliest_tracking_selected_baseframes) == 1:
                                    loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls, far_depth_filter_thres=far_depth_filter_thres)
                            
                            else:
                                loss, tracking_cat_variables, losses, presence_sil_mask_mse_ls, sil_thres_ls = get_loss(tracking_cat_params, tracking_curr_data, tracking_cat_variables, iter_time_idx, config['tracking']['loss_weights'],
                                                                config['tracking']['use_sil_for_loss'], tracking_sil_thres,
                                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, dataset_name=dataset_name,
                                                                plot_dir=eval_dir, visualize_tracking_loss=visualize_tracking,
                                                                tracking_iteration=iter, presence_sil_mask_mse_ls=presence_sil_mask_mse_ls, sil_thres_ls=sil_thres_ls, far_depth_filter_thres=far_depth_filter_thres)
                    
                    if config['use_wandb']:
                        # Report Loss
                        wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                    # Backprop


                    if dataset_name == 'replica':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        if config['tracking']['onlybase_overlap']:
                            if idx_in_base_frame_set != 0:
                                choose_metric = loss
                            else:
                                target_frame_idx = time_idx - 1
                                target_overlap_frame_idx = earliest_baseframe_corr_list[-1][0]
                                source_frame_idx = time_idx
                                if source_frame_idx != time_idx:
                                    raise ValueError('source_frame_idx != time_idx')
                                # Select base overlap p2pdist
                                latest_w2c_CD = torch.eye(4).cuda().float()
                                if base_frame_idx == 0:
                                    latest_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., target_frame_idx].detach())
                                    latest_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., target_frame_idx].detach()
                                elif base_frame_idx >= 1:
                                    latest_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., target_frame_idx].detach())
                                    latest_cam_tran_CD = tracking_cat_params['cam_trans'][..., target_frame_idx].detach()
                                latest_w2c_CD[:3, :3] = build_rotation(latest_cam_rot_CD)
                                latest_w2c_CD[:3, 3] = latest_cam_tran_CD

                                latest_overlap_w2c_CD = torch.eye(4).cuda().float()
                                latest_overlap_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., target_overlap_frame_idx].detach())
                                latest_overlap_cam_tran_CD = tracking_cat_params['cam_trans'][..., target_overlap_frame_idx].detach()
                                latest_overlap_w2c_CD[:3, :3] = build_rotation(latest_overlap_cam_rot_CD)
                                latest_overlap_w2c_CD[:3, 3] = latest_overlap_cam_tran_CD

                                curr_w2c_CD = torch.eye(4).cuda().float()
                                if base_frame_idx == 0:
                                    curr_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., source_frame_idx].detach())
                                    curr_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., source_frame_idx].detach()
                                elif base_frame_idx >= 1:
                                    curr_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., source_frame_idx].detach())
                                    curr_cam_tran_CD = tracking_cat_params['cam_trans'][..., source_frame_idx].detach()
                                curr_w2c_CD[:3, :3] = build_rotation(curr_cam_rot_CD)
                                curr_w2c_CD[:3, 3] = curr_cam_tran_CD

                                overlap_point2plane_dist = compute_point2plane_dist(dataset, latest_frame_id=target_overlap_frame_idx, curr_frame_id=source_frame_idx, latest_w2c=latest_overlap_w2c_CD, curr_w2c=curr_w2c_CD, config=config, frustum=config['tracking']['frustum'], method=config['tracking']['p2p_method'])
                                choose_metric = overlap_point2plane_dist 
                        else:
                            target_frame_idx = time_idx - 1
                            source_frame_idx = time_idx
                    
                            # Select base overlap p2pdist
                            latest_w2c_CD = torch.eye(4).cuda().float()
                            if base_frame_idx == 0:
                                latest_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., target_frame_idx].detach())
                                latest_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., target_frame_idx].detach()
                            elif base_frame_idx >= 1:
                                latest_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., target_frame_idx].detach())
                                latest_cam_tran_CD = tracking_cat_params['cam_trans'][..., target_frame_idx].detach()
                            latest_w2c_CD[:3, :3] = build_rotation(latest_cam_rot_CD)
                            latest_w2c_CD[:3, 3] = latest_cam_tran_CD

                            curr_w2c_CD = torch.eye(4).cuda().float()
                            if base_frame_idx == 0:
                                curr_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., source_frame_idx].detach())
                                curr_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., source_frame_idx].detach()
                            elif base_frame_idx >= 1:
                                curr_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., source_frame_idx].detach())
                                curr_cam_tran_CD = tracking_cat_params['cam_trans'][..., source_frame_idx].detach()
                            curr_w2c_CD[:3, :3] = build_rotation(curr_cam_rot_CD)
                            curr_w2c_CD[:3, 3] = curr_cam_tran_CD

                            point2plane_dist = compute_point2plane_dist(dataset, latest_frame_id=target_frame_idx, curr_frame_id=source_frame_idx, latest_w2c=latest_w2c_CD, curr_w2c=curr_w2c_CD, config=config, frustum=config['tracking']['frustum'], method=config['tracking']['p2p_method'])
                            choose_metric = point2plane_dist



                        with torch.no_grad():
                            # Save the best candidate rotation & translation
                            if choose_metric < current_min_loss:
                                current_min_loss = choose_metric
                                if base_frame_idx == 0:
                                    candidate_cam_unnorm_rot = params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx].detach().clone()
                                    candidate_cam_tran = params_ls[base_frame_idx]['cam_trans'][..., time_idx].detach().clone()
                                elif base_frame_idx >= 1:
                                    candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                                    candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                                
                            # Report Progress
                            if config['report_iter_progress']:
                                pass
                                # TODO
                            else:
                                progress_bar.update(1)

                    # for other dataset: tum, scannet, scannetpp
                    elif dataset_name == 'tum' or dataset_name == 'scannet' or dataset_name == 'scannetpp':
                        if base_frame_idx >= 1:
                            if idx_in_base_frame_set == 0 and iter <= 30:
                                if len(earliest_tracking_selected_baseframes) == 2:
                                    loss.backward()
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                                    loss1.backward()
                                    optimizer1.step()
                                    optimizer1.zero_grad(set_to_none=True)
                                elif len(earliest_tracking_selected_baseframes) == 3:
                                    loss.backward()
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                                    loss1.backward()
                                    optimizer1.step()
                                    optimizer1.zero_grad(set_to_none=True)
                                    loss2.backward()
                                    optimizer2.step()
                                    optimizer2.zero_grad(set_to_none=True)
                                elif len(earliest_tracking_selected_baseframes) == 1:
                                    loss.backward()
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                            else:
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                            
                        else:
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                        if idx_in_base_frame_set == 0 and iter <= 30 and len(earliest_tracking_selected_baseframes) > 1:
                            if iter == 30:
                                with torch.no_grad():
                                    if len(earliest_tracking_selected_baseframes) == 2:
                                        min_loss = min(loss_ls)
                                        min_loss1 = min(loss_ls1)
                                        print('min_loss_ls', min_loss, min_loss1)
                                        if min_loss1 < min_loss:
                                            tracking_cat_params = tracking_cat_params1
                                            tracking_cat_variables = tracking_cat_variables1
                                            print('choose baseframe 1', earliest_tracking_selected_baseframes[1])
                                            choose_overlapping_baseframe_id = earliest_tracking_selected_baseframes[1] * config['baseframe_every']
                                        else:
                                            tracking_cat_params = tracking_cat_params
                                            tracking_cat_variables = tracking_cat_variables
                                            print('choose baseframe 0', earliest_tracking_selected_baseframes[0])
                                            choose_overlapping_baseframe_id = earliest_tracking_selected_baseframes[0] * config['baseframe_every']
                                        del tracking_cat_params1, tracking_cat_variables1
                                        
                                    elif len(earliest_tracking_selected_baseframes) == 3:
                                        min_loss = min(loss_ls)
                                        min_loss1 = min(loss_ls1)
                                        min_loss2 = min(loss_ls2)
                                        min_loss_idx = np.argmin([min_loss, min_loss1, min_loss2])
                                        print('min_loss_ls', min_loss, min_loss1, min_loss2)
                                        if min_loss_idx == 0:
                                            tracking_cat_params = tracking_cat_params
                                            tracking_cat_variables = tracking_cat_variables
                                            print('choose baseframe 0', earliest_tracking_selected_baseframes[0])
                                            choose_overlapping_baseframe_id = earliest_tracking_selected_baseframes[0] * config['baseframe_every']
                                        elif min_loss_idx == 1:
                                            tracking_cat_params = tracking_cat_params1
                                            tracking_cat_variables = tracking_cat_variables1
                                            print('choose baseframe 1', earliest_tracking_selected_baseframes[1])
                                            choose_overlapping_baseframe_id = earliest_tracking_selected_baseframes[1] * config['baseframe_every']
                                        elif min_loss_idx == 2:
                                            tracking_cat_params = tracking_cat_params2
                                            tracking_cat_variables = tracking_cat_variables2
                                            print('choose baseframe 2', earliest_tracking_selected_baseframes[2])
                                            choose_overlapping_baseframe_id = earliest_tracking_selected_baseframes[2] * config['baseframe_every']
                                        del tracking_cat_params1, tracking_cat_variables1, tracking_cat_params2, tracking_cat_variables2


                                        

                                        torch.cuda.empty_cache()
                                
                                for k, v in tracking_cat_params.items():
                                    if not isinstance(v, torch.Tensor):
                                        tracking_cat_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                                    else:
                                        tracking_cat_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
                                optimizer = initialize_optimizer(tracking_cat_params, config['tracking']['lrs'], tracking=True)

                                curr_w2c = torch.eye(4).cuda().float()
                                curr_cam_rot = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach())
                                curr_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach()
                                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                                curr_w2c[:3, 3] = curr_cam_tran

                                overlap_w2c = torch.eye(4).cuda().float()
                                overlap_cam_rot = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., choose_overlapping_baseframe_id].detach())
                                overlap_cam_tran = tracking_cat_params['cam_trans'][..., choose_overlapping_baseframe_id].detach()
                                overlap_w2c[:3, :3] = build_rotation(overlap_cam_rot)
                                overlap_w2c[:3, 3] = overlap_cam_tran

                                _, overlap_gtdepth, _, _ = dataset[choose_overlapping_baseframe_id]
                                overlap_gtdepth = overlap_gtdepth.permute(2, 0, 1)

                                if dataset_name == 'scannet' or dataset_name == 'scannetpp':
                                    overlap_mid_w2c = torch.eye(4).cuda().float()
                                    overlap_mid_id = int(choose_overlapping_baseframe_id + config['baseframe_every'] // 2)
                                    overlap_mid_cam_rot = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., overlap_mid_id].detach())
                                    overlap_mid_cam_tran = tracking_cat_params['cam_trans'][..., overlap_mid_id].detach()
                                    overlap_mid_w2c[:3, :3] = build_rotation(overlap_mid_cam_rot)
                                    overlap_mid_w2c[:3, 3] = overlap_mid_cam_tran

                                    overlap_last_w2c = torch.eye(4).cuda().float()
                                    overlap_last_id = int(choose_overlapping_baseframe_id + config['baseframe_every'] - 1)
                                    overlap_last_cam_rot = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., overlap_last_id].detach())
                                    overlap_last_cam_tran = tracking_cat_params['cam_trans'][..., overlap_last_id].detach()
                                    overlap_last_w2c[:3, :3] = build_rotation(overlap_last_cam_rot)
                                    overlap_last_w2c[:3, 3] = overlap_last_cam_tran

                                    

                                    _,overlap_mid_gtdepth, _, _ = dataset[overlap_mid_id]
                                    overlap_mid_gtdepth = overlap_mid_gtdepth.permute(2, 0, 1)

                                    _, overlap_last_gtdepth, _, _ = dataset[overlap_last_id]
                                    overlap_last_gtdepth = overlap_last_gtdepth.permute(2, 0, 1)

                                else:
                                    overlap_mid_w2c = None
                                    overlap_last_w2c = None
                                    overlap_mid_gtdepth = None
                                    overlap_last_gtdepth = None
                                
    
                                                



                        else:
                            if dataset_name == 'scannetpp':
                                choose_metric = loss
                            elif dataset_name == 'scannet' or dataset_name == 'tum':
                                if config['tracking']['onlybase_overlap']:
                                    if idx_in_base_frame_set != 0:
                                        choose_metric = loss
                                    else:
                                        target_frame_idx = time_idx - 1
                                        target_overlap_frame_idx = earliest_baseframe_corr_list[-1][0]
                                        target_overlap_frame_idx = int(target_overlap_frame_idx / config['baseframe_every'])*config['baseframe_every']
                                        source_frame_idx = time_idx
                                        if source_frame_idx != time_idx:
                                            raise ValueError('source_frame_idx != time_idx')
                                        # Select base overlap p2pdist
                                        latest_w2c_CD = torch.eye(4).cuda().float()
                                        if base_frame_idx == 0:
                                            latest_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., target_frame_idx].detach())
                                            latest_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., target_frame_idx].detach()
                                        elif base_frame_idx >= 1:
                                            latest_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., target_frame_idx].detach())
                                            latest_cam_tran_CD = tracking_cat_params['cam_trans'][..., target_frame_idx].detach()
                                        latest_w2c_CD[:3, :3] = build_rotation(latest_cam_rot_CD)
                                        latest_w2c_CD[:3, 3] = latest_cam_tran_CD

                                        latest_overlap_w2c_CD = torch.eye(4).cuda().float()
                                        latest_overlap_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., target_overlap_frame_idx].detach())
                                        latest_overlap_cam_tran_CD = tracking_cat_params['cam_trans'][..., target_overlap_frame_idx].detach()
                                        latest_overlap_w2c_CD[:3, :3] = build_rotation(latest_overlap_cam_rot_CD)
                                        latest_overlap_w2c_CD[:3, 3] = latest_overlap_cam_tran_CD

                                        curr_w2c_CD = torch.eye(4).cuda().float()
                                        if base_frame_idx == 0:
                                            curr_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., source_frame_idx].detach())
                                            curr_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., source_frame_idx].detach()
                                        elif base_frame_idx >= 1:
                                            curr_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., source_frame_idx].detach())
                                            curr_cam_tran_CD = tracking_cat_params['cam_trans'][..., source_frame_idx].detach()
                                        curr_w2c_CD[:3, :3] = build_rotation(curr_cam_rot_CD)
                                        curr_w2c_CD[:3, 3] = curr_cam_tran_CD

                                        overlap_point2plane_dist = compute_point2plane_dist(dataset, latest_frame_id=target_overlap_frame_idx, curr_frame_id=source_frame_idx, latest_w2c=latest_overlap_w2c_CD, curr_w2c=curr_w2c_CD, config=config, frustum=config['tracking']['frustum'], method=config['tracking']['p2p_method'])
                                        choose_metric = overlap_point2plane_dist 
                                else:
                                    target_frame_idx = time_idx - 1
                                    source_frame_idx = time_idx
                            
                                    # Select base overlap p2pdist
                                    latest_w2c_CD = torch.eye(4).cuda().float()
                                    if base_frame_idx == 0:
                                        latest_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., target_frame_idx].detach())
                                        latest_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., target_frame_idx].detach()
                                    elif base_frame_idx >= 1:
                                        latest_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., target_frame_idx].detach())
                                        latest_cam_tran_CD = tracking_cat_params['cam_trans'][..., target_frame_idx].detach()
                                    latest_w2c_CD[:3, :3] = build_rotation(latest_cam_rot_CD)
                                    latest_w2c_CD[:3, 3] = latest_cam_tran_CD

                                    curr_w2c_CD = torch.eye(4).cuda().float()
                                    if base_frame_idx == 0:
                                        curr_cam_rot_CD = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., source_frame_idx].detach())
                                        curr_cam_tran_CD = params_ls[base_frame_idx]['cam_trans'][..., source_frame_idx].detach()
                                    elif base_frame_idx >= 1:
                                        curr_cam_rot_CD = F.normalize(tracking_cat_params['cam_unnorm_rots'][..., source_frame_idx].detach())
                                        curr_cam_tran_CD = tracking_cat_params['cam_trans'][..., source_frame_idx].detach()
                                    curr_w2c_CD[:3, :3] = build_rotation(curr_cam_rot_CD)
                                    curr_w2c_CD[:3, 3] = curr_cam_tran_CD

                                    point2plane_dist = compute_point2plane_dist(dataset, latest_frame_id=target_frame_idx, curr_frame_id=source_frame_idx, latest_w2c=latest_w2c_CD, curr_w2c=curr_w2c_CD, config=config, frustum=config['tracking']['frustum'], method=config['tracking']['p2p_method'])
                                    choose_metric = point2plane_dist



                        with torch.no_grad():
                            # Save the best candidate rotation & translation
                            if choose_metric < current_min_loss:
                                current_min_loss = choose_metric
                                if base_frame_idx == 0:
                                    candidate_cam_unnorm_rot = params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx].detach().clone()
                                    candidate_cam_tran = params_ls[base_frame_idx]['cam_trans'][..., time_idx].detach().clone()
                                elif base_frame_idx >= 1:
                                    candidate_cam_unnorm_rot = tracking_cat_params['cam_unnorm_rots'][..., time_idx].detach().clone()
                                    candidate_cam_tran = tracking_cat_params['cam_trans'][..., time_idx].detach().clone()
                                
                            # Report Progress
                            if config['report_iter_progress']:
                                pass
                                # TODO
                            else:
                                progress_bar.update(1)

                    # Update the runtime numbers
                    iter_end_time = time.time()
                    tracking_iter_time_sum += iter_end_time - iter_start_time
                    tracking_iter_time_count += 1
                    # Check if we should stop tracking
                    iter += 1
                    if iter == num_iters_tracking and dataset_name == 'scannetpp':
                        frame_color_loss.append(losses['im'].item())
                        frame_depth_loss.append(losses['depth'].item())
 
                    if iter == num_iters_tracking:
                        if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                            break
                        elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                            do_continue_slam = True
                            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                            num_iters_tracking = 2*num_iters_tracking
                            if config['use_wandb']:
                                wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                            "Tracking/step": wandb_time_step})
                        else:
                            break

                progress_bar.close()
                # Copy over the best candidate rotation & translation
                with torch.no_grad():
                    if base_frame_idx == 0:
                        params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                        params_ls[base_frame_idx]['cam_trans'][..., time_idx] = candidate_cam_tran
                    elif base_frame_idx >= 1:
                        if idx_in_base_frame_set == 0:
                            candidate_cam_rot = torch.nn.functional.normalize(candidate_cam_unnorm_rot)
                            curr_w2c = torch.eye(4).cuda().float()
                            curr_w2c[:3, :3] = build_rotation(candidate_cam_rot)
                            curr_w2c[:3, 3] = candidate_cam_tran

                            print(f"Adding New Base Frame at Frame {time_idx}")
                            if seperate_densification_res:
                                params, variables = initialize_params_base_timestep(dataset, num_frames, time_idx, curr_w2c,
                                                                                    config['scene_radius_depth_ratio'],
                                                                                    config['mean_sq_dist_method'],
                                                                                    densify_dataset=densify_dataset, 
                                                                                    gaussian_distribution=config['gaussian_distribution'],
                                                                                    mask_variation=mask_variation)
                            else:
                                params, variables = initialize_params_base_timestep(dataset, num_frames, time_idx, curr_w2c,
                                                                                    config['scene_radius_depth_ratio'],
                                                                                    config['mean_sq_dist_method'],
                                                                                    densify_dataset=dataset, 
                                                                                    gaussian_distribution=config['gaussian_distribution'],
                                                                                    mask_variation=mask_variation)

                            num_gs_per_frame_ls.append(params['means3D'].shape[0])
                            
                            params['cam_unnorm_rots'] = params_ls[base_frame_idx-1]['cam_unnorm_rots']
                            params['cam_trans'] = params_ls[base_frame_idx-1]['cam_trans']
                            params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                            params['cam_trans'][..., time_idx] = candidate_cam_tran
                    
                            params_ls.append(params)
                            variables_ls.append(variables)  

                            tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx]['cam_unnorm_rots']
                            tracking_cat_params['cam_trans'] = params_ls[base_frame_idx]['cam_trans']
                        else:
                            params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                            params_ls[base_frame_idx]['cam_trans'][..., time_idx] = candidate_cam_tran

                            tracking_cat_params['cam_unnorm_rots'] = params_ls[base_frame_idx]['cam_unnorm_rots']
                            tracking_cat_params['cam_trans'] = params_ls[base_frame_idx]['cam_trans']


                  

            elif time_idx > 0 and config['tracking']['use_gt_poses']:
                with torch.no_grad():
                    # Get the ground truth pose relative to frame 0
                    rel_w2c = curr_gt_w2c[-1]
                    rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = rel_w2c[:3, 3].detach()

                    # get the absolute pose
                    curr_cam_rot = torch.nn.functional.normalize(rel_w2c_rot_quat)
                    curr_cam_tran = rel_w2c_tran
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Update the camera parameters
                    base_frame_idx = int(time_idx/config['baseframe_every'])
                    idx_in_base_frame_set = time_idx % config['baseframe_every']
                    if time_idx != 0 and idx_in_base_frame_set == 0:
                        # add new params into the list
                        print(f"Adding New Base Frame at Frame {time_idx}")
                            ############################################$$$$$$$$$$$$$$$$############################################

                        params, variables = initialize_params_base_timestep(dataset, num_frames, time_idx, curr_w2c,
                                                                            config['scene_radius_depth_ratio'],
                                                                            config['mean_sq_dist_method'],
                                                                            densify_dataset=densify_dataset, 
                                                                            gaussian_distribution=config['gaussian_distribution'],
                                                                            mask_variation=mask_variation)
                        params['cam_unnorm_rots'] = params_ls[-1]['cam_unnorm_rots']
                        params['cam_trans'] = params_ls[-1]['cam_trans']
                        params_ls.append(params)
                        variables_ls.append(variables)  

                    params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                    params_ls[base_frame_idx]['cam_trans'][..., time_idx] = rel_w2c_tran

            # Update the runtime numbers
            tracking_end_time = time.time()
            tracking_frame_time_sum += tracking_end_time - tracking_start_time
            tracking_frame_time_count += 1



            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Final Tracking Progress
                    progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            if base_frame_idx == 0:
                                report_progress(params_ls[base_frame_idx], tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                                wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                            elif base_frame_idx >= 1:
                                report_progress(tracking_cat_params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                                wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                            
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params_ls[time_idx-1], ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')



            # Densification & KeyFrame-based Mapping
            if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
                # Densification
                if config['mapping']['add_new_gaussians'] and time_idx > 0:
                    # Setup Data for Densification
                    if seperate_densification_res:
                        # Load RGBD frames incrementally instead of all frames
                        densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                        densify_color = densify_color.permute(2, 0, 1) / 255
                        densify_depth = densify_depth.permute(2, 0, 1)
                        densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                    'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                        ori_curr_data = curr_data

                    else:
                        densify_curr_data = curr_data
                        ori_curr_data = curr_data


                    # Add new Gaussians to the scene based on the Silhouette

                    if time_idx % config['baseframe_every'] == 0:
                        # no need to add new gaussians
                        pass

                    else:
                        base_frame_idx = int(time_idx/config['baseframe_every'])
                        
                        # update the params in the list
                        params, variables, add_number = add_new_gaussians_base_frame(params, variables, ori_curr_data, densify_curr_data, config['mapping']['sil_thres'],
                                                                        time_idx, config['mean_sq_dist_method'], config['gaussian_distribution'], config, mask_variation=mask_variation)
                        base_frame_idx = int(time_idx/config['baseframe_every'])
                        params_ls[base_frame_idx] = params
                        variables_ls[base_frame_idx] = variables

                        num_gs_per_frame_ls.append(add_number)
                        

                        print(f"After Adding New Gaussians at Frame {time_idx}", params['means3D'].shape[0])
         

                    post_num_pts += params['means3D'].shape[0]
                    if config['use_wandb']:
                        wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                    "Mapping/step": wandb_time_step})
                
                        

                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    base_frame_idx = int(time_idx/config['baseframe_every'])
                    idx_in_base_frame_set = time_idx % config['baseframe_every']

                    curr_cam_rot = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params_ls[base_frame_idx]['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    if base_frame_idx != 0:
                        if idx_in_base_frame_set == 0:
                            # Select the most overlapping baseframes for Mapping
                            num_baseframes = config['mapping_window_size']-2
                            if base_frame_idx == 1:
                                selected_baseframes = [0]
                                selected_baseframes_time_idx = [0]
                            else:
                                if dataset_name == 'replica':
                                    selected_baseframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, baseframe_list[:-1], num_baseframes) 
                                    selected_baseframes_time_idx = [baseframe_list[frame_idx]['id'] for frame_idx in selected_baseframes]
                                elif dataset_name == 'scannet' or dataset_name == 'scannetpp' or dataset_name == 'tum':
                                    ignore_curr_keyframe_id = int(config['baseframe_every'] / config['overlap_every'])
                                    selected_baseframes, _ = keyframe_selection_overlap_visbased(depth, curr_w2c, intrinsics, baseframe_list[:-ignore_curr_keyframe_id], 
                                                                                            num_baseframes, kf_depth_thresh=config['tracking']['kf_depth_thresh'])

                                    selected_baseframes_time_idx = [int(baseframe_list[selected_baseframes[0]]['id'] / config['baseframe_every']) * config['baseframe_every']]
                            print(f"Selected Baseframes for Mapping at Frame {time_idx}: {selected_baseframes_time_idx}, {selected_baseframes}")


                    
                    if base_frame_idx == 0:
                        if idx_in_base_frame_set == 0:
            
                            selected_time_idx = [time_idx]
                            selected_keyframes = [time_idx]
                                
                        else:
                            se_list = [*range(base_frame_idx * config['baseframe_every'], time_idx)]
                            se_list.append(time_idx)
                            selected_time_idx = se_list
                            selected_keyframes = se_list
            
                    else:
                        if idx_in_base_frame_set == 0:
                            # Select Baseframes for Mapping
                            selected_time_idx = []
                            selected_keyframes = [] 
            
                            choose_lastest_base_frame = False
                            choose_overlap_base_frame = False
                            if choose_lastest_base_frame == True:
                                # Add last baseframe to the selected keyframes
                                selected_time_idx.append((base_frame_idx-1)*config['baseframe_every'])
                                selected_keyframes.append((base_frame_idx-1)*config['baseframe_every'])
                            elif choose_overlap_base_frame == True:
                                # Add the most overlapping baseframe to the selected keyframes
                                selected_time_idx.append(selected_baseframes_time_idx[-1])
                                selected_keyframes.append(selected_baseframes[-1]*config['baseframe_every'])
                            else:
                                # Add the most overlapping baseframe and last baseframe to the selected keyframes
                                selected_time_idx.append(selected_baseframes_time_idx[-1])
                                selected_keyframes.append(selected_baseframes[-1]*config['baseframe_every'])
                                selected_time_idx.append((base_frame_idx-1)*config['baseframe_every'])
                                selected_keyframes.append((base_frame_idx-1)*config['baseframe_every'])


                            # Add current frame to the selected keyframes
                            selected_time_idx.append(time_idx)
                            selected_keyframes.append(time_idx)
            
                                
                        else:
                            se_list = [*range(base_frame_idx * config['baseframe_every'], time_idx)]
                            se_list.append(time_idx)
                            selected_time_idx = se_list
                            selected_keyframes = se_list



                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

                    if idx_in_base_frame_set == 0 and base_frame_idx != 0:
                        # save the corresponding baseframe idx
                        baseframe_corr_list.append(selected_time_idx)
                        print('mapping_baseframe_corr_list', baseframe_corr_list)

                
                

                if idx_in_base_frame_set == 0:
                    if base_frame_idx != 0:
                        fix_params_idx = [int(idx / config['baseframe_every']) for idx in selected_time_idx[-3: -1]]
                        print(f"Fixed Base Frame Indices: {fix_params_idx}")
                        fixed_params_ls = [params_ls[idx] for idx in fix_params_idx]
                        fixed_variables_ls = [variables_ls[idx] for idx in fix_params_idx]
                        fixed_params, fixed_variables, = concat_global(fixed_params_ls[0], fixed_variables_ls[0], None, fixed_params_ls[1], fixed_variables_ls[1])
     
                        for k, v in fixed_params.items():
                            # Check if value is already a torch tensor
                            if not isinstance(v, torch.Tensor):
                                fixed_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
                            else:
                                fixed_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

                        cat_params, cat_variables, num_gs_per_base_frame = concat_keyframes_params_base_frame(params_ls, variables_ls, selected_time_idx[-1:], config['baseframe_every'])
                    else:
                        cat_params, cat_variables, num_gs_per_base_frame = concat_keyframes_params_base_frame(params_ls, variables_ls, selected_time_idx, config['baseframe_every'])
                else:
                    cat_params, cat_variables, num_gs_per_base_frame = concat_keyframes_params_base_frame(params_ls, variables_ls, selected_time_idx, config['baseframe_every'])
                
                
                optimizer = initialize_optimizer(cat_params, config['mapping']['lrs'], tracking=False) 
                if base_frame_idx != 0:
                    optimizer_fixed = initialize_optimizer(fixed_params, config['mapping']['fixed_lrs'], tracking=False)
                    cat_params_global, cat_variables_global, num_gs_per_base_frame_global = concat_global(cat_params, cat_variables, num_gs_per_base_frame, fixed_params, fixed_variables)

        


            
                ################################################################################
                # Mapping
                ################################################################################
                mapping_start_time = time.time()
                if num_iters_mapping > 0:
                    progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                
            

                for iter in range(num_iters_mapping):

                    iter_start_time = time.time()
                    select_one_keyframe = True 
                    if select_one_keyframe:
                        if idx_in_base_frame_set == 0:
                            iter_time_idx = time_idx
                            iter_color = color
                            iter_depth = depth

                            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                            iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                    'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                            

                            if time_idx == 0:
                                ba = False
                            else:
                                ba = True

                            loss, cat_variables, losses = get_loss(cat_params, iter_data, cat_variables, iter_time_idx, config['mapping']['loss_weights'],
                                                            config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                            config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, dataset_name=dataset_name, 
                                                            do_ba=ba, additional_mask=None)
                            
         
                            if base_frame_idx != 0:
                                loss_global, cat_variables_global, losses_global = get_loss(cat_params_global, iter_data, cat_variables_global, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], do_ba=ba, mapping=True, dataset_name=dataset_name)
       
                                loss = loss + loss_global


                        else:
                            ##### Select one keyframe per iteration
                            optimize_current_frame_first = False
                            if optimize_current_frame_first:
                                if iter <= 20:
                                    selected_rand_keyframe_idx = time_idx
                                else:
                                    rand_idx = np.random.randint(0, len(selected_keyframes))
                                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                            else:
                                # Randomly select a frame until current time step amongst keyframes
                                rand_idx = np.random.randint(0, len(selected_keyframes))
                                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                        
                            # print(f"Selected Keyframe Index: {selected_rand_keyframe_idx}")
                            if selected_rand_keyframe_idx == -1 or selected_rand_keyframe_idx == time_idx:
                                # Use Current Frame Data
                                iter_time_idx = time_idx
                                iter_color = color
                                iter_depth = depth
                                
                            else:
                                # Use Keyframe Data
                                iter_time_idx = selected_rand_keyframe_idx
                                iter_color, iter_depth, _, _ = dataset[iter_time_idx]
                                iter_color = iter_color.permute(2, 0, 1) / 255
                                iter_depth = iter_depth.permute(2, 0, 1)
                
                                
                    

                            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                            iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                        'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
           
                      
                            loss, cat_variables, losses = get_loss(cat_params, iter_data, cat_variables, iter_time_idx, config['mapping']['loss_weights'],
                                                            config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                            config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, dataset_name=dataset_name, additional_mask=None)
                            
                            if base_frame_idx != 0 and (selected_rand_keyframe_idx % config['baseframe_every'] == 0):
                                loss_global, cat_variables_global, losses_global = get_loss(cat_params_global, iter_data, cat_variables_global, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, dataset_name=dataset_name)
                                loss = loss + loss_global

                        
                        
                
                        
                    else:
                        ##### Use all keyframes per iteration
                        # Loss for all keyframes
                        loss = 0
                        if idx_in_base_frame_set == 0:
                            for idx in selected_keyframes[-1:]:
                                if idx == -1 or idx == time_idx:
                                    # Use Current Frame Data
                                    iter_time_idx = time_idx
                                    iter_color = color
                                    iter_depth = depth
                                else:
                                    # Use Keyframe Data
                                    iter_time_idx = keyframe_list[idx]['id']
                                    iter_color = keyframe_list[idx]['color']
                                    iter_depth = keyframe_list[idx]['depth']
         
                                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                            'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                                # Loss for current frame
                                loss_per, cat_variables, losses = get_loss(cat_params, iter_data, cat_variables, iter_time_idx, config['mapping']['loss_weights'],
                                                            config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                            config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, dataset_name=dataset_name)
                                
                                if base_frame_idx != 0:
                                    loss_per_global, cat_variables_global, losses_global = get_loss(cat_params_global, iter_data, cat_variables_global, iter_time_idx, config['mapping']['loss_weights'],
                                                        config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                        config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, dataset_name=dataset_name)
      
                                    loss_per = loss_per + loss_per_global   
      

                                loss = loss_per + loss
                        
                        else:

                            for idx in selected_keyframes:
                                if idx == -1 or idx == time_idx:
                                    # Use Current Frame Data
                                    iter_time_idx = time_idx
                                    iter_color = color
                                    iter_depth = depth
                                else:
                                    # Use Keyframe Data
                                    iter_time_idx = keyframe_list[idx]['id']
                                    iter_color = keyframe_list[idx]['color']
                                    iter_depth = keyframe_list[idx]['depth']
                
                                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                            'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                                # Loss for current frame
                                loss_per, cat_variables, losses = get_loss(cat_params, iter_data, cat_variables, iter_time_idx, config['mapping']['loss_weights'],
                                                            config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                            config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, dataset_name=dataset_name)

                                loss = loss_per + loss
                        

                    if config['use_wandb']:
                        # Report Loss
                        wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                    

                    # baseframe cam pose
                    if base_frame_idx != 0 and idx_in_base_frame_set == 0:
                        curr_baseframe_cam_rot = F.normalize(cat_params['cam_unnorm_rots'][..., time_idx].detach())
                        curr_baseframe_cam_tran = cat_params['cam_trans'][..., time_idx].detach()
                        curr_baseframe_w2c = torch.eye(4).cuda().float()
                        curr_baseframe_w2c[:3, :3] = build_rotation(curr_baseframe_cam_rot)
                        curr_baseframe_w2c[:3, 3] = curr_baseframe_cam_tran
                        curr_baseframe_c2w = torch.linalg.inv(curr_baseframe_w2c)
                    


                    # Backprop
                    loss.backward()
                   
                    with torch.no_grad():
                        # Prune Gaussians
                        if config['mapping']['prune_gaussians']:
                            pass
                            # We do not prune gaussians during mapping
                        # Optimizer Update
                        optimizer.step()
                        
                        if base_frame_idx != 0:
                            optimizer_fixed.step()


                        optimizer.zero_grad(set_to_none=True)
                        if base_frame_idx != 0:
                            optimizer_fixed.zero_grad(set_to_none=True)


                        # update gs pos
                        if base_frame_idx != 0 and idx_in_base_frame_set == 0:
                            opt_curr_baseframe_cam_rot = F.normalize(cat_params['cam_unnorm_rots'][..., time_idx].detach())
                            opt_curr_baseframe_cam_tran = cat_params['cam_trans'][..., time_idx].detach()
                            opt_curr_baseframe_w2c = torch.eye(4).cuda().float()
                            opt_curr_baseframe_w2c[:3, :3] = build_rotation(opt_curr_baseframe_cam_rot)
                            opt_curr_baseframe_w2c[:3, 3] = opt_curr_baseframe_cam_tran
                            opt_curr_baseframe_c2w = torch.linalg.inv(opt_curr_baseframe_w2c)


                            # transform gaussians into optimized cam pose
                            num_gs_curr = num_gs_per_frame_ls[-1]
                            pts = cat_params['means3D'][-num_gs_curr:]
                            pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
                            pts4 = torch.cat((pts, pts_ones), dim=1)
                            transformed_pts = (curr_baseframe_w2c @ pts4.T).T[:, :3]
                            transformed_pts_ones = torch.ones(transformed_pts.shape[0], 1).cuda().float()
                            transformed_pts4 = torch.cat((transformed_pts, transformed_pts_ones), dim=1)

                            opt_transformed_pts = (opt_curr_baseframe_c2w @ transformed_pts4.T).T[:, :3]

                            # update the gaussians
                            cat_params['means3D'][-num_gs_curr:] = opt_transformed_pts
                        


                        
                        # update global
                        if base_frame_idx != 0:
                            cat_params_global, cat_variables_global, num_gs_per_base_frame_global = concat_global(cat_params, cat_variables, num_gs_per_base_frame, fixed_params, fixed_variables)
        


                    
                        # Report Progress
                        if config['report_iter_progress']:
                            pass
                                # TODO
                        else:
                            progress_bar.update(1)

                        
                    
                 
                    # Update the runtime numbers
                    iter_end_time = time.time()
                    mapping_iter_time_sum += iter_end_time - iter_start_time
                    mapping_iter_time_count += 1

               
                if idx_in_base_frame_set == 0:
                    if base_frame_idx != 0:
                        params_ls = update_params_ls(params_ls, selected_time_idx[-1:], cat_params, num_gs_per_base_frame, config['baseframe_every'])
                        variables_ls = update_variables_ls(variables_ls, selected_time_idx[-1:], cat_variables, num_gs_per_base_frame, config['baseframe_every'])
                        del cat_params_global, cat_variables_global
                        del cat_params, cat_variables
                        torch.cuda.empty_cache()

                    else:
                        params_ls = update_params_ls(params_ls, selected_time_idx, cat_params, num_gs_per_base_frame, config['baseframe_every'])
                        variables_ls = update_variables_ls(variables_ls, selected_time_idx, cat_variables, num_gs_per_base_frame, config['baseframe_every'])
                        del cat_params, cat_variables
                        torch.cuda.empty_cache()
                else:
                    params_ls = update_params_ls(params_ls, selected_time_idx, cat_params, num_gs_per_base_frame, config['baseframe_every'])
                    variables_ls = update_variables_ls(variables_ls, selected_time_idx, cat_variables, num_gs_per_base_frame, config['baseframe_every'])
                    del cat_params, cat_variables
                    torch.cuda.empty_cache()
               

                if num_iters_mapping > 0:
                    progress_bar.close()
                # Update the runtime numbers
                mapping_end_time = time.time()
                mapping_frame_time_sum += mapping_end_time - mapping_start_time
                mapping_frame_time_count += 1


            # Add frame to keyframe list
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    base_frame_idx = int(time_idx/config['baseframe_every'])
                    # idx_in_base_frame_set = time_idx % config['keyframe_every']
                    curr_cam_rot = F.normalize(params_ls[base_frame_idx]['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params_ls[base_frame_idx]['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}


                    for k, v in curr_keyframe.items():
                        if isinstance(v, torch.Tensor):
                            curr_keyframe[k] = curr_keyframe[k].detach().cpu()
                    torch.cuda.empty_cache()

                    # Add to baseframe list
                    if dataset_name == 'replica':
                        if idx_in_base_frame_set == 0:
                            baseframe_list.append(curr_keyframe)
                            baseframe_time_indices.append(time_idx)
                    elif dataset_name == 'tum' or dataset_name == 'scannet' or dataset_name == 'scannetpp':
                        if time_idx % config['overlap_every'] == 0:
                            baseframe_list.append(curr_keyframe)
                            baseframe_time_indices.append(time_idx)

                    for i in range(len(baseframe_list)):
                        for k, v in baseframe_list[i].items():
                            if isinstance(v, torch.Tensor):
                                baseframe_list[i][k] = baseframe_list[i][k].detach().cpu()
                    torch.cuda.empty_cache()
            

            
            # Checkpoint every iteration
            #############################
            # TODO
            

            # Increment WandB Time Step
            if config['use_wandb']:
                wandb_time_step += 1


            for i in range(len(params_ls)):
                for k, v in params_ls[i].items():
                    if isinstance(v, torch.Tensor):
                        params_ls[i][k] = params_ls[i][k].detach().cpu()
            for i in range(len(variables_ls)):
                for k, v in variables_ls[i].items():
                    if isinstance(v, torch.Tensor):
                        variables_ls[i][k] = variables_ls[i][k].cpu()
            
            
            
            torch.cuda.empty_cache()


        # Compute Average Runtimes
        if tracking_iter_time_count == 0:
            tracking_iter_time_count = 1
            tracking_frame_time_count = 1
        if mapping_iter_time_count == 0:
            mapping_iter_time_count = 1
            mapping_frame_time_count = 1
        tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
        tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
        mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
        mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        print('Number of Gaussians:', post_num_pts)
        if config['use_wandb']:
            wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                        "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                        "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                        "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                        "Final Stats/step": 1})
        
        
        # Save params_ls as numpy
        for i in range(len(params_ls)):
            for k, v in params_ls[i].items():
                if isinstance(v, torch.Tensor):
                    params_ls[i][k] = params_ls[i][k].cpu()
        params_ls_np = np.array(params_ls)
        np.save(os.path.join(output_dir, "params_ls.npy"), params_ls_np)



        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(dataset, params_ls, num_frames, eval_dir, num_gs_per_frame=None, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True, baseframe_every=config['baseframe_every'])


        # Close WandB Run
        if config['use_wandb']:
            wandb.finish()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)