import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation, calc_psnr
from utils.slam_helpers import (
    transform_to_frame, transformed_params2rendervar, transformed_params2depthplussilhouette,
    quat_mult, matrix_to_quaternion
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import open3d as o3d
from utils.evaluate_reconstruction import evaluate_reconstruction
from scipy.ndimage import median_filter
from kornia.filters.median import median_blur, MedianBlur

loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error


def report_loss(losses, wandb_run, wandb_step, tracking=False, mapping=False):
    # Update loss dict
    loss_dict = {'Loss': losses['loss'].item(),
                 'Image Loss': losses['im'].item(),
                 'Depth Loss': losses['depth'].item(),}
    if tracking:
        tracking_loss_dict = {}
        for k, v in loss_dict.items():
            tracking_loss_dict[f"Per Iteration Tracking/{k}"] = v
        tracking_loss_dict['Per Iteration Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif mapping:
        mapping_loss_dict = {}
        for k, v in loss_dict.items():
            mapping_loss_dict[f"Per Iteration Mapping/{k}"] = v
        mapping_loss_dict['Per Iteration Mapping/step'] = wandb_step
        wandb_run.log(mapping_loss_dict)
    else:
        frame_opt_loss_dict = {}
        for k, v in loss_dict.items():
            frame_opt_loss_dict[f"Per Iteration Current Frame Optimization/{k}"] = v
        frame_opt_loss_dict['Per Iteration Current Frame Optimization/step'] = wandb_step
        wandb_run.log(frame_opt_loss_dict)
    
    # Increment wandb step
    wandb_step += 1
    return wandb_step
        

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, 
                         save_plot=False, wandb_run=None, wandb_step=None, wandb_title=None, diff_rgb=None):
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    fig_width = fig_width * aspect_ratio
    # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
    fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    diff_rgb = torch.abs(color - rastered_color).detach().cpu()
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[0, 1].set_title("Ground Truth Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}cm".format(depth_l1*100))
 

    diff_rgb = torch.clamp(diff_rgb, 0, 1)
        
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb.cpu().permute(1, 2, 0), cmap='jet') #, vmin=0, vmax=6)
        axs[0, 2].set_title("Diff RGB L1")
        axs[0, 3].imshow(presence_sil_mask, cmap='gray')
        axs[0, 3].set_title("Rasterized Silhouette")
    else:
        axs[0, 2].imshow(presence_sil_mask, cmap='gray')
        axs[0, 2].set_title("Rasterized Silhouette")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=6)
    axs[1, 2].set_title("Diff Depth L1")
    for ax in axs.flatten():
        ax.axis('off')
    fig.suptitle(fig_title, y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log({wandb_title: fig})
        else:
            wandb_run.log({wandb_title: fig}, step=wandb_step)
    plt.close()



def report_progress(params, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, wandb_run=None, wandb_step=None, wandb_save_qual=False, online_time_idx=None,
                    global_logging=True):
    plot_dir = os.path.join("tracking_plots")
    os.makedirs(plot_dir, exist_ok=True)
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if (torch.isnan(gt_w2c_list[idx]).sum() > 0) or (torch.isinf(gt_w2c_list[idx]).sum() > 0):
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            

            gt_c2w_list = [torch.linalg.inv(x) for x in gt_w2c_list]
            est_c2w_list = [torch.linalg.inv(x) for x in latest_est_w2c_list]
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_c2w_list, est_c2w_list)
            ate_rmse = np.round(ate_rmse, decimals=6)

            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error":iter_pt_error, 
                               f"{stage}/Latest Relative Pose Error":rel_pt_error,
                               f"{stage}/ATE RMSE":ate_rmse}
               

        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'], 
                                                                     transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        # print(data['depth'])
        valid_depth_mask = (data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)
        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        
        if wandb_run is not None:
            wandb_log = {f"{stage}/PSNR": psnr,
                         f"{stage}/Depth RMSE": rmse,
                         f"{stage}/Depth L1": depth_l1,
                         f"{stage}/step": wandb_step}
            if tracking:
                wandb_log = {**wandb_log, **tracking_log}
            wandb_run.log(wandb_log)
        
        if wandb_save_qual and (i % qual_every_i == 0 or i == 1):
            # Silhouette Mask
            presence_sil_mask = presence_sil_mask.detach().cpu().numpy()

            # Log plot to wandb
            if not mapping:
                fig_title = f"Time-Step: {iter_time_idx} | Iter: {i} | Frame: {data['id']}"
            else:
                fig_title = f"Time-Step: {online_time_idx} | Iter: {i} | Frame: {data['id']}"
            plot_rgbd_silhouette(data['im'], data['depth'], im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir=plot_dir, plot_name=f"Frame: {data['id']}", 
                         save_plot=True, wandb_run=wandb_run, wandb_step=wandb_step, 
                                 wandb_title=f"{stage} Qual Viz")
            
        if tracking:
            return iter_pt_error




def concat_params(params_ls):
    params = {}
    for idx in range(len(params_ls)):
        for k, v in params_ls[idx].items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).cuda().float().contiguous()
            else:
                v = v.cuda().float().contiguous()
            if (k not in params) and (k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']):
                params[k] = v
            elif k in params:
                params[k] = torch.cat((params[k], v), dim=0)

    # print(selected_time_idx, selected_time_idx[-1])
    params['cam_unnorm_rots'] = params_ls[-1]['cam_unnorm_rots']
    params['cam_trans'] = params_ls[-1]['cam_trans']
    
    return params

def get_relative_pose(gt_w2c_list, est_w2c_list):
    rel_gt_w2c_list = []
    rel_est_w2c_list = []
    for idx in range(len(gt_w2c_list)-1):
        # rel_gt_w2c = relative_transformation(gt_w2c_list[idx], gt_w2c_list[idx+1])
        # rel_est_w2c = relative_transformation(est_w2c_list[idx], est_w2c_list[idx+1])
        rel_gt_w2c = torch.linalg.inv(gt_w2c_list[idx]) @ gt_w2c_list[idx+1]
        rel_est_w2c = torch.linalg.inv(est_w2c_list[idx]) @ est_w2c_list[idx+1]
        rel_gt_w2c_list.append(rel_gt_w2c)
        rel_est_w2c_list.append(rel_est_w2c)
    return rel_gt_w2c_list, rel_est_w2c_list


def eval(dataset, final_params_ls, num_frames, eval_dir, num_gs_per_frame=None, sil_thres=None, 
         mapping_iters=None, add_new_gaussians=None, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False, unet_model=None, baseframe_corr_list=None, rot=None, trans=None, baseframe_every=10):

    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

    gt_w2c_list = []
    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = dataset[time_idx]

        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
        
        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx) % eval_every != 0:
            continue
        
        use_keyframe_params = True #False
        if use_keyframe_params:
            # concatenate the final params
            base_frame_idx = int(time_idx/baseframe_every) #config['keyframe_every'])
            # final_params_idx = [*range(base_frame_idx*5, (base_frame_idx+1)*5)]
            if baseframe_corr_list is None:
                final_params = final_params_ls[base_frame_idx]
            else:
                if base_frame_idx == 0:
                    final_params = final_params_ls[0]
                else:
                    baseframe_corr_idx = baseframe_corr_list[base_frame_idx-1]
                    final_params_corr_ls = [final_params_ls[int(idx/baseframe_every)] for idx in baseframe_corr_idx]
                    final_params = concat_params(final_params_corr_ls)

            for k, v in final_params.items():
                if isinstance(v, torch.Tensor):
                    final_params[k] = final_params[k].cuda()

            
            # Get current frame Gaussians
            transformed_gaussians = transform_to_frame(final_params, time_idx, 
                                                    gaussians_grad=False, 
                                                    camera_grad=False)
            
            # Define current frame data
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}
            

            # Initialize Render Variables
            rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
            depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                        transformed_gaussians)
        else:
            # Get current frame Gaussians
            transformed_gaussians = transform_to_frame(final_params_ls[time_idx], time_idx, 
                                                       gaussians_grad=False, 
                                                       camera_grad=False)
    
            # Define current frame data
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

            # Initialize Render Variables
            rendervar = transformed_params2rendervar(final_params_ls[time_idx], transformed_gaussians)
            depth_sil_rendervar = transformed_params2depthplussilhouette(final_params_ls[time_idx], curr_data['w2c'],
                                                                         transformed_gaussians)

        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        
      
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        
        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        

        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)


        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")

        for k, v in final_params.items():
            if isinstance(v, torch.Tensor):
                final_params[k] = final_params[k].cpu()
        torch.cuda.empty_cache()

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params_ls[-1]['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if (torch.isnan(gt_w2c_list[idx]).sum() > 0) or (torch.isinf(gt_w2c_list[idx]).sum() > 0):
                continue
            interm_cam_rot = F.normalize(final_params_ls[-1]['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params_ls[-1]['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        gt_c2w_list = [torch.linalg.inv(x) for x in gt_w2c_list]
        est_c2w_list = [torch.linalg.inv(x) for x in latest_est_w2c_list]
        ate_rmse = evaluate_ate(gt_c2w_list, est_c2w_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
        if wandb_run is not None:
            wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                        "Final Stats/step": 1})
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()






def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()

def filter_depth_outliers(depth_map, kernel_size=3, threshold=1.0):
    median_filtered = median_filter(depth_map, size=kernel_size)
    abs_diff = np.abs(depth_map - median_filtered)
    outlier_mask = abs_diff > threshold
    depth_map_filtered = np.where(outlier_mask, median_filtered, depth_map)
    return depth_map_filtered

def filter_depth_outliers_torch(depth_map, kernel_size=3, threshold=1.0):
    mblur = MedianBlur((kernel_size, kernel_size))
    median_filtered = mblur(depth_map)
    abs_diff = torch.abs(depth_map - median_filtered)
    outlier_mask = abs_diff > threshold
    depth_map_filtered = torch.where(outlier_mask, median_filtered, depth_map)
    return depth_map_filtered

def eval_recon(dataset, final_params_ls, num_frames, eval_dir, eval_every=1, baseframe_corr_list=None, baseframe_every=10, first_frame_w2c_abs=None, scene_name=None):
    import time
    color, depth, intrinsics, pose = dataset[0]
    print(pose)
    width, height = color.shape[1], color.shape[0]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, FX, FY, CX, CY)
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


    gt_w2c_list = []
    est_w2c_list = []
    os.makedirs(os.path.join(eval_dir, 'recon'), exist_ok=True)
    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        if first_frame_w2c_abs is not None:
            gt_w2c = gt_w2c @ first_frame_w2c_abs
            gt_w2c = torch2np(gt_w2c)
            gt_w2c_list.append(gt_w2c)


        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

        idx = time_idx
        interm_cam_rot = F.normalize(final_params_ls[-1]['cam_unnorm_rots'][..., idx].detach())
        interm_cam_trans = final_params_ls[-1]['cam_trans'][..., idx].detach()
        intermrel_w2c = torch.eye(4).cuda().float()
        intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
        intermrel_w2c[:3, 3] = interm_cam_trans
        estimate_w2c = torch2np(intermrel_w2c)
        # transform based the absolute pose of the first frame
        if first_frame_w2c_abs is not None:
            estimate_w2c = intermrel_w2c @ first_frame_w2c_abs
            estimate_w2c = torch2np(estimate_w2c)
            est_w2c_list.append(estimate_w2c)

 
        
        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx) % eval_every != 0:
            continue
        
       
        # concatenate the final params
        base_frame_idx = int(time_idx/baseframe_every) 
        if baseframe_corr_list is None:
            final_params = final_params_ls[base_frame_idx]
        else:
            if base_frame_idx == 0:
                final_params = final_params_ls[0]
            else:
                baseframe_corr_idx = baseframe_corr_list[base_frame_idx-1]
                final_params_corr_ls = [final_params_ls[int(idx/baseframe_every)] for idx in baseframe_corr_idx]
                final_params = concat_params(final_params_corr_ls)

        for k, v in final_params.items():
            if isinstance(v, torch.Tensor):
                final_params[k] = final_params[k].cuda()

        
        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(final_params, time_idx, 
                                                gaussians_grad=False, 
                                                camera_grad=False)

        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}


        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                    transformed_gaussians)
    

        # Render Depth & Silhouette
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        

        # Render RGB and Calculate PSNR
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
       
        rendered_color = im.detach()
        rendered_depth = rastered_depth.detach()
    
        
        rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

        rendered_color = (
            torch2np(rendered_color.permute(1, 2, 0)) * 255).astype(np.uint8)
        rendered_depth = torch2np(rendered_depth.permute(1, 2, 0))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
            o3d.geometry.Image(rendered_depth),
            depth_scale=scale,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        
    

        volume.integrate(rgbd, intrinsic, estimate_w2c)

        if time_idx % 50 == 0 or time_idx == num_frames-1:
            mesh = volume.extract_triangle_mesh()
            compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                            scale / 512.0, -2.5 * scale / 512.0)
            mesh = mesh.translate(compensate_vector)
            o3d.io.write_triangle_mesh(
                os.path.join(eval_dir, f"recon/mesh_{time_idx}.ply"), mesh)
         
        
    np.save(os.path.join(eval_dir, 'abs_gt_w2c.npy'), gt_w2c_list)
    np.save(os.path.join(eval_dir, 'abs_est_w2c.npy'), est_w2c_list)

    o3d_mesh = volume.extract_triangle_mesh()
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                            scale / 512.0, -2.5 * scale / 512.0)
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    file_name = os.path.join(eval_dir, "final_mesh.ply")
    o3d.io.write_triangle_mesh(str(file_name), o3d_mesh)
    evaluate_reconstruction(file_name,
                            f"data/Replica-SLAM/cull_replica_mesh/{scene_name}.ply",
                            f"data/Replica-SLAM/cull_replica_mesh/{scene_name}_pc_unseen.npy",
                            eval_dir)