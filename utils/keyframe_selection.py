"""
Code for Keyframe Selection based on re-projection of points from 
the current frame to the keyframes.
"""

import torch
import numpy as np
import torch.nn.functional as F

def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
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

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, save_percent=False):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_list):
            # Get the estimated world2cam of the keyframe
            est_w2c = keyframe['est_w2c']
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = edge_value
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum()/projected_pts.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        # selected_keyframe_list = list(np.random.permutation(
        #     np.array(selected_keyframe_list))[:k])
        selected_keyframe_list = list(
            np.array(selected_keyframe_list)[:k])
        
        # pts= pts.cpu()
        # pts4= pts4.cpu()
        # torch.cuda.empty_cache()

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
        torch.cuda.empty_cache()

        if save_percent:
            return list_keyframe
        else:
            return selected_keyframe_list




def keyframe_selection_overlap_visbased(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, earliest_thres=0.5):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        # sampled_indices = valid_depth_indices[indices]
        sampled_indices = valid_depth_indices

        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
        # print('pts:', pts.shape)

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_list):
            # Get the estimated world2cam of the keyframe
            est_w2c = keyframe['est_w2c']
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]


            # Filter out the points that are outside the image
            edge = edge_value
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)


            # Filter out the points that are invisible based on the depth
            curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
            vgrid = projected_pts.reshape(1, 1, -1, 2)
            # normalize to [-1, 1]
            vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
            vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
            depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
            depth_sample = depth_sample.reshape(-1)
            mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
            mask = mask & mask_visible
 

            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum()/projected_pts.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # print('list_keyframe:', list_keyframe)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        earliest_selected_keyframe_list = [keyframe_dict['id']
                                    for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
        # selected_keyframe_list = list(np.random.permutation(
        #     np.array(selected_keyframe_list))[:k])
        # print('selected_keyframe_list:', selected_keyframe_list)
        # print('earliest_selected_keyframe_list:', earliest_selected_keyframe_list)
        selected_keyframe_list = list(
            np.array(selected_keyframe_list)[:k])
        earliest_selected_keyframe_list = list(
            np.array(earliest_selected_keyframe_list)[-1:]) # find the earliest keyframe
        if len(earliest_selected_keyframe_list) == 0:
            earliest_selected_keyframe_list = selected_keyframe_list
        
        # print('selected_keyframe_list:', selected_keyframe_list)
        # print('earliest_selected_keyframe_list:', earliest_selected_keyframe_list)
        print(list_keyframe)
        
        # pts= pts.cpu()
        # pts4= pts4.cpu()
        # torch.cuda.empty_cache()

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
        torch.cuda.empty_cache()

        if save_percent:
            return list_keyframe
        else:
            return selected_keyframe_list, earliest_selected_keyframe_list
        
def keyframe_selection_overlap_visbased_earliest(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, earliest_thres=0.5):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]


        # Filter out the points that are outside the image
        edge = edge_value
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)


        # Filter out the points that are invisible based on the depth
        curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
        vgrid = projected_pts.reshape(1, 1, -1, 2)
        # normalize to [-1, 1]
        vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
        vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
        depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
        depth_sample = depth_sample.reshape(-1)
        mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
        mask = mask & mask_visible


        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    latest_list_keyframe = list_keyframe[-1]
    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # Filter out the keyframes based on the percentage of points that are inside the image
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
    print('list_keyframe_after_percent:', list_keyframe)
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe) == 0:
        list_keyframe = [latest_list_keyframe]
    list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    
    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list


def keyframe_selection_overlap_visbased_earliest_freq(gt_depth, w2c, intrinsics, keyframe_list, k, config, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, earliest_thres=0.5):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]


        # Filter out the points that are outside the image
        edge = edge_value
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)


        # Filter out the points that are invisible based on the depth
        curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
        vgrid = projected_pts.reshape(1, 1, -1, 2)
        # normalize to [-1, 1]
        vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
        vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
        depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
        depth_sample = depth_sample.reshape(-1)
        mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
        mask = mask & mask_visible


        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    latest_list_keyframe = list_keyframe[-1]
    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # Filter out the keyframes based on the percentage of points that are inside the image
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
    print('list_keyframe_after_percent:', list_keyframe)
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe) == 0:
        list_keyframe = [latest_list_keyframe]
    if len(list_keyframe) > 0:
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        bin_edges = np.array(range(0, len(keyframe_list), num_overlap_in_base))
        quantized_list_keyframe = np.digitize(list_keyframe_id, bin_edges, right=False) - 1
        count = np.bincount(quantized_list_keyframe)
        print('count:', count)
        if (count>0).sum() > 3:
            first2base_basedcount = count[count>0][:2]
            count_idx = []
            for i in range(len(count)):
                if count[i] > 0:
                    count_idx.append(i)
                    if len(count_idx) == 2:
                        break
            unique_base, unique_base_idx = np.unique(quantized_list_keyframe, return_index=True)

            if first2base_basedcount[0] >= first2base_basedcount[1]:
                list_keyframe = [list_keyframe[unique_base_idx[0]]]
                print('list_keyframe_earliest:', list_keyframe)
            else:
                
                list_keyframe = [list_keyframe[unique_base_idx[1]]]
                print('list_keyframe_earliest:', list_keyframe)



    

    list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    
    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list


def keyframe_selection_overlap_visbased_earliest_dynamic(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, earliest_thres=0.5, lower_earliest_thres=0.25):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]


        # Filter out the points that are outside the image
        edge = edge_value
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)


        # Filter out the points that are invisible based on the depth
        curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
        vgrid = projected_pts.reshape(1, 1, -1, 2)
        # normalize to [-1, 1]
        vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
        vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
        depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
        depth_sample = depth_sample.reshape(-1)
        mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
        mask = mask & mask_visible


        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    latest_list_keyframe = list_keyframe[-1]
    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # Filter out the keyframes based on the percentage of points that are inside the image
    list_keyframe_first = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
    print('list_keyframe_after_percent:', list_keyframe_first)
    # Sort the keyframes based on id
    list_keyframe_first = sorted(
        list_keyframe_first, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe_first) == 0:
        list_keyframe_second = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > lower_earliest_thres]
        print('list_keyframe_after_percent_dynamic:', list_keyframe_second)
        # Sort the keyframes based on id
        list_keyframe = sorted(
            list_keyframe_second, key=lambda i: i['id'], reverse=False)
        if len(list_keyframe) == 0:
            list_keyframe = [latest_list_keyframe]
    else:
        list_keyframe = list_keyframe_first

    list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    
    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list

def quantize_selected_time_idx(selected_time_idx, num_frames_each_base_frame):
    quantized_selected_time_idx = []
    for idx in selected_time_idx:
        base_frame_idx = int(idx/num_frames_each_base_frame)
        quantized_selected_time_idx.append(base_frame_idx)
    # remove duplicates
    quantized_selected_time_idx = list(set(quantized_selected_time_idx))
    return quantized_selected_time_idx


def keyframe_selection_overlap_visbased_earliest_dynamic_new_topkbase(gt_depth, w2c, intrinsics, keyframe_list, k, config, pixels=1600, edge_value=20, 
                                         kf_depth_thresh=0.01, earliest_thres=0.5, lower_earliest_thres_percent=0.8, topk_base=3):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]


        # Filter out the points that are outside the image
        edge = edge_value
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)


        # Filter out the points that are invisible based on the depth
        curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
        vgrid = projected_pts.reshape(1, 1, -1, 2)
        # normalize to [-1, 1]
        vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
        vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
        depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
        depth_sample = depth_sample.reshape(-1)
        mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
        mask = mask & mask_visible


        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    latest_list_keyframe = list_keyframe[-1]
    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # # Filter out the keyframes based on the percentage of points that are inside the image
    # list_keyframe_first = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
    # print('list_keyframe_after_percent:', list_keyframe_first)
    # # Sort the keyframes based on id
    # list_keyframe_first = sorted(
    #     list_keyframe_first, key=lambda i: i['id'], reverse=False)
    
    iter = 0
    # print('list_keyframe:', list_keyframe)
    while True:
        if iter == 0:
            percent_thres = earliest_thres
        else:
            percent_thres = lower_earliest_thres_percent*percent_thres
        list_keyframe_filtered = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > percent_thres]
        list_keyframe_filtered_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe_filtered]
        # print('list_keyframe_id:', list_keyframe_filtered)
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        quantized_baseframe_id_ls = sorted(quantize_selected_time_idx(list_keyframe_filtered_id, num_overlap_in_base))
        print('quantized_baseframe_id_ls:', quantized_baseframe_id_ls)

        iter += 1
        
        # if len(list_keyframe_filtered) > 0:
        if len(quantized_baseframe_id_ls) >= 3 or (len(list_keyframe) <= 3*num_overlap_in_base and len(quantized_baseframe_id_ls) > 0) or percent_thres < 0.01:
            break
    
    if len(list_keyframe_filtered) == 0:
        list_keyframe_filtered = [latest_list_keyframe]
        
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe_filtered, key=lambda i: i['id'], reverse=False)
    print('list_keyframe_after_sort:', list_keyframe)
    
    if topk_base is None:
        list_keyframe = [list_keyframe[0]]
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        print('list_keyframe_id:', list_keyframe_id)
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        earliest_selected_quantized_baseframe_id_ls = sorted(quantize_selected_time_idx(list_keyframe_id, num_overlap_in_base))
        print('earliest_selected_quantized_baseframe_id_ls:', earliest_selected_quantized_baseframe_id_ls)

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
        torch.cuda.empty_cache()

        return earliest_selected_quantized_baseframe_id_ls
    
    elif topk_base is not None:
        # get earliest k baseframe
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        print('list_keyframe_id:', list_keyframe_id)
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        quantized_baseframe_id_ls = sorted(quantize_selected_time_idx(list_keyframe_id, num_overlap_in_base))
        print('quantized_baseframe_id_ls:', quantized_baseframe_id_ls)
        topk_base = min(topk_base, len(quantized_baseframe_id_ls))
        earliest_selected_quantized_baseframe_id_ls = quantized_baseframe_id_ls[:topk_base]
        print('earliest_selected_quantized_baseframe_id_ls:', earliest_selected_quantized_baseframe_id_ls)

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
        torch.cuda.empty_cache()

        return earliest_selected_quantized_baseframe_id_ls


def keyframe_selection_overlap_visbased_earliest_dynamic_new_topkbase_after2(gt_depth, w2c, intrinsics, keyframe_list, k, config, pixels=1600, edge_value=20, 
                                         kf_depth_thresh=0.01, earliest_thres=0.5, lower_earliest_thres_percent=0.8, topk_base=3):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]


        # Filter out the points that are outside the image
        edge = edge_value
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)


        # Filter out the points that are invisible based on the depth
        curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
        vgrid = projected_pts.reshape(1, 1, -1, 2)
        # normalize to [-1, 1]
        vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
        vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
        depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
        depth_sample = depth_sample.reshape(-1)
        mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
        mask = mask & mask_visible


        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    latest_list_keyframe = list_keyframe[-1]
    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # # Filter out the keyframes based on the percentage of points that are inside the image
    # list_keyframe_first = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
    # print('list_keyframe_after_percent:', list_keyframe_first)
    # # Sort the keyframes based on id
    # list_keyframe_first = sorted(
    #     list_keyframe_first, key=lambda i: i['id'], reverse=False)
    
    iter = 0
    while True:
        if iter == 0:
            percent_thres = earliest_thres
        else:
            percent_thres = lower_earliest_thres_percent*percent_thres
        list_keyframe_filtered = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > percent_thres]
        list_keyframe_filtered_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe_filtered]
        # print('list_keyframe_id:', list_keyframe_id_filtered)
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        quantized_baseframe_id_ls = sorted(quantize_selected_time_idx(list_keyframe_filtered_id, num_overlap_in_base))
        print('quantized_baseframe_id_ls:', quantized_baseframe_id_ls)

        iter += 1
        
        # if len(list_keyframe_filtered) > 0:
        if len(quantized_baseframe_id_ls) >= 3 or (len(list_keyframe) <= 2*num_overlap_in_base and len(quantized_baseframe_id_ls) >= 2):
            break

    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe_filtered, key=lambda i: i['id'], reverse=False)
    print('list_keyframe_after_sort:', list_keyframe)
    
    if topk_base is None:
        list_keyframe = [list_keyframe[0]]
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        print('list_keyframe_id:', list_keyframe_id)
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        earliest_selected_quantized_baseframe_id_ls = sorted(quantize_selected_time_idx(list_keyframe_id, num_overlap_in_base))
        print('earliest_selected_quantized_baseframe_id_ls:', earliest_selected_quantized_baseframe_id_ls)

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
        torch.cuda.empty_cache()

        return earliest_selected_quantized_baseframe_id_ls
    
    elif topk_base is not None:
        # get earliest k baseframe
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        print('list_keyframe_id:', list_keyframe_id)
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        quantized_baseframe_id_ls = sorted(quantize_selected_time_idx(list_keyframe_id, num_overlap_in_base))
        print('quantized_baseframe_id_ls:', quantized_baseframe_id_ls)
        topk_base = min(topk_base, len(quantized_baseframe_id_ls))
        earliest_selected_quantized_baseframe_id_ls = quantized_baseframe_id_ls[:topk_base]
        print('earliest_selected_quantized_baseframe_id_ls:', earliest_selected_quantized_baseframe_id_ls)

        for idx in range(len(keyframe_list)):
            for key, v in keyframe_list[idx].items():
                if isinstance(v, torch.Tensor):
                    keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
        torch.cuda.empty_cache()

        return earliest_selected_quantized_baseframe_id_ls





        


    

def get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=20, kf_depth_thresh=0.01):
    # Get the estimated world2cam of the keyframe
    est_w2c = keyframe['est_w2c']
    # Transform the 3D pointcloud to the keyframe's camera space
    pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    transformed_pts = (est_w2c @ pts4.T).T[:, :3]
    # Project the 3D pointcloud to the keyframe's image space
    points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_z = points_2d[:, 2:] + 1e-5
    points_2d = points_2d / points_z
    projected_pts = points_2d[:, :2]


    # Filter out the points that are outside the image
    edge = edge_value
    mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
        (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
    mask = mask & (points_z[:, 0] > 0)

    # Filter out the points that are invisible based on the depth
    curr_gt_depth = keyframe['depth'].to(projected_pts.device).reshape(1, 1, height, width)
    vgrid = projected_pts.reshape(1, 1, -1, 2)
    # normalize to [-1, 1]
    vgrid[..., 0] = (vgrid[..., 0] / (width-1) * 2.0 - 1.0)
    vgrid[..., 1] = (vgrid[..., 1] / (height-1) * 2.0 - 1.0)
    depth_sample = F.grid_sample(curr_gt_depth, vgrid, padding_mode='zeros', align_corners=True)
    depth_sample = depth_sample.reshape(-1)
    mask_visible = torch.abs(depth_sample - points_z[:, 0]) < kf_depth_thresh * torch.min(depth_sample, points_z[:, 0])
    mask = mask & mask_visible

    # Compute the percentage of points that are inside the image
    percent_inside = mask.sum()/projected_pts.shape[0]

    return percent_inside

# def keyframe_selection_overlap_visbased_doubledir(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
#                                         save_percent=False, kf_depth_thresh=0.01, earliest_thres=0.5, percent_inside_var_thres=0.01):
#         """
#         Select overlapping keyframes to the current camera observation.

#         Args:
#             gt_depth (tensor): ground truth depth image of the current frame.
#             w2c (tensor): world to camera matrix (4 x 4).
#             keyframe_list (list): a list containing info for each keyframe.
#             k (int): number of overlapping keyframes to select.
#             pixels (int, optional): number of pixels to sparsely sample 
#                 from the image of the current camera. Defaults to 1600.
#         Returns:
#             selected_keyframe_list (list): list of selected keyframe id.
#         """
#         # Radomly Sample Pixel Indices from valid depth pixels
#         width, height = gt_depth.shape[2], gt_depth.shape[1]
#         valid_depth_indices = torch.where(gt_depth[0] > 0)
#         valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
#         # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
#         # sampled_indices = valid_depth_indices[indices]
#         sampled_indices = valid_depth_indices

#         # Back Project the selected pixels to 3D Pointcloud
#         pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
#         # print('pts:', pts.shape)

#         for idx in range(len(keyframe_list)):
#             for key, v in keyframe_list[idx].items():
#                 if isinstance(v, torch.Tensor):
#                     keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

#         list_keyframe_c2k = []
#         list_keyframe_k2c = []  
#         for keyframeid, keyframe in enumerate(keyframe_list):
#             percent_inside_c2k = get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
#             list_keyframe_c2k.append(
#                 {'id': keyframeid, 'percent_inside_c2k': percent_inside_c2k})
            
#             # inverse direction
#             curr_keyframe_pts = get_pointcloud(keyframe['depth'], intrinsics, keyframe['est_w2c'], sampled_indices)
#             percent_inside_k2c = get_keyframe_percent_inside({'depth': gt_depth, 'est_w2c': w2c}, intrinsics, curr_keyframe_pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
#             list_keyframe_k2c.append(
#                 {'id': keyframeid, 'percent_inside_k2c': percent_inside_k2c})
            
#         list_keyframe = []
#         for i in range(len(list_keyframe_c2k)):
#             percent_inside_mean = (list_keyframe_c2k[i]['percent_inside_c2k'] + list_keyframe_k2c[i]['percent_inside_k2c'])/2.0
#             percent_inside_var = (list_keyframe_c2k[i]['percent_inside_c2k'] - percent_inside_mean)**2 + (list_keyframe_k2c[i]['percent_inside_k2c'] - percent_inside_mean)**2
#             if percent_inside_var < percent_inside_var_thres:
#                 list_keyframe.append(
#                     {'id': list_keyframe_c2k[i]['id'], 
#                     'percent_inside_mean': percent_inside_mean,
#                     'percent_inside_var': percent_inside_var,
#                     'percent_inside': list_keyframe_c2k[i]['percent_inside_c2k'],}) 
            

#         # Sort the keyframes based on the percentage of points that are inside the image
#         list_keyframe = sorted(
#             list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
#         # print('list_keyframe:', list_keyframe)
#         # Select the keyframes with percentage of points inside the image > 0
#         selected_keyframe_list = [keyframe_dict['id']
#                                   for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
#         earliest_selected_keyframe_list = [keyframe_dict['id']
#                                     for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > earliest_thres]
#         # selected_keyframe_list = list(np.random.permutation(
#         #     np.array(selected_keyframe_list))[:k])
#         # print('selected_keyframe_list:', selected_keyframe_list)
#         # print('earliest_selected_keyframe_list:', earliest_selected_keyframe_list)
#         selected_keyframe_list = list(
#             np.array(selected_keyframe_list)[:k])
#         earliest_selected_keyframe_list = list(
#             np.array(earliest_selected_keyframe_list)[-1:]) # find the earliest keyframe
#         if len(earliest_selected_keyframe_list) == 0:
#             earliest_selected_keyframe_list = selected_keyframe_list
        
#         # print('selected_keyframe_list:', selected_keyframe_list)
#         # print('earliest_selected_keyframe_list:', earliest_selected_keyframe_list)
#         print(list_keyframe)
        
#         # pts= pts.cpu()
#         # pts4= pts4.cpu()
#         # torch.cuda.empty_cache()

#         for idx in range(len(keyframe_list)):
#             for key, v in keyframe_list[idx].items():
#                 if isinstance(v, torch.Tensor):
#                     keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
#         torch.cuda.empty_cache()

#         if save_percent:
#             return list_keyframe
#         else:
#             return selected_keyframe_list, earliest_selected_keyframe_list
        
# def keyframe_selection_overlap_visbased_doubledir(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
#                                         save_percent=False, kf_depth_thresh=0.01, mean_topk=30, var_botk=20, topk=10):
#         """
#         Select overlapping keyframes to the current camera observation.

#         Args:
#             gt_depth (tensor): ground truth depth image of the current frame.
#             w2c (tensor): world to camera matrix (4 x 4).
#             keyframe_list (list): a list containing info for each keyframe.
#             k (int): number of overlapping keyframes to select.
#             pixels (int, optional): number of pixels to sparsely sample 
#                 from the image of the current camera. Defaults to 1600.
#         Returns:
#             selected_keyframe_list (list): list of selected keyframe id.
#         """
#         # Radomly Sample Pixel Indices from valid depth pixels
#         width, height = gt_depth.shape[2], gt_depth.shape[1]
#         valid_depth_indices = torch.where(gt_depth[0] > 0)
#         valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
#         # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
#         # sampled_indices = valid_depth_indices[indices]
#         sampled_indices = valid_depth_indices

#         # Back Project the selected pixels to 3D Pointcloud
#         pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
#         # print('pts:', pts.shape)

#         for idx in range(len(keyframe_list)):
#             for key, v in keyframe_list[idx].items():
#                 if isinstance(v, torch.Tensor):
#                     keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

#         list_keyframe_c2k = []
#         list_keyframe_k2c = []  
#         for keyframeid, keyframe in enumerate(keyframe_list):
#             percent_inside_c2k = get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
#             list_keyframe_c2k.append(
#                 {'id': keyframeid, 'percent_inside_c2k': percent_inside_c2k})
            
#             # inverse direction
#             valid_depth_indices = torch.where(keyframe['depth'][0] > 0)
#             valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
#             sampled_indices = valid_depth_indices
#             curr_keyframe_pts = get_pointcloud(keyframe['depth'], intrinsics, keyframe['est_w2c'], sampled_indices)
#             percent_inside_k2c = get_keyframe_percent_inside({'depth': gt_depth, 'est_w2c': w2c}, intrinsics, curr_keyframe_pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
#             list_keyframe_k2c.append(
#                 {'id': keyframeid, 'percent_inside_k2c': percent_inside_k2c})
            
#         list_keyframe = []
#         for i in range(len(list_keyframe_c2k)):
#             percent_inside_mean = (list_keyframe_c2k[i]['percent_inside_c2k'] + list_keyframe_k2c[i]['percent_inside_k2c'])/2.0
#             percent_inside_var = (list_keyframe_c2k[i]['percent_inside_c2k'] - percent_inside_mean)**2 + (list_keyframe_k2c[i]['percent_inside_k2c'] - percent_inside_mean)**2
#             list_keyframe.append(
#                 {'id': list_keyframe_c2k[i]['id'], 
#                 'percent_inside_mean': percent_inside_mean,
#                 'percent_inside_var': percent_inside_var,
#                 'percent_inside': list_keyframe_c2k[i]['percent_inside_c2k'],}) 
            
#         print('list_keyframe_at_the_beginning:', list_keyframe)
#         # Sort the keyframes based on the mean percentage of points that are inside the image
#         list_keyframe = sorted(
#             list_keyframe, key=lambda i: i['percent_inside_mean'], reverse=True)
#         # Top k keyframes with highest mean percentage
#         mean_topk = min(mean_topk, len(list_keyframe))
#         list_keyframe = list_keyframe[:mean_topk]
#         print('list_keyframe_after_mean:', list_keyframe)
#         # Sort the keyframes based on the variance of the percentage of points that are inside the image
#         list_keyframe = sorted(
#             list_keyframe, key=lambda i: i['percent_inside_var'], reverse=False)
#         # Bottom k keyframes with lowest variance percentage
#         var_botk = min(var_botk, len(list_keyframe))
#         list_keyframe = list_keyframe[:var_botk]
#         print('list_keyframe_after_var:', list_keyframe)
#         # Sort the keyframes based on the percentage of points that are inside the image
#         list_keyframe = sorted(
#             list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
#         topk = min(topk, len(list_keyframe))
#         list_keyframe = list_keyframe[:topk]
#         print('list_keyframe_after_percent:', list_keyframe)
#         # Sort the keyframes based on id
#         # list_keyframe = sorted(
#         #     list_keyframe, key=lambda i: i['id'], reverse=False)
#         list_keyframe = [list_keyframe[0]]
#         print('list_keyframe_earliest:', list_keyframe)
        

#         # Select the keyframes with percentage of points inside the image > 0
#         earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
 
#         for idx in range(len(keyframe_list)):
#             for key, v in keyframe_list[idx].items():
#                 if isinstance(v, torch.Tensor):
#                     keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
#         torch.cuda.empty_cache()

#         if save_percent:
#             return list_keyframe
#         else:
#             return earliest_selected_keyframe_list

def keyframe_selection_overlap_visbased_doubledir_mean_var(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, mean_thresh=0.2, var_thresh=0.01):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe_c2k = []
    list_keyframe_k2c = []  
    for keyframeid, keyframe in enumerate(keyframe_list):
        percent_inside_c2k = get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_c2k.append(
            {'id': keyframeid, 'percent_inside_c2k': percent_inside_c2k})
        
        # inverse direction
        valid_depth_indices = torch.where(keyframe['depth'][0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        sampled_indices = valid_depth_indices
        curr_keyframe_pts = get_pointcloud(keyframe['depth'], intrinsics, keyframe['est_w2c'], sampled_indices)
        percent_inside_k2c = get_keyframe_percent_inside({'depth': gt_depth, 'est_w2c': w2c}, intrinsics, curr_keyframe_pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_k2c.append(
            {'id': keyframeid, 'percent_inside_k2c': percent_inside_k2c})
        
    list_keyframe = []
    for i in range(len(list_keyframe_c2k)):
        percent_inside_mean = (list_keyframe_c2k[i]['percent_inside_c2k'] + list_keyframe_k2c[i]['percent_inside_k2c'])/2.0
        percent_inside_var = (list_keyframe_c2k[i]['percent_inside_c2k'] - percent_inside_mean)**2 + (list_keyframe_k2c[i]['percent_inside_k2c'] - percent_inside_mean)**2
        list_keyframe.append(
            {'id': list_keyframe_c2k[i]['id'], 
            'percent_inside_mean': percent_inside_mean,
            'percent_inside_var': percent_inside_var,
            'percent_inside': list_keyframe_c2k[i]['percent_inside_c2k'],}) 
        
    latest_list_keyframe = list_keyframe[-1]
    
    print('list_keyframe_at_the_beginning:', list_keyframe)
    # Filter out the keyframes based on the mean    
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside_mean'] > mean_thresh]
    print('list_keyframe_after_mean:', list_keyframe)
    # Filter out the keyframes based on the variance
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside_var'] < var_thresh]
    print('list_keyframe_after_var:', list_keyframe)
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe) == 0:
        list_keyframe = [latest_list_keyframe]
    list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    

    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list

def keyframe_selection_overlap_visbased_doubledir_earlybaseframe(gt_depth, w2c, intrinsics, keyframe_list, k, config, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, earliest_thres_c2k=0.5, earliest_thres_k2c=0.5):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe_c2k = []
    list_keyframe_k2c = []  
    for keyframeid, keyframe in enumerate(keyframe_list):
        percent_inside_c2k = get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_c2k.append(
            {'id': keyframeid, 'percent_inside_c2k': percent_inside_c2k})
        
        # inverse direction
        valid_depth_indices = torch.where(keyframe['depth'][0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        sampled_indices = valid_depth_indices
        curr_keyframe_pts = get_pointcloud(keyframe['depth'], intrinsics, keyframe['est_w2c'], sampled_indices)
        percent_inside_k2c = get_keyframe_percent_inside({'depth': gt_depth, 'est_w2c': w2c}, intrinsics, curr_keyframe_pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_k2c.append(
            {'id': keyframeid, 'percent_inside_k2c': percent_inside_k2c})
        
    list_keyframe = []
    for i in range(len(list_keyframe_c2k)):
        # percent_inside_mean = (list_keyframe_c2k[i]['percent_inside_c2k'] + list_keyframe_k2c[i]['percent_inside_k2c'])/2.0
        # percent_inside_var = (list_keyframe_c2k[i]['percent_inside_c2k'] - percent_inside_mean)**2 + (list_keyframe_k2c[i]['percent_inside_k2c'] - percent_inside_mean)**2
        list_keyframe.append(
            {'id': list_keyframe_c2k[i]['id'], 
            'percent_inside_c2k': list_keyframe_c2k[i]['percent_inside_c2k'], 
            'percent_inside_k2c': list_keyframe_k2c[i]['percent_inside_k2c'],})
        
    latest_list_keyframe = list_keyframe[-1]
    
    print('list_keyframe_at_the_beginning:', list_keyframe)
    # Filter out the keyframes based on the percentage of points that are inside the image
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside_c2k'] > earliest_thres_c2k]
    print('list_keyframe_after_percent:', list_keyframe)
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe) == 0:
        list_keyframe = [latest_list_keyframe]
    else:
        # Split the list_keyframe 
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        bin_edges = np.array(range(0, len(keyframe_list), num_overlap_in_base))
        quantized_list_keyframe = np.digitize(list_keyframe_id, bin_edges, right=False)
        unique_base, unique_base_idx = np.unique(quantized_list_keyframe, return_index=True)
        if len(unique_base) >=3:
            value1 = []
            value2 = []
            value3 = []
            for keyframe_id, keyframe in enumerate(list_keyframe):
                if quantized_list_keyframe[keyframe_id] == unique_base[0]:
                    value1.append(keyframe['percent_inside_k2c'])
                elif quantized_list_keyframe[keyframe_id] == unique_base[1]:
                    value2.append(keyframe['percent_inside_k2c'])
                elif quantized_list_keyframe[keyframe_id] == unique_base[2]:
                    value3.append(keyframe['percent_inside_k2c'])
                else:
                    break
          
            max_value_ls = [max(value1), max(value2), max(value3)]
            # filter out the max value based on the thres
            max_value_ls_filtered = [max_value for max_value in max_value_ls if max_value > earliest_thres_k2c]
            if len(max_value_ls_filtered) == 0:
                max_value_idx = max_value_ls.index(max(max_value_ls))
                list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
            else:
                max_value_idx = max_value_ls.index(max_value_ls_filtered[0])
                list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
        elif len(unique_base) == 2:
            value1 = []
            value2 = []
            for keyframe_id, keyframe in enumerate(list_keyframe):
                if quantized_list_keyframe[keyframe_id] == unique_base[0]:
                    value1.append(keyframe['percent_inside_k2c'])
                elif quantized_list_keyframe[keyframe_id] == unique_base[1]:
                    value2.append(keyframe['percent_inside_k2c'])
                else:
                    break
            max_value_ls = [max(value1), max(value2)]
            # filter out the max value based on the thres
            max_value_ls_filtered = [max_value for max_value in max_value_ls if max_value > earliest_thres_k2c]
            if len(max_value_ls_filtered) == 0:
                max_value_idx = max_value_ls.index(max(max_value_ls))
                list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
            else:
                max_value_idx = max_value_ls.index(max_value_ls_filtered[0])
                list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
        else:
            list_keyframe = [list_keyframe[0]]
                
        
    
    # list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    

    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list
    

def keyframe_selection_overlap_visbased_doubledir_earlybaseframe_dynamic(gt_depth, w2c, intrinsics, keyframe_list, k, config, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, earliest_thres_c2k=0.5, earliest_thres_k2c=0.5, lower_earliest_thres_k2c=0.2):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe_c2k = []
    list_keyframe_k2c = []  
    for keyframeid, keyframe in enumerate(keyframe_list):
        percent_inside_c2k = get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_c2k.append(
            {'id': keyframeid, 'percent_inside_c2k': percent_inside_c2k})
        
        # inverse direction
        valid_depth_indices = torch.where(keyframe['depth'][0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        sampled_indices = valid_depth_indices
        curr_keyframe_pts = get_pointcloud(keyframe['depth'], intrinsics, keyframe['est_w2c'], sampled_indices)
        percent_inside_k2c = get_keyframe_percent_inside({'depth': gt_depth, 'est_w2c': w2c}, intrinsics, curr_keyframe_pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_k2c.append(
            {'id': keyframeid, 'percent_inside_k2c': percent_inside_k2c})
        
    list_keyframe = []
    for i in range(len(list_keyframe_c2k)):
        # percent_inside_mean = (list_keyframe_c2k[i]['percent_inside_c2k'] + list_keyframe_k2c[i]['percent_inside_k2c'])/2.0
        # percent_inside_var = (list_keyframe_c2k[i]['percent_inside_c2k'] - percent_inside_mean)**2 + (list_keyframe_k2c[i]['percent_inside_k2c'] - percent_inside_mean)**2
        list_keyframe.append(
            {'id': list_keyframe_c2k[i]['id'], 
            'percent_inside_c2k': list_keyframe_c2k[i]['percent_inside_c2k'], 
            'percent_inside_k2c': list_keyframe_k2c[i]['percent_inside_k2c'],})
        
    latest_list_keyframe = list_keyframe[-1]
    
    print('list_keyframe_at_the_beginning:', list_keyframe)
    # Filter out the keyframes based on the percentage of points that are inside the image
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside_c2k'] > earliest_thres_c2k]
    print('list_keyframe_after_percent:', list_keyframe)
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe) == 0:
        list_keyframe = [latest_list_keyframe]
    else:
        # Split the list_keyframe 
        list_keyframe_id = [keyframe_dict['id'] for keyframe_dict in list_keyframe]
        num_overlap_in_base = int(config['baseframe_every'] / config['overlap_every'])
        bin_edges = np.array(range(0, len(keyframe_list), num_overlap_in_base))
        quantized_list_keyframe = np.digitize(list_keyframe_id, bin_edges, right=False)
        unique_base, unique_base_idx = np.unique(quantized_list_keyframe, return_index=True)
        if len(unique_base) >=3:
            value1 = []
            value2 = []
            value3 = []
            for keyframe_id, keyframe in enumerate(list_keyframe):
                if quantized_list_keyframe[keyframe_id] == unique_base[0]:
                    value1.append(keyframe['percent_inside_k2c'])
                elif quantized_list_keyframe[keyframe_id] == unique_base[1]:
                    value2.append(keyframe['percent_inside_k2c'])
                elif quantized_list_keyframe[keyframe_id] == unique_base[2]:
                    value3.append(keyframe['percent_inside_k2c'])
                else:
                    break
          
            max_value_ls = [max(value1), max(value2), max(value3)]
            # filter out the max value based on the thres
            max_value_ls_filtered = [max_value for max_value in max_value_ls if max_value > earliest_thres_k2c]
            if len(max_value_ls_filtered) == 0:
                max_value_ls_filtered_lower = [max_value for max_value in max_value_ls if max_value > lower_earliest_thres_k2c]
                if len(max_value_ls_filtered_lower) == 0:
                    max_value_idx = max_value_ls.index(max(max_value_ls))
                    list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
                else:
                    max_value_idx = max_value_ls.index(max_value_ls_filtered_lower[0])
                    list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
            else:
                max_value_idx = max_value_ls.index(max_value_ls_filtered[0])
                list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
        elif len(unique_base) == 2:
            value1 = []
            value2 = []
            for keyframe_id, keyframe in enumerate(list_keyframe):
                if quantized_list_keyframe[keyframe_id] == unique_base[0]:
                    value1.append(keyframe['percent_inside_k2c'])
                elif quantized_list_keyframe[keyframe_id] == unique_base[1]:
                    value2.append(keyframe['percent_inside_k2c'])
                else:
                    break
            max_value_ls = [max(value1), max(value2)]
            # filter out the max value based on the thres
            max_value_ls_filtered = [max_value for max_value in max_value_ls if max_value > earliest_thres_k2c]
            if len(max_value_ls_filtered) == 0:
                max_value_ls_filtered_lower = [max_value for max_value in max_value_ls if max_value > lower_earliest_thres_k2c]
                if len(max_value_ls_filtered_lower) == 0:
                    max_value_idx = max_value_ls.index(max(max_value_ls))
                    list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
                else:
                    max_value_idx = max_value_ls.index(max_value_ls_filtered_lower[0])
                    list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
            else:
                max_value_idx = max_value_ls.index(max_value_ls_filtered[0])
                list_keyframe = [list_keyframe[unique_base_idx[max_value_idx]]]
        else:
            list_keyframe = [list_keyframe[0]]
                
        
    
    # list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    

    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list
    

def keyframe_selection_overlap_visbased_doubledir_mean(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600, edge_value=20, 
                                        save_percent=False, kf_depth_thresh=0.01, mean_thresh=0.2):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    # indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    # sampled_indices = valid_depth_indices[indices]
    sampled_indices = valid_depth_indices

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
    # print('pts:', pts.shape)

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cuda()

    list_keyframe_c2k = []
    list_keyframe_k2c = []  
    for keyframeid, keyframe in enumerate(keyframe_list):
        percent_inside_c2k = get_keyframe_percent_inside(keyframe, intrinsics, pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_c2k.append(
            {'id': keyframeid, 'percent_inside_c2k': percent_inside_c2k})
        
        # inverse direction
        valid_depth_indices = torch.where(keyframe['depth'][0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        sampled_indices = valid_depth_indices
        curr_keyframe_pts = get_pointcloud(keyframe['depth'], intrinsics, keyframe['est_w2c'], sampled_indices)
        percent_inside_k2c = get_keyframe_percent_inside({'depth': gt_depth, 'est_w2c': w2c}, intrinsics, curr_keyframe_pts, width, height, edge_value=edge_value, kf_depth_thresh=kf_depth_thresh)
        list_keyframe_k2c.append(
            {'id': keyframeid, 'percent_inside_k2c': percent_inside_k2c})
        
    list_keyframe = []
    for i in range(len(list_keyframe_c2k)):
        percent_inside_mean = (list_keyframe_c2k[i]['percent_inside_c2k'] + list_keyframe_k2c[i]['percent_inside_k2c'])/2.0
        percent_inside_var = (list_keyframe_c2k[i]['percent_inside_c2k'] - percent_inside_mean)**2 + (list_keyframe_k2c[i]['percent_inside_k2c'] - percent_inside_mean)**2
        list_keyframe.append(
            {'id': list_keyframe_c2k[i]['id'], 
            'percent_inside_mean': percent_inside_mean,
            'percent_inside_var': percent_inside_var,
            'percent_inside': list_keyframe_c2k[i]['percent_inside_c2k'],}) 
    latest_list_keyframe = list_keyframe[-1]
    
    print('list_keyframe_at_the_beginning:', list_keyframe)
    # Filter out the keyframes based on the mean    
    list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside_mean'] > mean_thresh]
    print('list_keyframe_after_mean:', list_keyframe)
    # Filter out the keyframes based on the variance
    # list_keyframe = [keyframe_dict for keyframe_dict in list_keyframe if keyframe_dict['percent_inside_var'] < var_thresh]
    # print('list_keyframe_after_var:', list_keyframe)
    # Sort the keyframes based on id
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['id'], reverse=False)
    if len(list_keyframe) == 0:
        list_keyframe = [latest_list_keyframe]
    list_keyframe = [list_keyframe[0]]
    print('list_keyframe_earliest:', list_keyframe)
    

    # Select the keyframes with percentage of points inside the image > 0
    earliest_selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe]

    for idx in range(len(keyframe_list)):
        for key, v in keyframe_list[idx].items():
            if isinstance(v, torch.Tensor):
                keyframe_list[idx][key] = keyframe_list[idx][key].cpu()
    torch.cuda.empty_cache()

    if save_percent:
        return list_keyframe
    else:
        return earliest_selected_keyframe_list
    

def find_earliest_keyframe(corr_list, gt_depth, w2c, intrinsics, keyframe_list, k, edge_value, baseframe_every, threshold, pixels=1600):
    """
    Find the earliest keyframe from the list of corr keyframes.

    Args:
        corr_list (list): list of corr keyframes. [keyframe, latest, current]

    Returns:
        earliest_keyframe (int): index of the earliest keyframe.
    """
    print('corr_list:', corr_list)
    corr_list = corr_list[::-1]
    current_frame_idx = corr_list[0][2]
    current_keyframe_idx = corr_list[0][0]
    # print('keyframe_list:', keyframe_list)

    earliest_keyframe = current_keyframe_idx
    while current_keyframe_idx >= 0:
        print('current_keyframe_idx_beforenext:', current_keyframe_idx)
        # Find the previous keyframe
        current_keyframe_idx = next(
            (i for i, _, x in corr_list if x == current_keyframe_idx), -100)
        print('current_keyframe_idx:', current_keyframe_idx)
        if current_keyframe_idx >= 0:
            # compute the percentage of points that are inside the image
            list_keyframe = keyframe_selection_overlap(gt_depth, w2c, intrinsics, [keyframe_list[int(current_keyframe_idx/baseframe_every)]], k, edge_value=edge_value, save_percent=True, pixels=pixels)
            print('list_keyframe:', list_keyframe)
            if list_keyframe[0]['percent_inside'].item() > threshold:
                earliest_keyframe = current_keyframe_idx
            else:
                break

    return [earliest_keyframe, None, current_frame_idx]



def find_earliest_keyframe_visbased(corr_list, gt_depth, w2c, intrinsics, keyframe_list, k, edge_value, baseframe_every, threshold, kf_depth_thresh=0.01, pixels=1600):
    """
    Find the earliest keyframe from the list of corr keyframes.

    Args:
        corr_list (list): list of corr keyframes. [keyframe, latest, current]

    Returns:
        earliest_keyframe (int): index of the earliest keyframe.
    """
    print('corr_list:', corr_list)
    corr_list = corr_list[::-1]
    current_frame_idx = corr_list[0][2]
    current_keyframe_idx = corr_list[0][0]
    # print('keyframe_list:', keyframe_list)

    earliest_keyframe = current_keyframe_idx
    while current_keyframe_idx >= 0:
        print('current_keyframe_idx_beforenext:', current_keyframe_idx)
        # Find the previous keyframe
        current_keyframe_idx = next(
            (i for i, _, x in corr_list if x == current_keyframe_idx), -100)
        print('current_keyframe_idx:', current_keyframe_idx)
        if current_keyframe_idx >= 0:
            # compute the percentage of points that are inside the image
            list_keyframe = keyframe_selection_overlap_visbased(gt_depth, w2c, intrinsics, [keyframe_list[int(current_keyframe_idx/baseframe_every)]], 
                                                                k, edge_value=edge_value, save_percent=True, kf_depth_thresh=kf_depth_thresh, pixels=pixels)
            print('list_keyframe:', list_keyframe)
            if list_keyframe[0]['percent_inside'].item() > threshold:
                earliest_keyframe = current_keyframe_idx
            else:
                break

    return [earliest_keyframe, None, current_frame_idx]


def find_earliest_keyframe_addaptivebase(corr_list, gt_depth, w2c, intrinsics, keyframe_list, k, edge_value, baseframe_idx_ls, threshold):
    """
    Find the earliest keyframe from the list of corr keyframes.

    Args:
        corr_list (list): list of corr keyframes. [keyframe, latest, current]

    Returns:
        earliest_keyframe (int): index of the earliest keyframe.
    """
    print('corr_list:', corr_list)
    corr_list = corr_list[::-1]
    current_frame_idx = corr_list[0][2]
    current_keyframe_idx = corr_list[0][0]
    # print('keyframe_list:', keyframe_list)

    earliest_keyframe = current_keyframe_idx
    while current_keyframe_idx >= 0:
        print('current_keyframe_idx_beforenext:', current_keyframe_idx)
        # Find the previous keyframe
        current_keyframe_idx = next(
            (i for i, _, x in corr_list if x == current_keyframe_idx), -100)
        print('current_keyframe_idx:', current_keyframe_idx)
        if current_keyframe_idx >= 0:
            # compute the percentage of points that are inside the image
            keyframe_list_idx = baseframe_idx_ls.index(current_keyframe_idx)
            list_keyframe = keyframe_selection_overlap(gt_depth, w2c, intrinsics, [keyframe_list[keyframe_list_idx]], k, edge_value=edge_value, save_percent=True)
            print('list_keyframe:', list_keyframe)
            if list_keyframe[0]['percent_inside'].item() > threshold:
                earliest_keyframe = current_keyframe_idx
            else:
                break

    return [earliest_keyframe, None, current_frame_idx]

