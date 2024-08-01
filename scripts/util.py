# Code adapted from Cascaded Point Completion by Yuan et al. (2019)
import numpy as np
import sys
import os
import torch
import math
import torch.nn as nn
import torch_cluster
from torch_geometric.nn import fps

# Normalise points in the unit sphere
def unit_sphere_normalisation(post_processed_points):
    center = np.mean(post_processed_points, axis=0)
    centered_points = post_processed_points - center
    radius = np.max(np.linalg.norm(centered_points, axis=1))
    normalised_points = centered_points / radius
    return normalised_points


def symmetric_sample(points, num):
    # Permute since tensor size previously was (B x 2048 x 3) but need (B x 3 x 2048)
    # Be wary that maybe you don't want to permute the tensor yet in case the function takes the input differently
    # points = points.permute(0, 2, 1)

    batch_size, num_points, channels = points.shape  # (B x 2048 x 3)
    points_fps = points.view(-1, channels)  # (B x 2048) x 3
    batch = torch.arange(batch_size).repeat_interleave(num_points)  # Reshape to 1D tensor for all batches and num_points so (B x num_points)
    p1_idx = fps(points_fps, batch, ratio=num/num_points)  # Output is (B x num)
    p1_idx = p1_idx.view(batch_size, -1) % num_points
    input_fps = masked_gather(points, p1_idx)  # Output is (B x num x 3)
    # This function flips the z dimension of the model to achieve symmetry, essentially mirroring sides (tensor shape therefore remains the same)
    input_fps_flip = torch.cat(
        [input_fps[:, :, [0]],
         input_fps[:, :, [1]],
         -input_fps[:, :, [2]]], dim=2
    )  # Select the x, y, and z points of the fps sampled points and concatenate
    # Concatenate along the point dimension
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps  # (B x (2 * num) x 3)


def gen_grid_up(up_ratio):
    sq_root = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sq_root + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break
    grid_x = torch.tensor(torch.linspace(-0.2, 0.2, num_x))  # linearly spaced num_x points between -0.2 and 0.2
    grid_y = torch.tensor(torch.linspace(-0.2, 0.2, num_y))  # linearly spaced num_y points between -0.2 and 0.2

    x, y = torch.meshgrid(grid_x, grid_y)  # 2D coordinate matrices
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)  # (num_x * num_y, 2) so [2, 2, 2] -> [4, 2]
    return grid


def gen_grid(num_grid_point):
    x = np.linspace(-0.05, 0.05, num_grid_point)
    x, y = np.meshgrid(x, x)
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)
    return grid


# Adaptation from Cascaded Point Completion
def conv2d(inputs, num_output_channels, kernel_size, stride=(1,1), activation_fn=None):
    layers = []
    conv = nn.Conv2d(in_channels=inputs.shape[1], out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
    layers.append(conv)
    if activation_fn:
        layers.append(activation_fn)

    return nn.Sequential(*layers)


def contract_expand_operation(inputs, up_ratio):
    batch_size, channels, num_points = inputs.shape  # (B, 64, 2048)
    # net = inputs.view(batch_size, up_ratio, num_points // up_ratio , channels)  # (B, 2, 1024, 64)
    net = inputs.view(batch_size, channels, up_ratio, num_points // up_ratio)
    # net = net.permute(0, 2, 1, 3)  # (B, 1024, 2, 64)
    # net = net.permute(0, 1, 3, 2)

    conv1 = conv2d(net, 64, (1, 1), activation_fn=nn.ReLU())
    net = conv1(net)

    conv2 = conv2d(net, 128, (1, 1), activation_fn=nn.ReLU())
    net = conv2(net)

    # net = net.view(batch_size, -1, up_ratio, 64)
    conv3 = conv2d(net, 64, (1, 1), activation_fn=nn.ReLU())
    net = conv3(net)
    net = net.view(batch_size, 64, 2048)
    return net


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, use_xyz=True, use_nchw=True):
    batch_size, num_points, channels = xyz.shape  # (B x 2048 x 3)
    xyz_fps = xyz.view(batch_size * num_points, channels)  # (B x 2048) x 3
    batch = torch.arange(batch_size).view(-1, 1).repeat(1, num_points).view(-1)  # Reshape to 1D tensor for all batches and num_points so (B x num_points)
    p1_idx = fps(xyz_fps, batch, ratio=npoint/num_points)  # Output is (B x num)
    new_xyz = masked_gather(xyz, p1_idx)
    new_points_list = []

    for i in range(len(radius_list)):
        radius = radius_list[i]
        nsample = nsample_list[i]
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = group_point(xyz, idx)
        grouped_xyz -= torch.tile(new_xyz.unsqueeze(2), (1, 1, nsample, 1))
        if points is not None:
            grouped_points = group_point(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        else:
            grouped_points = grouped_xyz
        if use_nchw:
            grouped_points = grouped_points.permute(0, 3, 1, 2)
        for j, num_out_channel in enumerate(mlp_list[i]):
            grouped_points = conv2d(grouped_points, num_out_channel, kernel_size=(1, 1), stride=(1, 1), activation_fn=nn.LeakyReLU())
        if use_nchw:
            grouped_points = grouped_points.permute(0, 2, 3, 1)
        new_points, _ = torch.max(grouped_points, dim=2)
        new_points_list.append(new_points)
    new_points_concat = torch.cat(new_points_list, dim=-1)
    return new_xyz, new_points_concat


# GPT Function (Investigate pytorch3d function) - will attempt to optimise with C++
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Args:
        radius: float, radius of the ball query
        nsample: int, maximum number of points to gather in the ball
        xyz: (batch_size, ndataset, 3) float32 array, input points
        new_xyz: (batch_size, npoint, 3) float32 array, query points

    Returns:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Compute squared distance between each point in xyz and new_xyz
    dists = torch.cdist(new_xyz, xyz, p=2)  # (B, S, N)

    # Find points within the specified radius
    mask = dists <= radius  # (B, S, N)

    # Initialize output arrays
    idx = torch.zeros((B, S, nsample), dtype=torch.int32, device=xyz.device)
    pts_cnt = torch.zeros((B, S), dtype=torch.int32, device=xyz.device)

    for b in range(B):
        for s in range(S):
            valid_indices = torch.nonzero(mask[b, s])[:, 0]
            if valid_indices.numel() == 0:
                continue

            if valid_indices.numel() > nsample:
                # Randomly sample nsample points
                selected_indices = valid_indices[torch.randperm(valid_indices.numel())[:nsample]]
            else:
                # Pad the remaining with the last valid index
                selected_indices = torch.cat([
                    valid_indices,
                    valid_indices[-1].repeat(nsample - valid_indices.numel())
                ])

            idx[b, s, :selected_indices.numel()] = selected_indices
            pts_cnt[b, s] = valid_indices.numel()

    return idx, pts_cnt

# GPT Function
def group_point(points, idx):
    """
    Args:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points

    Returns:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    """
    B, N, C = points.shape
    _, S, K = idx.shape

    # Expand idx to have the same number of dimensions as points
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, S, K, C)

    # Gather points along the 1st dimension (ndataset)
    grouped_points = torch.gather(points.unsqueeze(1).expand(-1, S, -1, -1), 2, idx_expanded)  # (B, S, K, C)

    return grouped_points

# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/utils.html

def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points