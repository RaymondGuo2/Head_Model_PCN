# Code adapted from Cascaded Point Completion by Yuan et al. (2019)
import numpy as np
import fpsample
import torch
import math
import torch.nn as nn
from torch_geometric.nn import fps
# from object_autocompletion.tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
# from pytorch3d import masked_gather


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
    points_fps = points.view(batch_size * num_points, channels)  # (B x 2048) x 3
    batch = torch.arange(batch_size).view(-1, 1).repeat(1, num_points).view(-1)  # Reshape to 1D tensor for all batches and num_points so (B x num_points)
    p1_idx = fps(points_fps, batch, ratio=num/num_points)  # Output is (B x num)
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
    grid_x = np.linspace(-0.2, 0.2, num_x)  # linearly spaced num_x points between -0.2 and 0.2
    grid_y = np.linspace(-0.2, 0.2, num_y)  # linearly spaced num_y points between -0.2 and 0.2

    x, y = np.meshgrid(grid_x, grid_y)  # 2D coordinate matrices
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)  # (num_x * num_y, 2) so [2, 2, 2] -> [4, 2]
    return grid


def gen_grid(num_grid_point):
    x = np.linspace(-0.05, 0.05, num_grid_point)
    x, y = np.meshgrid(x, x)
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)
    return grid


# Adaptation from Cascaded Point Completion
def conv2d(inputs, num_output_channels, kernel_size, stride=(1,1), activation_fn=nn.LeakyReLU()):
    layers = []
    conv = nn.Conv2d(in_channels=inputs.shape[-1], out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
    layers.append(conv)
    if activation_fn:
        layers.append(activation_fn())

    return nn.Sequential(*layers)


def contract_expand_operation(inputs, up_ratio):
    batch_size, channels, num_points = inputs.shape  # (B, 64, 2048)
    net = inputs.view(batch_size, up_ratio, num_points // up_ratio , channels)  # (B, 2, 1024, 64)
    net = net.permute(0, 2, 1, 3)  # (B, 1024, 2, 64)

    conv1 = conv2d(net, 64, (1, up_ratio), activation_fn=nn.ReLU())
    net = conv1(net)

    conv2 = conv2d(net, 128, (1, 1), activation_fn=nn.ReLU())
    net = conv2(net)

    net = net.view(batch_size, -1, up_ratio, 64)
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