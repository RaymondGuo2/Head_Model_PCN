# Code adapted from Cascaded Point Completion by Yuan et al. (2019)
import numpy as np
import fpsample
import torch
import math
import torch.nn as nn
from object_autocompletion.tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point



# Normalise points in the unit sphere
def unit_sphere_normalisation(post_processed_points):
    center = np.mean(post_processed_points, axis=0)
    centered_points = post_processed_points - center
    radius = np.max(np.linalg.norm(centered_points, axis=1))
    normalised_points = centered_points / radius
    return normalised_points


def symmetric_sample(points, num):
    p1_idx = fpsample.fps_sampling(points, num)
    input_fps = points.gather(1, p1_idx.unsqueeze(-1).expand(-1, -1, points.size(-1)))
    input_fps_flip = torch.cat([input_fps[:, :, [0]], input_fps[:, :, [1]], -input_fps[:, :, [2]]], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


def gen_grid_up(up_ratio):
    sq_root = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sq_root + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break
    grid_x = np.linspace(-0.2, 0.2, num_x)
    grid_y = np.linspace(-0.2, 0.2, num_y)

    x, y = np.meshgrid(grid_x, grid_y)
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)
    return grid


def gen_grid(num_grid_point):
    x = np.linspace(-0.05, 0.05, num_grid_point)
    x, y = np.meshgrid(x, x)
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)
    return grid


def contract_expand_operation(inputs, up_ratio):
    net = inputs
    net = net.reshape(net.size(0), up_ratio, -1, net.size(-1))
    net = net.permute(0, 2, 1, 3)
    net = nn.Sequential(
        nn.Conv2d(net, 64, [1, up_ratio], 1, padding=0),
        nn.ReLU()
    )
    net = nn.Sequential(
        nn.Conv2d(net, 128, [1, 1], 1, padding=0),
        nn.ReLU()
    )
    net = net.view(net.size(0), -1, up_ratio, 64)
    net = nn.Sequential(
        nn.Conv2d(net, 64, [1, 1], 1, padding=0),
        nn.ReLU()
    )
    net = net.view(net.size(0), -1, 64)
    return net


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, scope, use_xyz=True, use_nchw=True):
    p1_idx = fpsample.fps_sampling(xyz, npoint)
    new_xyz = xyz.gather(1, p1_idx.unsqueeze(-1).expand(-1, -1, points.size(-1)))
    new_points_list = []
    for i in range(len(radius_list)):
        radius = radius_list[i]
        nsample = nsample_list[i]
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = group_point(xyz, idx)
        grouped_xy
