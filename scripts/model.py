# Code taken from Cascaded Point Completion by Yuan et al. (2019)

import torch
import torch.nn as nn
import util
import fpsample
import math
from pytorch3d import masked_gather
from torch_geometric.nn import fps



class Encoder(nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=1),  # 3 in_channels to represent the xyz coordinates
            nn.Conv1d(128, 256, kernel_size=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(256 + 256, 512, kernel_size=1),
            nn.Conv1d(512, embed_size, kernel_size=1)
        )

    def forward(self, out):
        # Initial tensor shape: (B, 2048, 3)
        # Permute since nn.Conv1d expects B x num_channels x num_points
        out = out.permute(0, 2, 1)  # (B, 3, 2048)
        features = self.layer1(out)  # (B, 256, 2048)
        features_global, _ = torch.max(features, dim=2, keepdim=True)  # (B, 256, 1)
        features_global_tiled = features_global.repeat(1, 1, out.size(2))  # (B, 256, 2048)
        features = torch.cat((features, features_global_tiled), dim=1)  # (B, 512, 2048)
        features = self.layer2(features)  # (B, embed_size, 2048)
        features, _ = torch.max(features, dim=2, keepdim=False)  # (B, embed_size, 1)
        return features  # (B, embed_size)


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, mean_feature_dim=1024):
        super().__init__()
        self.level0 = nn.Sequential(
            nn.Linear(latent_dim, 1024),  # (B x 1024) -> (B x 1024)
            nn.Linear(1024, 1024),  # (B x 1024) -> (B x 1024)
            nn.Linear(1024, 512 * 3),  # (B x 1024) -> (B x 1536)
            nn.Tanh()  # (B x 1536) -> (B x 1536)
        )
        self.mean_feature_layer = nn.Sequential(
            nn.Linear(mean_feature_dim, 128),
            nn.ReLU()
        )
        input_channels = 2 + 3 + latent_dim + 128
        self.feat1_layer = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=1),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        self.fine_layer = nn.Sequential(
            nn.Conv1d(64, 512, kernel_size=1),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.Conv1d(512, 3, kernel_size=1),
        )

    def forward(self, latent_code, partial_inputs, step_ratio, num_extract=512, mean_feature=None):
        # latent_code taken from the encoder is in the size of embed_size (since default is 1024), then self.level0 takes 1024 as its first dimension
        level0 = self.level0(latent_code)  # (B x 1024) -> (B x 1536)
        level0 = level0.view(-1, 512, 3)  # (B x 1536) -> (B x 512 x 3)
        # Coarse reconstruction completed
        coarse = level0  # (B x 512 x 3)
        # partial_inputs is still (B x 2048 x 3)
        input_fps = util.symmetric_sample(partial_inputs, int(num_extract/2))  # (B x num_extract x 3)
        level0 = torch.cat([input_fps, level0], dim=1)  # (B x (num_extract + 512) x 3) or in this case (B x 1024 x 3)

        # Point subsampling
        if self.num_extract > 512:
            b, n, c = level0.shape
            level0_fps = level0.view(b * n, c)
            b_reshape = torch.arange(b).view(-1, 1).repeat(1, n).view(-1)
            fps_num = fps(level0_fps, b_reshape, ratio=1024/n)
            level0 = masked_gather(level0, fps_num)

        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = util.gen_grid_up(2 ** (i+1))
            grid = grid.unsqueeze(0)
            grid_feat = grid.tile((level0.shape[0], 1024, 1))
            point_feat = level0.unsqueeze(2).tile((1, 1, 2, 1))
            point_feat = point_feat.view(-1, num_fine, 3)
            global_feat = self.latent_input.unsqueeze(1).tile((1, num_fine, 1))

            mean_feature_use = self.mean_feature_layer()
            mean_feature_use = mean_feature_use.unsqueeze(1)
            mean_feature_use = mean_feature_use.tile((1, num_fine, 1))
            feat = torch.concat((grid_feat, point_feat, global_feat, mean_feature_use), dim=2)
            feat1 = self.feat1_layer(feat)
            feat2 = util.contract_expand_operation(feat1, 2)
            feat = feat1 + feat2
            fine = self.fine_layer(feat) + point_feat
            level0 = fine

        return coarse, fine


class Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_pl, step_ratio, num_extract=512, mean_feature=None):
        features_partial = self.encoder(input_pl)
        coarse, fine = self.decoder(features_partial, input_pl, step_ratio, num_extract, mean_feature)
        return coarse, fine


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pcd, divide_ratio=1):
        l0_xyz = pcd
        l0_points = None
        num_point = pcd.shape[1].value
        l1_xyz, l1_points = util.pointnet_sa_module_msg(l0_xyz, l0_points, int(num_point/8), [0.1, 0.2, 0.4], [16, 32, 128],
                                                        [[32//divide_ratio, 32//divide_ratio, 64//divide_ratio],
                                                         [64//divide_ratio, 64//divide_ratio, 128//divide_ratio],
                                                         [64//divide_ratio, 96//divide_ratio, 128//divide_ratio]],
                                                        scope='layer1', use_nchw=False)
        patch_values = nn.Conv1d(l1_points,1, kernel_size=1)
        return patch_values








