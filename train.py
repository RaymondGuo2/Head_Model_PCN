# Code adapted from Cascaded Point Completion by Yuan et al. (2019) - Pytorch Implementation

import argparse
import torch.cuda
import sys
import os
from torch.utils.data import DataLoader
from scripts.DatasetLoader import DatasetLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from scripts.model import Encoder, Decoder, Generator, Discriminator
from scripts.model import Encoder, Decoder, Generator
from pytorch3d.loss import chamfer



def train(args):
    # Load datasets
    train_data = DatasetLoader(args.train_data, 'train')
    data_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # val_data = DatasetLoader(args.val_data, 'val', args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Set up the Generator
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    generator = Generator(encoder, decoder).to(args.device)
    # discriminator = Discriminator().to(args.device)

    # Optimisers
    optimiser_g = torch.optim.Adam(list(generator.parameters()), lr=args.generator_learning_rate)
    # optimiser_d = torch.optim.Adam(list(discriminator.parameters()), lr=args.discriminator_learning_rate)

    train_num = len(data_train)
    num_batches = train_num // args.batch_size
    for epoch in range(args.epochs):
        print(f"Training epoch {epoch + 1} / {args.epochs}")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, train_num)
            partial_input_batch, ground_truth_batch = data_train[start_idx:end_idx]
            print(partial_input_batch, ground_truth_batch)
            # Chamfer Distance from Pytorch3D
            coarse_batch, fine_batch = generator(partial_input_batch, args.step_ratio)
            chamfer_coarse, _ = chamfer_distance(coarse_batch, ground_truth_batch, point_reduction=None)
            dist1_coarse, dist2_coarse = chamfer_coarse
            chamfer_fine = chamfer_distance(fine_batch, ground_truth_batch, point_reduction=None)
            dist1_fine, dist2_fine = chamfer_fine
            print(f"Coarse Chamfer: {dist1_coarse, dist2_coarse}")
            print(f"Fine Chamfer: {dist1_fine, dist2_fine}")

            # Generator Loss Function
            total_loss_fine = (torch.mean(torch.sqrt(dist1_fine)) + torch.mean(torch.sqrt(dist2_fine))) / 2
            total_loss_coarse = (torch.mean(torch.sqrt(dist1_coarse)) + torch.mean(torch.sqrt(dist2_coarse))) / 2
            alpha = 0.5
            total_loss_rec_batch = alpha * total_loss_fine + (1 - alpha) * total_loss_coarse
            print(f"Total batch reconstruction loss: {total_loss_rec_batch}")

            # # Discriminator
            # d_fake = discriminator(fine_batch, divide_ratio=2)
            # d_real = discriminator(ground_truth_batch, divide_ratio=2)
            # d_loss_real = torch.mean((d_real - 1) ** 2)
            # d_loss_fake = torch.mean(d_fake ** 2)
            # errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
            # errG_loss_batch = torch.mean((d_fake - 1) ** 2)

            # total_gen_loss_batch = errG_loss_batch + total_loss_rec_batch * args.rec_weight
            # total_dis_loss_batch = errD_loss_batch

            optimiser_g.zero_grad()
            # total_gen_loss_batch.backward()
            total_loss_rec_batch.backward()
            optimiser_g.step()

            # optimiser_d.zero_grad()
            # total_dis_loss_batch.backward()
            # optimiser_d.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='~/../vol/bitbucket/rqg23/faceCompletionData')
    parser.add_argument('--mode', default='train', type=str)
    # parser.add_argument('--val_data', default='./data/val_data')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--generator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--discriminator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_num_points', default=2048, type=int)
    parser.add_argument('--gt_num_points', default=2048, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--rec_weight', default=200.0, type=float)
    args = parser.parse_args()
    train(args)
