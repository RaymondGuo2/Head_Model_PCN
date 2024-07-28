# Code adapted from Cascaded Point Completion by Yuan et al. (2019) - Pytorch Implementation

import argparse
import torch.cuda
from scripts.DatasetLoader import DatasetLoader
from scripts.model import Encoder, Decoder, Generator, Discriminator


def train(args):
    # Load datasets
    train_data = DatasetLoader(args.train_data, 'train', args.batch_size, shuffle=True)
    # val_data = DatasetLoader(args.val_data, 'val', args.batch_size, shuffle=False)

    # Set up the Generator
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    generator = Generator(encoder, decoder).to(args.device)
    discriminator = Discriminator().to(args.device)

    # Optimisers
    optimiser_g = torch.optim.Adam(generator.parameters(), lr=args.generator_learning_rate)
    optimiser_d = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate)

    train_num = len(train_data)
    num_batches = train_num // args.batch_size
    for epoch in range(args.epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, train_num)
            partial_input_batch, ground_truth_batch = train_data[start_idx:end_idx]

            coarse_batch, fine_batch = generator(partial_input_batch, args.step_ratio)
            dist1_coarse, dist2_coarse = chamfer_distance(coarse_batch, ground_truth_batch)
            dist1_fine, dist2_fine = chamfer_distance(fine_batch, ground_truth_batch)

            # Generator Loss Function
            total_loss_fine = (torch.mean(torch.sqrt(dist1_fine)) + torch.mean(torch.sqrt(dist2_fine))) / 2
            total_loss_coarse = (torch.mean(torch.sqrt(dist1_coarse)) + torch.mean(torch.sqrt(dist2_coarse))) / 2
            total_loss_rec_batch = alpha * total_loss_fine + total_loss_coarse

            # Discriminator
            d_fake = discriminator(fine_batch, divide_ratio=2)
            d_real = discriminator(ground_truth_batch, divide_ratio=2)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss_fake = torch.mean(d_fake ** 2)
            errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
            errG_loss_batch = torch.mean((d_fake - 1) ** 2)

            total_gen_loss_batch = errG_loss_batch + total_loss_rec_batch * args.rec_weight
            total_dis_loss_batch = errD_loss_batch

            optimiser_g.zero_grad()
            total_gen_loss_batch.backward()
            optimiser_g.step()

            optimiser_d.zero_grad()
            total_dis_loss_batch.backward()
            optimiser_d.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./data/train_data')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--val_data', default='./data/val_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--generator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--discriminator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_num_points', default=2048, type=int)
    parser.add_argument('--gt_num_points', default=2048, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--rec_weight', default=200.0, type=float)
    args = parser.parse_args()
    train(args)
