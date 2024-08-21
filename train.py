# Code adapted from Cascaded Point Completion by Yuan et al. (2019) - Pytorch Implementation

import argparse
import torch.cuda
import sys
import os
from torch.utils.data import DataLoader
from scripts.DatasetLoader import DatasetLoader
from scripts.model import Encoder, Decoder, Generator, Discriminator
from pytorch3d.loss import chamfer_distance
import torch.autograd as autograd
import matplotlib.pyplot as plt


def train(args):
    # Load datasets
    # autograd.set_detect_anomaly(True)
    train_data = DatasetLoader(args.train_data, 'train')
    val_data = DatasetLoader(args.val_data, 'val')
    data_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    data_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set up the Generator
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    generator = Generator(encoder, decoder).to(device)
    discriminator = Discriminator().to(device)

    # Optimisers
    optimiser_g = torch.optim.Adam(list(generator.parameters()), lr=args.generator_learning_rate)
    optimiser_d = torch.optim.Adam(list(discriminator.parameters()), lr=args.discriminator_learning_rate)

    train_losses = []
    val_losses = []
    generator_losses = []
    discriminator_losses = []


    for epoch in range(args.epochs):
        print(f"Training epoch {epoch + 1} / {args.epochs}")

        generator.train()
        discriminator.train()

        train_loss_epoch = 0
        generator_loss_epoch = 0
        discriminator_loss_epoch = 0

        for batch_idx, (partial_input_batch, ground_truth_batch) in enumerate(data_train):
            print(f"Training batch {batch_idx + 1} / {len(data_train)}")
            partial_input_batch = partial_input_batch.to(device)
            ground_truth_batch = ground_truth_batch.to(device)
            # Chamfer Distance from Pytorch3D
            coarse_batch, fine_batch = generator(partial_input_batch, args.step_ratio)
            chamfer_coarse, _ = chamfer_distance(coarse_batch, ground_truth_batch, batch_reduction=None, point_reduction=None)
            dist1_coarse, dist2_coarse = chamfer_coarse
            chamfer_fine, _ = chamfer_distance(fine_batch, ground_truth_batch, batch_reduction=None, point_reduction=None)
            dist1_fine, dist2_fine = chamfer_fine
            # Generator Loss Function
            total_loss_fine = (torch.mean(torch.sqrt(dist1_fine)) + torch.mean(torch.sqrt(dist2_fine))) / 2
            total_loss_coarse = (torch.mean(torch.sqrt(dist1_coarse)) + torch.mean(torch.sqrt(dist2_coarse))) / 2
            alpha = 0.5
            total_loss_rec_batch = alpha * total_loss_fine + (1 - alpha) * total_loss_coarse
            print(f"Total batch reconstruction loss: {total_loss_rec_batch}")

            # Discriminator
            d_fake = discriminator(fine_batch, divide_ratio=2)
            d_real = discriminator(ground_truth_batch, divide_ratio=2)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss_fake = torch.mean(d_fake ** 2)
            errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)

            total_dis_loss_batch = errD_loss_batch
            optimiser_d.zero_grad()
            total_dis_loss_batch.backward(retain_graph=True)

            errG_loss_batch = torch.mean((d_fake - 1) ** 2)
            total_gen_loss_batch = errG_loss_batch + total_loss_rec_batch * args.rec_weight
            optimiser_g.zero_grad()
            total_gen_loss_batch.backward()

            optimiser_d.step()
            optimiser_g.step()

            train_loss_epoch += total_loss_fine.item()
            generator_loss_epoch += total_gen_loss_batch.item()
            discriminator_loss_epoch += total_dis_loss_batch.item()

        train_loss_epoch /= len(data_train)
        generator_loss_epoch /= len(data_train)
        discriminator_loss_epoch /= len(data_train)

        train_losses.append(train_loss_epoch)
        generator_losses.append(generator_loss_epoch)
        discriminator_losses.append(discriminator_loss_epoch)

        generator.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for batch_idx, (partial_input_batch, ground_truth_batch) in enumerate(data_val):
                partial_input_batch = partial_input_batch.to(device)
                ground_truth_batch = ground_truth_batch.to(device)
                coarse_batch, fine_batch = generator(partial_input_batch, args.step_ratio)

                chamfer_fine, _ = chamfer_distance(fine_batch, ground_truth_batch, batch_reduction=None, point_reduction=None)
                dist1_fine, dist2_fine = chamfer_fine

                total_loss_fine = (torch.mean(torch.sqrt(dist1_fine)) + torch.mean(torch.sqrt(dist2_fine))) / 2
                val_loss_epoch += total_loss_fine.item()

        val_loss_epoch /= len(data_val)
        val_losses.append(val_loss_epoch)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss_epoch:.4f}, "
              f"Validation Loss: {val_loss_epoch:.4f}, Generator Loss: {generator_loss_epoch:.4f}, "
              f"Discriminator Loss: {discriminator_loss_epoch:.4f}")

    torch.save(generator.state_dict(), os.path.join(args.checkpoint, 'train_1108_long.pth'))

    # Plot and save the Training vs Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint, 'train_vs_val_loss.png'))

    # Plot and save the Generator vs Discriminator Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), generator_losses, label='Generator Loss')
    plt.plot(range(1, args.epochs + 1), discriminator_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint, 'gen_vs_dis_loss.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='/vol/bitbucket/rqg23/project_data')
    parser.add_argument('--val_data', default='/vol/bitbucket/rqg23/project_data')
    parser.add_argument('--checkpoint', default='./checkpoint')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--generator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--discriminator_learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_num_points', default=2048, type=int)
    parser.add_argument('--gt_num_points', default=2048, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--rec_weight', default=200.0, type=float)
    parser.add_argument('--step_ratio', type=int, default=2)
    parser.add_argument('--plot_path', type=str, default='./plots')
    args = parser.parse_args()
    train(args)
