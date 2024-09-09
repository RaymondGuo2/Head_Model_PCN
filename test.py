# Code adapted from the original repository code for the paper Cascaded Refinement Network for Point Cloud Completion by Wang et al. (2020)
# Original repository can be found here: https://github.com/xiaogangw/cascaded-point-completion

import argparse
import torch.cuda
from torch.utils.data import DataLoader
from scripts.DatasetLoader import DatasetLoader
from scripts.model import Encoder, Decoder, Generator
from pytorch3d.loss import chamfer_distance
import matplotlib.pyplot as plt
import os


def test(args):
    test_data = DatasetLoader(args.test_data, 'test')
    data_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    generator = Generator(encoder, decoder).to(args.device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    generator.eval()
    test_loss = 0
    losses = []

    with torch.no_grad():
        for batch_idx, (partial_input_batch, ground_truth_batch) in enumerate(data_test):
            partial_input_batch = partial_input_batch.to(args.device)
            ground_truth_batch = ground_truth_batch.to(args.device)
            _, fine_batch = generator(partial_input_batch, args.step_ratio)

            chamfer_fine, _ = chamfer_distance(fine_batch, ground_truth_batch, batch_reduction=None, point_reduction=None)
            dist1_fine, dist2_fine = chamfer_fine

            total_loss_fine = (torch.mean(torch.sqrt(dist1_fine)) + torch.mean(torch.sqrt(dist2_fine))) / 2
            test_loss += total_loss_fine.item()
            losses.append(total_loss_fine.item())
            print(f"Test_loss: {test_loss}")

        test_loss /= len(data_test)
        print(f"Test Loss: {test_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Batch Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Test Loss per Batch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('./plots', 'test_loss.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='/Users/raymondguo/Desktop/IndividualProject/faceCompletionData')
    parser.add_argument('--checkpoint', default='/Users/raymondguo/Desktop/IndividualProject/object_autocompletion/checkpoint/train_0408_1056.pth')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--step_ratio', type=int, default=2)
    args = parser.parse_args()
    test(args)