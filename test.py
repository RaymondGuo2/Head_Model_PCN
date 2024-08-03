import argparse
import torch.cuda
from torch.utils.data import DataLoader
from scripts.DatasetLoader import DatasetLoader
from scripts.model import Encoder, Decoder, Generator
from pytorch3d.loss import chamfer_distance


def test(args):
    test_data = DatasetLoader(args.test_data, 'test')
    data_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    generator = Generator(encoder, decoder).to(args.device)
    generator.load_state_dict(torch.load(args.checkpoint))

    generator.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (partial_input_batch, ground_truth_batch) in enumerate(data_test):
            partial_input_batch = partial_input_batch.to(args.device)
            ground_truth_batch = ground_truth_batch.to(args.device)
            _, fine_batch = generator(partial_input_batch, args.step_ratio)

            chamfer_fine, _ = chamfer_distance(fine_batch, ground_truth_batch, batch_reduction=None, point_reduction=None)
            dist1_fine, dist2_fine = chamfer_fine

            total_loss_fine = (torch.mean(torch.sqrt(dist1_fine)) + torch.mean(torch.sqrt(dist2_fine))) / 2
            test_loss += total_loss_fine.item()

    test_loss /= len(data_test)

    print(f"Test Loss: {test_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='/Users/raymondguo/Desktop/IndividualProject/faceCompletionData')
    parser.add_argument('--checkpoint', default='/Users/raymondguo/Desktop/IndividualProject/object_autocompletion/checkpoint/generator_new.pth')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--step_ratio', type=int, default=2)
    args = parser.parse_args()
    test(args)