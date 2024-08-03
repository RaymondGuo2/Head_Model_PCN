import argparse
import numpy as np
from cloud_to_mesh import numpy_to_cloud, cloud_to_mesh
import torch
from scripts.model import Encoder, Decoder, Generator
from scripts.model import Decoder


def render(args):
    input = np.load(args.render_file)
    input = torch.as_tensor(input).unsqueeze(0).to(args.device)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    generator = Generator().to(args.device)

    generator.load_state_dict(torch.load(args.checkpoint))
    generator.eval()
    with torch.no_grad():
        _, fine_output = generator(input, args.step_ratio)

    dense_reconstruction = fine_output.squeeze(0).cpu().numpy()
    output = numpy_to_cloud(dense_reconstruction)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_file', type=str, help='Must be a numpy file!')
    parser.add_argument('--checkpoint', default='/Users/raymondguo/Desktop/IndividualProject/object_autocompletion/checkpoint/generator_new.pth')
    parser.add_argument('--step_ratio', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    render(args)
