import argparse
import numpy as np
from cloud_to_mesh import numpy_to_cloud, cloud_to_mesh
import torch
from model import Encoder, Decoder, Generator

def render(args):
    input = np.load(args.render_file)
    input = torch.as_tensor(input).unsqueeze(0).to(args.device)
    input = input.float()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    generator = Generator(encoder, decoder).to(device)

    generator.load_state_dict(torch.load(args.checkpoint))
    generator.eval()
    with torch.no_grad():
        _, fine_output = generator(input, args.step_ratio)
        print(fine_output.shape)

    dense_reconstruction = fine_output.squeeze(0).cpu().numpy()
    print(dense_reconstruction)
    np.save('trial.npy', dense_reconstruction)
    output = numpy_to_cloud(dense_reconstruction)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_file', default='/homes/rqg23/../../vol/bitbucket/rqg23/faceCompletionData/test/partial_inputs/00608.npy', type=str, help='Must be a numpy file!')
    parser.add_argument('--checkpoint', default='/homes/rqg23/individualproject/object_autocompletion/checkpoint/generator_new.pth')
    parser.add_argument('--step_ratio', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    render(args)
