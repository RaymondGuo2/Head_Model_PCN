from importlib.metadata import files
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch


class DatasetLoader(Dataset):
    def __init__(self, directory_path, mode='train', batch_size=32, shuffle=True):
        self.directory_path = directory_path
        self.partial_inputs = []
        self.ground_truths = []
        # Load the data depending on which mode (train, validation, test) it is
        self.initialise_inputs_and_ground_truths(directory_path, mode)

    def initialise_inputs_and_ground_truths(self, directory_path, mode):
        data_path = os.path.join(directory_path, mode)
        inputs = os.path.join(data_path, 'partial_inputs')
        gts = os.path.join(data_path, 'ground_truths')
        self.partial_inputs = [os.path.join(inputs, file) for file in os.listdir(inputs) if file.endswith('.npy')]
        self.ground_truths = [os.path.join(gts, file) for file in os.listdir(gts) if file.endswith('.npy')]

        self.partial_inputs.sort()
        self.ground_truths.sort()

    def __len__(self):
        return len(self.partial_inputs)

    # Recheck this
    def __getitem__(self, index):
        partial_input = self.partial_inputs[index]
        ground_truth = self.ground_truths[index]

        print(f"The partial input is {partial_input}")
        print(f"The ground truth is {ground_truth}")

        # Load the numpy file
        partial_input = np.load(partial_input)
        ground_truth = np.load(ground_truth)

        # Convert numpy arrays to tensors for processing
        partial_input = torch.tensor(partial_input, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        return partial_input, ground_truth


if __name__ == '__main__':
    data = DatasetLoader(directory_path='../dataset', mode='train')
    partial_input, gt_input = data[0]
    print(partial_input, gt_input)