import torch
import numpy as np
import datasets.Base_dataset as Base_dataset


class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_file, clean_file=None):
        self.noisy_data = np.load(noisy_file)
        self.clean_data = None if clean_file is None else np.load(clean_file)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        noisy_image = torch.tensor(self.noisy_data[idx], dtype=torch.float32)
        if self.clean_data is not None:
            clean_image = torch.tensor(self.clean_data[idx], dtype=torch.float32)
            return noisy_image, clean_image
        return noisy_image