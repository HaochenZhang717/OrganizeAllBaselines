import os
import torch
import numpy as np
from torch.utils.data import Dataset
from Utils.masking_utils import noise_mask


class PreSplitNPYDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        # window,
        # period='train',
        split='train',
        seed=123,
        # predict_length=None,
        # missing_ratio=None,
        # style='separate',
        # distribution='geometric',
        # mean_mask_length=3,
        # output_dir=None,
        **kwargs
    ):
        super().__init__()

        assert split in ['train', 'valid', 'test']
        # assert period in ['train', 'test']

        self.name = name
        self.data_root = data_root
        # self.window = window
        # self.period = period
        self.split = split
        self.seed = seed
        # self.pred_len = predict_length
        # self.missing_ratio = missing_ratio
        # self.style = style
        # self.distribution = distribution
        # self.mean_mask_length = mean_mask_length
        # self.output_dir = output_dir

        file_map = {
            'train': 'train_ts.npy',
            'valid': 'valid_ts.npy',
            'test': 'test_ts.npy'
        }

        path = os.path.join(data_root, file_map[split])
        self.samples = np.load(path).astype(np.float32)
        self.samples = torch.from_numpy(self.samples)
        assert self.samples.ndim == 3, f"Expected (N,L,D), got {self.samples.shape}"
        # assert self.samples.shape[1] == window, (
        #     f"Window mismatch: got {self.samples.shape[1]}, expected {window}"
        # )

        self.sample_num = self.samples.shape[0]


    def __getitem__(self, ind):
        # if self.period == 'test':
        #     return (
        #         torch.from_numpy(self.samples[ind]).float(),
        #         torch.from_numpy(self.masking[ind])
        #     )
        return self.samples[ind].float()

    def __len__(self):
        return self.sample_num