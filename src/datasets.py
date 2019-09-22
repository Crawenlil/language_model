from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):

    def __init__(self, dataset, left_context_len=None, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.lgram = left_context_len
        self.lengths = self.dataset.astype(bool).sum(axis=1) - 2
        self.lengths_cumsum = np.insert(np.cumsum(self.lengths), 0, 0)

    def __len__(self):
        return self.lengths.sum()

    def __getitem__(self, idx):
        row = self.lengths_cumsum.searchsorted(idx, 'right') - 1
        col = idx - self.lengths_cumsum[row] + 1
        word = self.dataset[row][col]
        lc_idx = max(0, col - self.lgram) if self.lgram is not None else 0
        l_context = self.dataset[row][lc_idx:col]
        l_context = np.pad(l_context, (0, self.lgram - len(l_context)))
        # ln = self.lengths[row]
        # rc_idx = min(ln, col + self.rgram) if self.rgram is not None else ln
        # r_context = self.dataset[row][col+1:rc_idx]

        sample = {
            'left_context': l_context,
            'word': word
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
