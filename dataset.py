from tqdm import trange

import torch
from torch.utils.data import DataLoader, Dataset

from utils import dec2bin, bin2dec

class SetMembershipDataset(Dataset):
    def __init__(self, m, k):
        self.m = m
        self.k = k
        # assert k < 5 or m < 11
        # we use sparse vector here in data
        self.data = list()
        # initialize the dataset

        if k < m//2:
            # subset mode, when k is smaller than m//2.
            # this subroutine samples the elements recursively.
            def sample_at_most_k(_k):
                if _k == 0:
                    return [[]]

                last = sample_at_most_k(_k-1)
                this = []
                for i in range(self.m):
                    for l in last:
                        this.append([i] + l)
                return this

            for eles in sample_at_most_k(k):
                eles = list(set(eles))
                if len(eles) == 1:
                    continue
                bin_repr = torch.zeros(self.m, dtype=torch.bool)
                bin_repr[eles] = True
                self.data.append(bin_repr.to_sparse())
        else:
            # iterative model, when k is large, all subsets are tried.
            for i in trange(2**self.m):
                bin_repr = dec2bin(torch.tensor(i), self.m)
                num_eles = bin_repr.sum()
                if num_eles > self.k or num_eles == 0:
                    continue
                else:
                    self.data.append(bin_repr.to_sparse())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bin = self.data[idx].to_dense()
        return bin, idx