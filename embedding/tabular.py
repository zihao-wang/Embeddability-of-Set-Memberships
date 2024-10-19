import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

class TabularEmbedding(nn.Module):
    def __init__(self, m, n, num_sets):
        super().__init__()
        self.m = m
        self.n = n
        self.ele_emb = nn.Embedding(m, n)
        self.set_emb = nn.Embedding(num_sets, n)

    def forward(self, set_id):
        set_id.squeeze_()
        return self.set_emb(set_id)