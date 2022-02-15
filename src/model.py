import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import data_util


class ResidualSelfAttention(nn.Module):
    def __init__(self, length: int, dim: int, num_heads: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros([]))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

        mask = torch.ones(length, length).tril() >= 0.5
        mask = mask.float().masked_fill(mask.logical_not(),
                                        float('-inf')).masked_fill(mask, float(0.0))
        self.register_buffer("attn_mask", mask)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        o, _ = self.mha(q, k, v, attn_mask=self.attn_mask)
        o = self.out(o)
        return x + self.alpha * F.relu(o)


class EmbeddingLayer(nn.Module):
    def __init__(self, indexer: data_util.SymbolIndexer, dim: int):
        super().__init__()
        self.indexer = indexer
        self.dim = dim
        self.embeddings = nn.Parameter(torch.randn(self.indexer.size(), dim))

    def one_hot(self, x: torch.ByteTensor) -> torch.Tensor:
        N, L = x.size()
        D = self.indexer.size()
        eye = torch.eye(D).to(x.device)
        x = x.reshape(N * L).long()
        x = torch.index_select(eye, dim=0, index=x)
        x = x.reshape(N, L, D)
        return x

    def interpret(self, x: torch.Tensor, k: int = 3) -> List[int]:
        D, = x.size()
        assert D == self.indexer.size()
        _, indices = torch.topk(x, k, sorted=True)
        return [self.indexer.to_symbol(i.item()) for i in indices]

    def forward(self, x: torch.ByteTensor) -> torch.Tensor:
        N, L = x.size()
        D = self.dim
        x = x.reshape(N * L).long()
        x = torch.index_select(self.embeddings, dim=0, index=x)
        x = x.reshape(N, L, D)
        return x


class PELayer(nn.Module):
    def __init__(self, length: int, dim: int):
        super().__init__()
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             (-math.log(10000.0) / dim))
        pe = torch.zeros(length, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class BasicModel(nn.Module):
    def __init__(self, length: int, indexer: data_util.SymbolIndexer, dim: int):
        super().__init__()
        self.embed = EmbeddingLayer(indexer, dim)
        self.pe = PELayer(length, dim)
        self.attention_0 = ResidualSelfAttention(length, dim, 4)
        self.attention_1 = ResidualSelfAttention(length, dim, 4)
        self.attention_2 = ResidualSelfAttention(length, dim, 4)
        self.attention_3 = ResidualSelfAttention(length, dim, 4)
        self.out = nn.Linear(dim, indexer.size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pe(x)
        x = self.attention_0(x)
        x = self.attention_1(x)
        x = self.attention_2(x)
        x = self.attention_3(x)
        x = self.out(x)
        return x
