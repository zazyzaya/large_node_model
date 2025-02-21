import math

import torch
import torch.functional as F
from torch import nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    '''
    Stolen from stack overflow
    https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[:x.size(0)]