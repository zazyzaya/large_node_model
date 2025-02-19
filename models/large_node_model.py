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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NodeGPT(nn.Module):
    '''
    Follows GPT-3 style of predicting next token
    '''
    def __init__(self, dict_size, in_dim=32, device='cpu', hidden_dim=64, inner_dim=128, heads=6, layers=6):
        super().__init__()

        self.device=device

        self.vector_store = nn.Embedding(dict_size, in_dim, device=device)
        self.pe = PositionalEncoding(in_dim)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*heads, device=device),
            nn.GELU()
        )
        self.transformers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_dim*heads,
                dim_feedforward=inner_dim*heads,
                nhead=heads,
                device=device,
                activation=nn.GELU()
            ),
            num_layers=layers
        )
        self.out = nn.Linear(
            hidden_dim*heads,
            dict_size,
            device=device
        )

        self.start = nn.parameter.Parameter(torch.empty(hidden_dim*heads), device=device)
        nn.init.xavier_normal_(self.start)

        self.end = nn.parameter.Parameter(torch.empty(hidden_dim*heads), device=device)
        nn.init.xavier_normal_(self.end)

        self.loss = nn.CrossEntropyLoss()

    def _tokenize_seq(self, seq):
        z = self.pe(self.vector_store(seq))

        st = self.start.repeat(1, z.size(1), 1)
        en = self.end.repeat(1, z.size(1), 1)
        return torch.cat(
            [st, z, en],
            dim=0
        )

    def next_node(self, seq):
        # Predict next
        z = self._tokenize_seq(seq) # 1+S+1 x B x d
        emb = self.transformers(z)
        preds = self.out(emb[-1])

        return preds

    def forward(self, sequences):
        '''
        Expect sequence of node ids
        S x B
        '''
        seq = sequences[:-1]
        y = sequences[-1]

        preds = self.next_node(seq)
        loss = self.loss(preds, y)
        return loss