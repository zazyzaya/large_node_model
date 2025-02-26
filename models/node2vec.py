import torch
from torch import nn

from torch_geometric.nn.models import Node2Vec
from models.positional_encoder import PositionalEncoding
from models.masked_attention import TrueNorm

class AdvancedN2V(Node2Vec):
    def __init__(self, edge_index, embedding_dim, walk_length, context_size, walks_per_node = 1, p = 1, q = 1, num_negative_samples = 1, num_nodes = None, sparse = False,
                 hidden_size=256, transformer_layers=4):
        super().__init__(edge_index, embedding_dim, walk_length, context_size, walks_per_node, p, q, num_negative_samples, num_nodes, sparse)

        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.GELU()
        )
        self.pe = PositionalEncoding(hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, hidden_size//64,
                dim_feedforward=hidden_size*4,
                activation=nn.GELU()
            ),
            num_layers=transformer_layers
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            TrueNorm()
        )
        self.out_embs = nn.Embedding(self.embedding.num_embeddings, embedding_dim)

        self.mse_loss = nn.MSELoss()

    def loss(self, pos_rw, neg_rw):
        pos_z = self.proj(self.embedding(pos_rw.T)) # S x B x d
        pos_pe = self.pe(pos_z)
        pos_z = pos_pe + pos_z
        pos_z = self.out(self.transformer(pos_z))
        pos_z = pos_z.transpose(0,1) # B x S x d

        start, rest = pos_z[:, 0], pos_z[:, 1:].contiguous()

        h_start = start.view(pos_z.size(0), 1, self.embedding_dim)
        h_rest = rest.view(-1).view(pos_z.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        recon_loss = self.mse_loss.forward(
            self.out_embs(pos_rw[:, 0]),
            start.detach()
        )

        # Negative loss.
        neg_z = self.proj(self.embedding(neg_rw.T)) # S x B x d
        neg_pe = self.pe(neg_z)
        neg_z = neg_pe + neg_z
        neg_z = self.out(self.transformer(neg_z))
        neg_z = neg_z.transpose(0,1) # B x S x d

        start, rest = neg_z[:, 0], neg_z[:, 1:].contiguous()

        h_start = start.view(neg_z.size(0), 1, self.embedding_dim)
        h_rest = rest.view(-1).view(neg_z.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return (pos_loss + neg_loss), recon_loss