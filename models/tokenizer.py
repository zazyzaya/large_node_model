from math import log10, floor, ceil

import torch
from torch import nn

from .positional_encoder import PositionalEncoding

class NodeTokenizer(nn.Module):
    def __init__(self, num_nodes, num_rels, dim, mask_rate=0.15, device='cpu'):
        super().__init__()

        self.node_digits = 1 + floor(log10(num_nodes))
        self.rel_digits = 1 + floor(log10(num_rels))
        self.device=device
        self.mask_rate = mask_rate

        # Break into sets of 1000 nodes (000-999)
        self.num_node_embs = ceil(self.node_digits / 3)
        self.num_rel_embs = ceil(self.rel_digits / 3)

        self.emb = nn.Embedding(
            1000 * (self.num_node_embs + self.num_rel_embs) +
            2 + # <node_id>, <rel_id>
            3,  # <mask>, <pad>, <cls>
            dim,
            device=device
        )
        self.pe = PositionalEncoding(dim, device=device)
        self.token_types = nn.Embedding(3, dim, device=device)
        self.norm = nn.LayerNorm(dim, device=device)

        self.NID = 1000 * (self.num_node_embs + self.num_rel_embs)
        self.RID = self.NID+1
        self.MASK = self.RID+1
        self.PAD = self.MASK+1
        self.CLS = self.PAD+1

        self.NODE = 0
        self.REL = 1
        self.SPECIAL = 2

    def _decompose(self, inputs, num_embs, special_token, offset=0):
        tokens = [
            (inputs // 1000 ** i) % 1000 + 1000*(i+offset)
            for i in range(num_embs-1, -1, -1)
        ]

        # B x S x num_embs
        tokens = torch.stack(tokens, dim=2)

        # Add <type> token
        st = torch.tensor([special_token], device=self.device).repeat(
            tokens.size(0),
            tokens.size(1),
            1
        )
        tokens = torch.cat([st, tokens], dim=-1)

        # B x S * num_embs
        tokens = tokens.view(tokens.size(0), -1)
        return tokens

    def tokenize(self, seq):
        '''
        Assume input seq is B x S random walk:
            node, rel, node, rel, ..., node
        '''
        node_mask = torch.tensor([True, False], device=self.device)
        node_mask = node_mask.repeat(seq.size(1) // 2 + 1)[:-1]
        nodes = seq[:, node_mask]
        rels = seq[:, ~node_mask]

        node_tokens = self._decompose(nodes, self.num_node_embs, self.NID)
        rel_tokens = self._decompose(rels, self.num_rel_embs, self.RID, offset=self.num_node_embs)

        # Weave them back together
        seq = torch.empty((
            node_tokens.size(1)+rel_tokens.size(1),
            node_tokens.size(0)
        ), device=self.device, dtype=torch.long)

        node_len = self.num_node_embs + 1
        rel_len = self.num_rel_embs + 1
        for i in range(seq.size(1)):
            reading_node = True
            n_idx = 0
            r_idx = 0

            # Alternate between reading from node tokens and rel tokens
            for j in range(seq.size(0)):
                if reading_node:
                    seq[j][i] = node_tokens[i, n_idx]
                    n_idx += 1

                    if n_idx % node_len == 0:
                        reading_node = False
                else:
                    seq[j][i] = rel_tokens[i, r_idx]
                    r_idx += 1
                    if r_idx % rel_len == 0:
                        reading_node = True

        return seq

    def _mask_seq(self, seq):
        '''
        seq: a S x B matrix of random walks through the graph

        Select 15% to mask,
        of those,   replace 80% with <MASK> tokens
                    replace 10% with random words
                    do nothing with the remaining 10%
        '''
        to_predict = torch.rand(seq.size()) < self.mask_rate
        predicting = to_predict.nonzero()
        idxs = torch.randperm(predicting.size(0))
        to_mask = predicting[ idxs[:int(predicting.size(0) * 0.8)] ]
        to_replace = predicting[ idxs[int(predicting.size(0) * 0.8):int(predicting.size(0) * 0.9)] ]

        tgt = seq[to_predict]

        m_rows,m_cols = to_mask.T
        seq[m_rows, m_cols] = self.MASK
        r_rows,r_cols = to_replace.T
        seq[r_rows,r_cols] = torch.randint(0, self.MASK, r_rows.size(), device=self.device)

        return seq,tgt,to_predict

    def type_encoder(self, tokens):
        types = torch.zeros(tokens.size(), device=self.device, dtype=torch.long)
        #types[tokens < self.REL] = self.NODE # Implicit
        types[(tokens >= self.REL).logical_and(tokens < self.NID)] = self.REL
        types[tokens >= self.NID] = self.SPECIAL

        return self.token_types(types)

    def forward(self, seq, mask=False):
        seq = seq.to(self.device)
        tokens = self.tokenize(seq)

        if mask:
            tokens,tgt,to_predict = self._mask_seq(tokens)

        embs = self.emb(tokens)
        pe = self.pe(tokens)
        ttypes = self.type_encoder(tokens)

        embs = self.norm(
            embs + pe + ttypes
        )

        if mask:
            return embs, tgt, to_predict
        return embs