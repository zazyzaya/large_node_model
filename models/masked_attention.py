import torch
from torch import nn

from .positional_encoder import PositionalEncoding

class MaskedAttentionEmb(nn.Module):
    def __init__(self, num_nodes, num_rels, emb_dim=128, context_window=3, device='cpu', hidden_size=768, layers=12):
        super().__init__()

        self.device=device
        self.context_window=context_window
        self.args = (num_nodes,num_rels)
        self.kwargs = dict(
            hidden_size=hidden_size,
            layers=layers,
            context_window=context_window,
            emb_dim=emb_dim
        )

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_embeddings = num_nodes+num_rels
        self.embed = nn.Sequential(
            nn.Embedding(num_nodes+num_rels, emb_dim, device=self.device),
            nn.LayerNorm(emb_dim, device=self.device)
        )

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_size, device=self.device),
            nn.GELU()
        )
        self.pe = PositionalEncoding(hidden_size, device=self.device)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, hidden_size//64,
                dim_feedforward=hidden_size*4,
                activation=nn.GELU(),
                device=self.device
            ),
            num_layers=layers
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, emb_dim, device=self.device),
            nn.LayerNorm(emb_dim, device=self.device)
        )
        self.out_embs = nn.Sequential(
            nn.Embedding(num_nodes+num_rels, emb_dim, device=self.device),
            nn.LayerNorm(emb_dim, device=self.device)
        )

        self.loss = nn.BCEWithLogitsLoss()

    def build_mask(self, seq_len):
        '''
        Mask s.t. only nodes within the context window are visible
        except for the target node.
        E.g. if context window was 2-hops, then mask will be
        [
            [0, 0, 0, 0, -inf, 0, 0, 0, 0, -inf, ..., -inf ]
            [-inf, 0, 0, 0, 0, -inf, 0, 0, 0, 0, -inf, ..., -inf ]
            ...
            [-inf, ... , -inf, 0, 0, 0, 0, -inf, 0, 0, 0, 0]
        ]

        4 zeros, not 2 to capture (node, relation) pairs in the random walk
        '''

        mask = torch.full((seq_len, seq_len), float('-inf'), device=self.device)
        submask = [0]*self.context_window*2
        submask = submask + [float('-inf')] + submask
        submask = torch.tensor(submask, device=self.device)

        offset = 0
        targets = []
        for row in mask:
            can_fit = row.size(-1)-offset
            if can_fit >= submask.size(0):
                row[offset:submask.size(0)+offset] = submask
                targets.append(self.context_window+offset+1)
            else:
                break
            offset += 1

        return mask, torch.tensor(targets, device=self.device)

    @torch.no_grad()
    def link_prediction(self, head):
        '''
        TODO include relations
        '''
        head = head.to(self.device)

        embs = self.out_embs(head)
        others = self.out_embs(torch.arange(self.num_nodes, device=self.device))
        ranks = embs @ others.T

        # Dot prod with itself will always be 1 so zero that out
        ranks[torch.arange(head.size(0), device=self.device), head] = 0
        return ranks

    def forward(self, seq):
        '''
        Expects B x S list of node and relation ids
        '''
        seq = seq.to(self.device).T # S x B
        mask,targets = self.build_mask(seq.size(0))

        # Get tokens
        z = self.embed(seq)
        tokens = self.proj(z)
        pe = self.pe(tokens)
        tokens = tokens + pe

        # Run them through model
        preds = self.transformer.forward(tokens, mask)
        preds = self.out(preds)

        # Use n2v-like loss to make Sum( f(u) * f(v) ) high
        # for input u and neighbors v
        # and Sum( f(u) * f(n) ) low for input u and neg samples n
        preds = preds.transpose(0,1).transpose(1,2) # B x d x S
        out_z = self.out_embs(seq).transpose(0,1)   # B x S x d
        pos_dist = out_z @ preds     # B x S x S

        # Each row is dot prod of target and all neighbors in context window
        mask_mask = mask==0
        mask[mask_mask] = 1
        mask[~mask_mask] = 0
        mask = mask.bool()

        pos_dist = pos_dist[:, mask].view(-1, self.context_window*2)
        pos_dist = pos_dist.mean(dim=1)

        # Calculate distance of target node from random context window
        neg = self.out_embs(torch.randint(0, self.num_nodes, seq.size(), device=self.device))
        neg = neg.transpose(0,1).transpose(1,2)
        neg_dist = out_z @ neg
        neg_dist = neg_dist[:,mask].view(-1, self.context_window*2)
        neg_dist = neg_dist.mean(dim=1)

        # Run loss s.t. dot prod of neighbors and input is 1, otherwise 0
        all_dists = torch.cat([pos_dist, neg_dist])
        labels = torch.zeros(all_dists.size(), device=self.device)
        labels[:pos_dist.size(0)] = 1

        loss = self.loss(all_dists, labels)
        return loss

        '''
        # Old loss function
        z_hat = self.out(preds)[targets]
        z = z[targets]

        # Use center of masked regions as targets
        pos_hat = z_hat.reshape(z_hat.size(0)*z_hat.size(1), -1)
        pos = z.reshape(z.size(0)*z.size(1), -1)
        pos = (pos * pos_hat).sum(dim=-1)

        # Negatively sample
        neg = self.embed(torch.randint(0, self.num_embeddings, (pos_hat.size(0),), device=self.device))
        neg = (neg * pos_hat).sum(dim=-1)

        # Loss is based on similarity of prediction to actual embedding
        # and dissimilarity to random embeddings
        labels = torch.zeros((pos.size(0) + neg.size(0)), device=self.device)
        labels[:pos.size(0)] = 1
        loss = self.loss(
            torch.cat([pos,neg]),
            labels
        )

        return loss
        '''