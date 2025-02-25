import torch
from torch import nn

from .positional_encoder import PositionalEncoding

class TrueNorm(nn.Module):
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return x / norm

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
        self.embed = nn.Embedding(num_nodes+num_rels, emb_dim, device=self.device)

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
            TrueNorm()
        )
        self.out_embs = nn.Sequential(
            nn.Embedding(num_nodes+num_rels, emb_dim, device=self.device),
            TrueNorm()
        )

        self.mse_loss = nn.MSELoss()

    def build_mask(self, seq_len):
        '''
        Mask s.t. only nodes within the context window are visible
        But also, ensure starting nodes have center node in their context
        window as well
        E.g. if context window was 1 hop, then mask will be
        [
            [0,     0, -inf,   -inf, ..., -inf, -inf, -inf]
            [0,     0,    0,   -inf, ..., -inf, -inf, -inf]
            [-inf,  0,    0,    0,   ..., -inf, -inf, -inf],
                         ...
            [-inf, -inf, ...,  -inf,   0,    0,   0,  -inf]
            [-inf, -inf, ...,  -inf,   -inf  0,   0,   0 ]
            [-inf, -inf, -inf, -inf, ..., -inf,   0,   0 ]
        ]

        4 zeros, not 2 to capture (node, relation) pairs in the random walk
        '''

        mask = torch.full((seq_len, seq_len), float('-inf'), device=self.device)
        sm_size = self.context_window*2 + 1

        targets = []
        for i in range(mask.size(0)):
            if i < self.context_window:
                c_size = self.context_window+1+i
                st = 0
            elif i >= mask.size(0)-self.context_window:
                st = i - self.context_window
            else:
                c_size = sm_size
                st = i - self.context_window
                targets.append(i)

            mask[i][st:st+c_size] = 0

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

        # Generate positive samples
        tokens = self.proj(self.embed(seq))
        pe = self.pe(tokens)
        tokens = tokens + pe
        preds = self.transformer.forward(tokens, mask)
        preds = self.out(preds)

        # Generate negative samples
        rnd_seq = torch.randint(0, self.num_nodes, seq.size(), device=self.device)
        neg_tokens = self.proj(self.embed(rnd_seq))
        pe = self.pe(neg_tokens)
        neg_tokens = neg_tokens + pe
        neg_preds = self.out(self.transformer(neg_tokens, mask))

        # Get middle nodes
        centers = preds[targets]        # S x B x d
        centers = centers.unsqueeze(1)  # S x 1 x B x d

        # Get surrounding nodes
        context_ids = torch.tensor([
            [
                j for j in range(i, i+self.context_window*2 + 1)
                if j != i+self.context_window # Don't eval z_i * z_i as it will always be 1
            ]
            for i in range(centers.size(0))
        ])
        context = preds[context_ids] # S x ctxt x B x d
        pos_scores = (centers * context).sum(dim=-1)
        pos_scores = pos_scores.view(-1)

        # Get negative sample preds
        neg_context = neg_preds[context_ids]
        neg_scores = (centers * neg_context).sum(dim=-1)
        neg_scores = neg_scores.view(-1)

        n2v_loss = (
            -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
            -torch.log(1-torch.sigmoid(neg_scores) + 1e-8).mean()
        )

        if torch.isnan(n2v_loss).any():
            print("Hm")

        # Make output embs more similar to transformer output
        target = preds[targets].detach() # Don't affect transformer, only embeddings
        out_emb = self.out_embs(seq[targets])
        recon_loss = self.mse_loss(out_emb, target)

        return n2v_loss, recon_loss

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