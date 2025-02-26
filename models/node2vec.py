import torch
from torch import nn

class Node2Vec(nn.Module):
    def __init__(self, num_nodes, context_window=2, emb_size=128, device='cpu'):
        super().__init__()
        self.args = (num_nodes,)
        self.kwargs = dict(emb_size=emb_size, context_window=context_window)

        self.embed = nn.Embedding(num_nodes, emb_size, device=device)
        self.device = device
        self.context_window = context_window

    def forward(self, seq):
        '''
        seq: B x S
        '''
        seq = seq.to(self.device)

        center_ids = torch.arange(self.context_window, seq.size(1)-self.context_window-2)
        context_ids = torch.tensor([
            [
                j for j in range(i, i+self.context_window*2 + 1)
                if j != i+self.context_window # Don't eval z_i * z_i as it will always be 1
            ]
            for i in range(center_ids.size(0))
        ])
        centers = seq[:, center_ids]    # B x S
        contexts = seq[:, context_ids]  # B x S x ctxt

        c_embs = self.embed(centers).unsqueeze(2) # B x S x 1 x d
        ctxt_embs = self.embed(contexts)          # B x S x ctxt x d
        rnd_embs = self.embed(torch.randint(0, self.embed.num_embeddings, contexts.size(), device=self.device))

        pos_scores = (c_embs * ctxt_embs).sum(dim=-1).view(-1)
        neg_scores = (c_embs * rnd_embs).sum(dim=-1).view(-1)

        loss = (
            -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
            -torch.log(1-torch.sigmoid(neg_scores) + 1e-8).mean()
        )

        return loss