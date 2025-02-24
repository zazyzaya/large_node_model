import torch
from torch import nn
from torch.optim.adam import Adam

from models.masked_attention import MaskedAttentionEmb
from wiki_sampler import load_g_ddi

FNAME = 'masked_attn-2.pt'
DEVICE=3
BS = 3000

class LPNet(nn.Module):
    def __init__(self, in_dim, hidden, layers=4, device='cpu'):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, device=device),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden, hidden, device=device),
                    nn.ReLU()
                )
                for _ in range(layers-2)
            ],
            nn.Linear(hidden, in_dim, device=device)
        )

        self.device = device
        self.loss = nn.BCEWithLogitsLoss()
        self.opt = Adam(self.parameters(), lr=0.001)

    def forward(self, x, pos):
        self.opt.zero_grad()
        x = self.net(x)

        neg = torch.stack([
            torch.randint(0, pos[0].max(), pos[0].size()),
            torch.randint(0, pos[1].max(), pos[1].size())
        ])

        pos_pred = self.predict(x, pos)
        neg_pred = self.predict(x, neg)

        labels = torch.zeros((pos_pred.size(0) + neg_pred.size(0),1), device=self.device)
        labels[:pos_pred.size(0)] = 1
        loss = self.loss(
            torch.cat([pos_pred, neg_pred]),
            labels
        )
        loss.backward()
        self.opt.step()

        return loss.item()

    def predict(self, x, edges):
        src,dst = edges
        pred = (x[src] * x[dst]).sum(dim=1, keepdim=True)
        return pred


def train():
    args,kwargs,sd = torch.load(FNAME, weights_only=False, map_location=f'cpu')
    fm = MaskedAttentionEmb(*args, **kwargs)
    fm.load_state_dict(sd)
    fm.eval()

    embs = fm.embed(torch.arange(fm.num_nodes)).detach()
    del fm

    embs = embs.to(DEVICE)
    net = LPNet(embs.size(1), 512, device=DEVICE)
    g = load_g_ddi(partition='train')
    ei = torch.stack([
        torch.from_numpy(g['head']),
        torch.from_numpy(g['tail'])
    ])

    for e in range(10_000):
        loss = net.forward(embs, ei)
        print(f'{e}: {loss}')

    with torch.no_grad():
        new_embs= net.net(embs).detach()
    torch.save(new_embs, 'ft_embs.pt')

def get_rank(x, indices):
   vals = x[range(len(x)), indices]
   return (x > vals[:, None]).long().sum(1)

def print_stats(mrr):
    print("MRR: ", (1/mrr).mean().item() )
    print("MR: ", mrr.float().mean().item())
    print('hits@1: ', (mrr<=1).float().mean())
    print('hits@5: ', (mrr<=5).float().mean())
    print('hits@10: ', (mrr<=10).float().mean())
    print('hits@20: ', (mrr<=20).float().mean())
    print('hits@100: ', (mrr<=100).float().mean())
    print('hits@1000: ', (mrr<=1000).float().mean())
    print()


@torch.no_grad()
def test():
    g = load_g_ddi()
    batches = torch.arange(g['head'].shape[0]).split(BS)
    mrr = torch.tensor([], dtype=torch.long)
    embs = torch.load('ft_embs.pt', weights_only=False)

    for i,b in enumerate(batches):
        h = torch.from_numpy(g['head'][b])
        #r = torch.from_numpy(g['relation'][b]) + 4267
        t = g['tail'][b]

        preds = embs[h] @ embs.T
        ranks = get_rank(preds, t)
        mrr = torch.cat([mrr, ranks.to('cpu')])

        if i % 10 == 0 :
            print(f'{i+1}/{len(batches)}')
            print_stats(mrr+1)

    print("Final:")
    print_stats(mrr+1)

test()