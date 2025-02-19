import torch

from models.node_t5 import NodeT5
from wiki_sampler import load_g

DEVICE=3
BS = 256

def get_rank(x, indices):
   vals = x[range(len(x)), indices]
   return (x > vals[:, None]).long().sum(1)

@torch.no_grad()
def eval(g, model: NodeT5):
    batches = torch.arange(g['head'].shape[0]).split(BS)
    mrr = torch.tensor([], dtype=torch.long)
    for b in batches:
        h = torch.from_numpy(g['head'][b])
        r = torch.from_numpy(g['relation'][b]) + 2_500_604
        t = g['tail'][b]

        preds = model.link_prediction(h,r)
        ranks = get_rank(preds, t)
        mrr = torch.cat([mrr, ranks.to('cpu')])

        print("MRR: ", (1/mrr).mean().item() )
        print("MR: ", mrr.float().mean().item())
        print('hits@1: ', (mrr<=1).float().mean())
        print('hits@5: ', (mrr<=5).float().mean())
        print('hits@10: ', (mrr<=10).float().mean())

if __name__ == '__main__':
    args,kwargs,sd = torch.load('t5.pt', weights_only=False, map_location=f'cuda:{DEVICE}')
    model = NodeT5(*args, **kwargs, device=DEVICE)
    model.eval()

    g = load_g('test')
    eval(g, model)