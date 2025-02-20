import torch

from models.node_bert import NodeBERT
from wiki_sampler import load_g

DEVICE=3
BS = 1000

def get_rank(x, indices):
   vals = x[range(len(x)), indices]
   return (x > vals[:, None]).long().sum(1)

def print_stats(mrr):
    print("MRR: ", (1/mrr).mean().item() )
    print("MR: ", mrr.float().mean().item())
    print('hits@1: ', (mrr<=1).float().mean())
    print('hits@5: ', (mrr<=5).float().mean())
    print('hits@10: ', (mrr<=10).float().mean())
    print('hits@100: ', (mrr<=100).float().mean())
    print('hits@1000: ', (mrr<=1000).float().mean())
    print()


@torch.no_grad()
def eval(g, model: NodeBERT):
    batches = torch.arange(g['head'].shape[0]).split(BS)
    mrr = torch.tensor([], dtype=torch.long)
    for i,b in enumerate(batches):
        h = torch.from_numpy(g['head'][b])
        r = torch.from_numpy(g['relation'][b]) + 2_500_604
        t = g['tail'][b]

        preds = model.link_prediction(h,r)
        ranks = get_rank(preds, t)
        mrr = torch.cat([mrr, ranks.to('cpu')])

        if i % 10 == 0 :
            print(f'{i+1}/{len(batches)}')
            print_stats(mrr)

    print("Final:")
    print_stats(mrr)

if __name__ == '__main__':
    args,kwargs,sd = torch.load('bert.pt', weights_only=False, map_location=f'cuda:{DEVICE}')
    model = NodeBERT(*args, **kwargs, device=DEVICE)
    model.eval()

    g = load_g('test')
    eval(g, model)
