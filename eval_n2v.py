import torch
from ogb.linkproppred import Evaluator

from models.node2vec import Node2Vec
from wiki_sampler import load_g, load_g_ddi

DEVICE=3
BS = 3000

evaluator = Evaluator('ogbl-ddi')
evaluator.K = 20

def get_ogb_stats(pos_test, neg_test):
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test,
        'y_pred_neg': neg_test,
    })['hits@20']
    print(test_hits)

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
def eval(g, embed):
    # Input embeddings
    h = torch.from_numpy(g['head'])
    t = torch.from_numpy(g['tail'])
    pos = (embed[h] * embed[t]).sum(dim=1)

    h,t = torch.from_numpy(g['neg'])
    neg = (embed[h] * embed[t]).sum(dim=1)

    topk = neg.topk(100)[0]
    print(f'hits@1:  ({topk[0]:0.2f})\t', (pos >= topk[0]).float().mean().item())
    print(f'hits@5:  ({topk[4]:0.2f})\t', (pos >= topk[4]).float().mean().item())
    print(f'hits@10: ({topk[9]:0.2f})\t', (pos >= topk[9]).float().mean().item())
    print(f'hits@20: ({topk[19]:0.2f})\t', (pos >= topk[19]).float().mean().item())
    print(f'hits@100: ({topk[99]:0.2f})\t', (pos >= topk[99]).float().mean().item())
    print()


if __name__ == '__main__':
    emb = torch.load('n2v.pt')

    g = load_g_ddi('test')
    eval(g, emb)
