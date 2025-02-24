import torch
from ogb.linkproppred import Evaluator

from models.masked_attention import MaskedAttentionEmb
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
def eval(g, model: MaskedAttentionEmb):
    h = torch.from_numpy(g['head']).to(DEVICE)
    t = torch.from_numpy(g['tail']).to(DEVICE)
    pos = (model.out_embs(h) * model.out_embs(t)).sum(dim=1)

    h,t = torch.from_numpy(g['neg']).to(DEVICE)
    neg = (model.out_embs(h) * model.out_embs(t)).sum(dim=1)

    topk = neg.topk(100)[0]
    print('hits@1: \t', (pos >= topk[0]).float().mean().item())
    print('hits@5: \t', (pos >= topk[4]).float().mean().item())
    print('hits@10:\t', (pos >= topk[9]).float().mean().item())
    print('hits@20:\t', (pos >= topk[19]).float().mean().item())
    print('hits@100:\t', (pos >= topk[99]).float().mean().item())


if __name__ == '__main__':
    args,kwargs,sd = torch.load('masked_attn-old_loss.pt', weights_only=False, map_location=f'cuda:{DEVICE}')
    model = MaskedAttentionEmb(*args, **kwargs, device=DEVICE)
    model.eval()

    g = load_g_ddi('test')
    eval(g, model)
