import time
import torch
from torch.optim import SparseAdam
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from in_memory_graph import Graph
from models.node2vec import AdvancedN2V

DEVICE = 2
BS = 32
WALK_LEN = 40
WALKS_PER_NODE = 10
CONTEXT_SIZE = 20
EPOCHS = 100


def train(model: AdvancedN2V):
    opt = Adam(model.parameters(), lr=1e-4)
    loader = model.loader(batch_size=BS, shuffle=True, num_workers=4)

    for e in range(1, EPOCHS+1):
        for i,(pos,neg) in enumerate(loader):
            opt.zero_grad()
            losses = model.loss(pos.to(DEVICE),neg.to(DEVICE))
            sum(losses).backward()
            opt.step()

            print(f'[{e}-{i}] n2v: {losses[0]:0.4f} recon: {losses[1]:0.4f}')

        torch.save(
            model.embedding.weight.data,
            'my_n2v.pt'
        )
        torch.save(
            model.out_embs.weight.data,
            'my_n2v_out.pt'
        )



if __name__ == '__main__':
    edge_index = Graph.load_ddi('train').edge_index

    model = AdvancedN2V(
        edge_index,
        512,
        walk_length=WALK_LEN,
        context_size=CONTEXT_SIZE,
        walks_per_node=WALKS_PER_NODE,
        sparse=False,
    ).to(DEVICE)

    train(model)