import time
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from in_memory_graph import Graph
from models.masked_attention import MaskedAttentionEmb

DEVICE = 2
EPOCHS = 100
MINI_BS = 512
BS = 512
WALK_LEN = 80 # Same as in original n2v paper

def minibatch(g, mb, model: MaskedAttentionEmb):
    walks = g.rw(mb, WALK_LEN)
    n2v_loss,recon_loss = model(walks)
    (n2v_loss + recon_loss).backward()

    return n2v_loss.item(), recon_loss.item()

def train(g: Graph, model: MaskedAttentionEmb):
    starters = g.nodes_with_neighbors
    opt = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    with open('log.txt', 'w+') as f:
        pass

    steps = 0
    updates = 0
    opt.zero_grad()
    st = time.time()

    e = 0
    for e in range(EPOCHS):
        minibatches = starters[torch.randperm(starters.size(0))]
        minibatches = minibatches.split(MINI_BS)

        for i,mb in enumerate(minibatches):
            st = time.time()
            opt.zero_grad()
            n2v_loss,recon_loss = minibatch(g, mb, model)
            opt.step()
            en = time.time()

            # Log epoch
            with open('log.txt', 'a') as f:
                f.write(f'{n2v_loss},{recon_loss},{updates}\n')
            print(f'[{updates}-{e}] n2v: {n2v_loss:0.4f} recon: {recon_loss:0.4f} ({en-st:0.2f}s)')

            updates += 1

        torch.save(
            (model.args, model.kwargs, model.state_dict()),
            'masked_attn.pt'
        )


if __name__ == '__main__':
    g = Graph.load_ddi('train')
    num_nodes = g.num_nodes
    num_rels = g.num_rels

    # BERT Mini
    model = MaskedAttentionEmb(
        num_nodes, num_rels,
        device=DEVICE,
        layers=4,
        hidden_size=256,
        context_window=5,
    )

    train(g,model)