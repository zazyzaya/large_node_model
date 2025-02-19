import time
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from csr import CSR_np as CSR
from models.node_t5 import NodeT5

DEVICE = 2
EPOCHS = 10
BS = 50
WALK_LEN = 6

class Scheduler(LRScheduler):
    def get_lr(self):
        return [(1 / ( (max(10_000, self.last_epoch) ** 0.5) *10))
                for group in self.optimizer.param_groups]

def minibatch(opt, sched, g, b, model: NodeT5):
    opt.zero_grad()
    walks = g.rw(b, WALK_LEN)
    loss = model(walks)
    loss.backward()
    opt.step()
    sched.step()

    return loss.item()

def train(g: CSR, model: NodeT5):
    starters = g.nodes_with_neighbors
    opt = Adam(model.parameters(), lr=0.01)
    scheduler = Scheduler(opt)

    with open('log.txt', 'w+') as f:
        pass

    for e in range(EPOCHS):
        batches = starters[torch.randperm(starters.size(0))]
        batches = batches.split(BS)

        losses = []

        for i,b in enumerate(batches):
            st = time.time()
            loss = minibatch(opt, scheduler, g, b, model)
            if i % 1000 == 999:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    't5.pt'
                )

            with open('log.txt', 'a+') as f:
                f.write(f'{loss},{i}\n')

            en = time.time()

            if i % 10 == 0:
                print(f'[{e}-({i}/{len(batches)})] {loss} ({en-st:0.2f}s)')

        torch.save(
            (model.args, model.kwargs, model.state_dict()),
            f't5-{e+1}.pt'
        )


if __name__ == '__main__':
    g = CSR().load('wikikg2_train')
    model = NodeT5(g.vocab_size, device=DEVICE)
    train(g,model)