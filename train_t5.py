import time
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from csr import CSR_np as CSR
from models.node_t5 import NodeT5

DEVICE = 2
EPOCHS = 10
MINI_BS = 4
BS = 64
WALK_LEN = 32

class Scheduler(LRScheduler):
    def get_lr(self):
        return [(1 / ( (max(10_000, self.last_epoch) ** 0.5) *10))
                for group in self.optimizer.param_groups]

def minibatch(g, mb, model: NodeT5):
    walks = g.rw(mb, WALK_LEN)
    loss = model(walks)
    loss.backward()

    return loss.item()

def train(g: CSR, model: NodeT5):
    starters = g.nodes_with_neighbors
    opt = Adam(model.parameters(), lr=0.01)
    scheduler = Scheduler(opt)

    with open('log.txt', 'w+') as f:
        pass

    steps = 0
    updates = 0
    opt.zero_grad()
    st = time.time()

    for e in range(EPOCHS):
        minibatches = starters[torch.randperm(starters.size(0))]
        minibatches = minibatches.split(MINI_BS)

        losses = []
        for i,mb in enumerate(minibatches):

            loss = minibatch(g, mb, model)
            if i % 1000 == 999:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    't5.pt'
                )

            en = time.time()
            steps += 1
            losses.append(loss)

            if steps*MINI_BS >= BS:
                opt.step()
                scheduler.step()
                en = time.time()

                # Log epoch
                avg_loss = sum(losses) / len(losses)
                with open('log.txt', 'a') as f:
                    f.write(f'{avg_loss},{updates}\n')
                print(f'[{e}-({i}/{len(minibatches)})] {sum(losses)/len(losses)} ({en-st:0.2f}s)')

                # Reset accumulators
                st = time.time()
                losses = []
                opt.zero_grad()
                steps = 0
                updates += 1

        torch.save(
            (model.args, model.kwargs, model.state_dict()),
            f't5-{e+1}.pt'
        )


if __name__ == '__main__':
    g = CSR().load('wikikg2_train')

    # BERT Mini
    model = NodeT5(g.vocab_size, device=DEVICE, layers=4, hidden_size=256)

    train(g,model)