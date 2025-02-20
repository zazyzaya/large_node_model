import time
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from csr import CSR_np as CSR
from models.node_bert import NodeBERT

DEVICE = 2
MAX_STEPS = 1_000_000
MINI_BS = 8
BS = 32
WALK_LEN = 32

class Scheduler(LRScheduler):
    def get_lr(self):
        # Warmup period of 10k steps
        if self.last_epoch < 10_000:
            return [group['initial_lr'] * (self.last_epoch / 10_000)
                    for group in self.optimizer.param_groups]
        # Linear decay after that
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-10_000)/(MAX_STEPS-10_000)))
                    for group in self.optimizer.param_groups]

def minibatch(g, mb, model: NodeBERT):
    walks = g.rw(mb, WALK_LEN)
    loss = model(walks)
    loss.backward()

    return loss.item()

def train(g: CSR, model: NodeBERT):
    starters = g.nodes_with_neighbors
    opt = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = Scheduler(opt)

    with open('log.txt', 'w+') as f:
        pass

    steps = 0
    updates = 0
    opt.zero_grad()
    st = time.time()

    e = 0
    while steps < MAX_STEPS:
        minibatches = starters[torch.randperm(starters.size(0))]
        minibatches = minibatches.split(MINI_BS)

        losses = []
        for i,mb in enumerate(minibatches):
            loss = minibatch(g, mb, model)
            if steps % 1000 == 999:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    'bert.pt'
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
                print(f'[{e}-({i}/{len(minibatches)})] {sum(losses)/len(losses)} (lr: {[g["lr"] for g in opt.param_groups][0]:0.2e}, {en-st:0.2f}s)')

                # Reset accumulators
                st = time.time()
                losses = []
                opt.zero_grad()
                steps = 0
                updates += 1

        e += 1
        torch.save(
            (model.args, model.kwargs, model.state_dict()),
            f'bert-{e}.pt'
        )



if __name__ == '__main__':
    g = CSR().load('wikikg2_train')

    # BERT Mini
    model = NodeBERT(
        g.vocab_size,
        device=DEVICE,
        layers=4,
        hidden_size=256
    )

    train(g,model)