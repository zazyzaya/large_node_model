import time
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from csr import CSR_np as CSR
from models.masked_attention import MaskedAttentionEmb

DEVICE = 2
MAX_STEPS = 1_000_000
MINI_BS = 64
BS = 64
WALK_LEN = 128

class Scheduler(LRScheduler):
    def get_lr(self):
        # Warmup period of 10k steps
        if self.last_epoch < 1_000:
            return [group['initial_lr'] * (self.last_epoch / 1_000)
                    for group in self.optimizer.param_groups]
        # Linear decay after that
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-1_000)/(MAX_STEPS-1_000)))
                    for group in self.optimizer.param_groups]

def minibatch(g, mb, model: MaskedAttentionEmb):
    walks = g.rw(mb, WALK_LEN, include_rel=False)
    loss = model(walks)
    loss.backward()

    return loss.item()

def train(g: CSR, model: MaskedAttentionEmb):
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
    while updates < MAX_STEPS:
        minibatches = starters[torch.randperm(starters.size(0))]
        minibatches = minibatches.split(MINI_BS)

        losses = []
        for i,mb in enumerate(minibatches):
            loss = minibatch(g, mb, model)

            if updates % 1000 == 999:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    'masked_attn.pt'
                )

            if updates % 10000 == 9999:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    f'masked_attn-{(updates+1)//10000}.pt'
                )

            losses.append(loss)
            steps += 1

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

                if updates > MAX_STEPS:
                    break

        e += 1
        '''
        # Don't do this for small dataset
        torch.save(
            (model.args, model.kwargs, model.state_dict()),
            f'bert-{e}.pt'
        )
        '''



if __name__ == '__main__':
    g = CSR().load('ddi_train')
    num_nodes = g.num_nodes
    num_rels = g.vocab_size - num_nodes

    # BERT Mini
    model = MaskedAttentionEmb(
        num_nodes, num_rels,
        device=DEVICE,
        layers=4,
        hidden_size=256,
        context_window=2
    )

    train(g,model)