import time
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from in_memory_graph import Graph
from models.masked_attention import MaskedAttentionEmb

DEVICE = 2
MAX_STEPS = 1_000_000
MINI_BS = 512
BS = 512
WALK_LEN = 80 # Same as in original n2v paper

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
    walks = g.rw(mb, WALK_LEN)
    n2v_loss,recon_loss = model(walks)
    (n2v_loss + recon_loss).backward()

    return n2v_loss.item(), recon_loss.item()

def train(g: Graph, model: MaskedAttentionEmb):
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

        n2v_losses = []
        recon_losses = []
        for i,mb in enumerate(minibatches):
            n2v_loss,recon_loss = minibatch(g, mb, model)

            if updates % 100 == 99:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    'masked_attn.pt'
                )

            if updates % 10000 == 9999:
                torch.save(
                    (model.args, model.kwargs, model.state_dict()),
                    f'masked_attn-{(updates+1)//10000}.pt'
                )

            n2v_losses.append(n2v_loss)
            recon_losses.append(recon_loss)
            steps += 1

            if steps*MINI_BS >= BS:
                opt.step()
                scheduler.step()
                en = time.time()

                # Log epoch
                avg_n2v_loss = sum(n2v_losses) / len(n2v_losses)
                avg_recon_loss = sum(recon_losses) / len(recon_losses)
                with open('log.txt', 'a') as f:
                    f.write(f'{avg_n2v_loss},{avg_recon_loss},{updates}\n')
                print(f'[{updates}-{e}] n2v: {avg_n2v_loss:0.4f} recon: {avg_recon_loss:0.4f} (lr: {[g["lr"] for g in opt.param_groups][0]:0.2e}, {en-st:0.2f}s)')

                # Reset accumulators
                st = time.time()
                n2v_losses = []
                recon_losses = []
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
    g = Graph.load_ddi('train')
    num_nodes = g.num_nodes
    num_rels = g.num_rels

    # BERT Mini
    model = MaskedAttentionEmb(
        num_nodes, num_rels,
        device=DEVICE,
        layers=4,
        hidden_size=256,
        context_window=2,
    )

    train(g,model)