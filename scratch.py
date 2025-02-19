import torch
from csr import CSR_np, CSR
import time

from models.node_t5 import NodeT5

csr = CSR_np().load('wikikg2_train')
model = NodeT5(csr.vocab_size, device=2)

ts = []
for _ in range(100):
    rw = csr.rw(torch.randint(0, csr.num_nodes, (32,)), 16)
    st = time.time()
    model._mask_seq(rw)
    en = time.time()
    print(en-st)
    ts.append(en-st)

print("avg: ", sum(ts) / len(ts))