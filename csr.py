from collections import defaultdict
from random import choice, randint
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

class CSR:
    def __init__(self):
        self.idx = []
        self.ptr = [0]
        self.rel = []

    @property
    def num_nodes(self):
        return len(self.ptr) - 1

    @property
    def nodes_with_neighbors(self):
        ret = []
        for i in range(len(self.ptr)-1):
            if self.ptr[i] != self.ptr[i+1]:
                ret.append(i)
        return torch.tensor(ret)

    @property
    def vocab_size(self):
        # +1 for preprocessing error
        # +1 for "self-loop" edge
        return self.rel.max() + 2

    def to_csr(self, head, rel, tail):
        self.idx = []
        self.ptr = [0]
        self.rel = []

        sorted = defaultdict(lambda : [[],[]])
        max_node = max(head.max(), tail.max())
        max_rel = rel.max()

        for i in tqdm(range(len(head))):
            r,t = sorted[head[i].item()]
            r.append(rel[i].item()); t.append(tail[i].item())

        for i in tqdm(range(max_node+1)):
            r,t = sorted[i]
            self.ptr.append(self.ptr[-1] + len(r))
            self.idx += t
            self.rel += r

            # Add self-loop if no neighbors
            if not t:
                self.idx += [i]
                self.rel += [max_rel+1]
                self.ptr[-1] += 1

        self.idx = torch.tensor(self.idx)
        self.ptr = torch.tensor(self.ptr)
        self.rel = torch.tensor(self.rel) + max_node

    def _get_one(self, idx):
        st = self.ptr[idx]
        en = self.ptr[idx+1]

        return self.idx[st:en], self.rel[st:en]+1 # Error in preprocessing

    def get(self, idx):
        if isinstance(idx, Iterable):
            return zip(*[self._get_one(i) for i in idx])
        return self._get_one(idx)

    def rw(self, batch, walk_len):
        walks = [batch]

        for _ in range(walk_len):
            neighbors, rels = self.get(batch)
            next_r = torch.zeros(batch.size(), dtype=torch.long)
            next_n = torch.zeros(batch.size(), dtype=torch.long)

            for j in range(len(neighbors)):
                next_r[j] = choice(rels[j])
                next_n[j] = choice(neighbors[j])

            walks += [next_r, next_n]
            batch = next_n

        return torch.stack(walks, dim=1) # B x S

    def save(self, fname):
        torch.save((self.idx, self.ptr, self.rel), fname)
    def load(self, fname):
        idx,ptr,rel = torch.load(fname)
        self.idx = idx.long(); self.ptr = ptr.long(); self.rel = rel.long()

class CSR_np(CSR):
    def from_torch(csr: CSR, fname='wikikg2_train'):
        np_csr = CSR_np()
        idx = csr.idx.numpy()
        ptr = csr.ptr.numpy()
        rel = csr.rel.numpy()

        np.savez(
            f'graphs/{fname}.npz',
            idx=idx, ptr=ptr, rel=rel
        )

        del idx, ptr, rel
        np_csr.load(fname)
        return np_csr


    def _rw_from_one(self, st, walk_len):
        walk = [st]
        for _ in range(walk_len):
            neigh, rels = self._get_one(walk[-1])
            idx = randint(0, len(neigh)-1)
            walk.append(rels[idx])
            walk.append(neigh[idx])

        return walk

    def rw(self, batch, walk_len):
        walks = []

        # TODO threaded?
        for b in batch:
            walks.append(self._rw_from_one(b, walk_len))

        return torch.tensor(walks) # B x S

    def load(self, fname):
        files = np.load(f'graphs/{fname}.npz', mmap_mode='r')
        self.idx = files['idx']
        self.ptr = files['ptr']
        self.rel = files['rel']
        return self