import torch
from torch_cluster import random_walk
from torch_geometric.utils import add_remaining_self_loops

from wiki_sampler import ROOT

class Graph:
    def __init__(self, ei):
        self.ei = ei
        self.num_nodes = ei.max()+1
        self.num_rels = 1
        self.nodes_with_neighbors = ei[0].unique()

    def load_ddi(partition):
        pt = torch.load(f'{ROOT}/ogbl_ddi/split/target/{partition}.pt', weights_only=False)
        head,tail = pt['edge'].T
        ei = torch.stack([
            torch.from_numpy(head),
            torch.from_numpy(tail)
        ])
        ei = add_remaining_self_loops(ei)[0]
        return Graph(ei)

    def rw(self, batch, walk_len):
        walks = random_walk(
            self.ei[0], self.ei[1],
            batch, walk_length=walk_len
        )

        return walks