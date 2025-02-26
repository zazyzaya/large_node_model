import torch
from torch_cluster import random_walk
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr

from wiki_sampler import ROOT

class Graph:
    def __init__(self, rowptr, col, num_nodes, edge_index):
        self.rowptr = rowptr
        self.col = col
        self.num_nodes = num_nodes
        self.num_rels = 0
        self.nodes_with_neighbors = torch.arange(num_nodes)
        self.edge_index = edge_index

    def load_ddi(partition):
        pt = torch.load(f'{ROOT}/ogbl_ddi/split/target/{partition}.pt', weights_only=False)
        head,tail = pt['edge'].T
        edge_index = torch.stack([
            torch.from_numpy(head),
            torch.from_numpy(tail)
        ])
        num_nodes = edge_index.max()+1

        row, col = sort_edge_index(edge_index, num_nodes=num_nodes).cpu()
        rowptr = index2ptr(row, num_nodes)

        return Graph(rowptr, col, num_nodes, edge_index)

    def rw(self, batch, walk_len, p=1, q=1):
        walks = torch.ops.torch_cluster.random_walk(
            self.rowptr, self.col, batch,
            walk_len, p, q
        )[0]

        return walks