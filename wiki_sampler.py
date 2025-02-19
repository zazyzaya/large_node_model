import os
import numpy as np
import torch

from csr import CSR

ROOT='/mnt/raid10/kg_datasets/'

def load_csr(partition='train'):
    from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset

    if os.path.exists(f'graphs/wikikg2_{partition}.pt'):
        csr = CSR()
        csr.load(f'graphs/wikikg2_{partition}.pt')
        return csr

    dataset = PygLinkPropPredDataset('ogbl-wikikg2', root=ROOT)
    tr = dataset.get_edge_split()[partition]
    csr = CSR()
    csr.to_csr(tr['head'], tr['relation'], tr['tail'])
    csr.save(f'graphs/wikikg2_{partition}.pt')
    return csr

def load_g(partition='test'):
    if os.path.exists(f'graphs/wikikg2_{partition}.npz'):
        return np.load(f'graphs/wikikg2_{partition}.npz', mmap_mode='r')

    # For some reason, saved with torch.save but uses only np arrays?
    pt = torch.load(f'{ROOT}/ogbl_wikikg2/split/time/{partition}.pt', weights_only=False)
    np.savez(f'graphs/wikikg2_{partition}.npz', **pt)
    return load_g(partition)