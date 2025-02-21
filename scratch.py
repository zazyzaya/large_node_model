import torch
from csr import CSR_np, CSR
import time

from models.masked_attention import MaskedAttentionEmb

ma = MaskedAttentionEmb(10,10, context_window=1)
out = ma.build_mask(10)
print(out)