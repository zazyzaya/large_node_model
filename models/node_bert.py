import torch
from torch import nn

from .tokenizer import NodeTokenizer

class NodeBERT(nn.Module):
    def __init__(self, num_nodes, num_rels, device='cpu', hidden_size=768, layers=12, mask_rate=0.15):
        super().__init__()

        self.device=device
        self.mask_rate = mask_rate

        self.args = (num_nodes,num_rels)
        self.kwargs = dict(
            hidden_size=hidden_size,
            layers=layers, mask_rate=mask_rate,
        )

        self.tokenizer = NodeTokenizer(num_nodes, num_rels, hidden_size, device=device)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, hidden_size//64,
                dim_feedforward=hidden_size*4,
                activation=nn.GELU(),
                device=self.device
            ),
            num_layers=layers
        )
        self.out = nn.Linear(
            hidden_size,
            self.tokenizer.MASK,
            device=device
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, sequences):
        '''
        Expect sequence of node ids
        S x B
        '''
        src,tgt,to_pred = self.tokenizer(sequences, mask=True)

        preds = self.out(self.transformer(src))

        preds = preds[to_pred]
        tgt = tgt.to(self.device)
        loss = self.loss(preds, tgt)
        return loss
