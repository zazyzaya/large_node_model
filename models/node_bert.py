import torch
from torch import nn

from .node_t5 import NodeT5
from .large_node_model import PositionalEncoding

class NodeBERT(nn.Module):
    def __init__(self, dict_size, device='cpu', hidden_size=768, layers=12, mask_rate=0.15):
        super().__init__()

        self.device=device
        self.mask_rate = mask_rate

        self.args = (dict_size,)
        self.kwargs = dict(
            hidden_size=hidden_size,
            layers=layers, mask_rate=mask_rate,
        )

        # Special tokens
        self.MASK = dict_size
        self.END = self.MASK + 1
        self.PAD = self.END + 1

        self.vector_store = nn.Embedding(self.PAD+1, hidden_size, device=device)
        self.pe = PositionalEncoding(hidden_size, device=device)

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
            self.PAD+1,
            device=device
        )

        self.loss = nn.CrossEntropyLoss()

    def _mask_seq(self, seq):
        '''
        seq: a B x S matrix of random walks through the graph

        Select 15% to mask,
        of those,   replace 80% with <MASK> tokens
                    replace 10% with random words
                    do nothing with the remaining 10%
        '''
        seq = seq.T # S x B

        to_predict = torch.rand(seq.size()) < self.mask_rate
        predicting = to_predict.nonzero()
        idxs = torch.randperm(predicting.size(0))
        to_mask = predicting[ idxs[:int(predicting.size(0) * 0.8)] ]
        to_replace = predicting[ idxs[int(predicting.size(0) * 0.8):int(predicting.size(0) * 0.9)] ]

        tgt = seq[to_predict]

        m_rows,m_cols = to_mask.T
        seq[m_rows, m_cols] = self.MASK
        r_rows,r_cols = to_replace.T
        seq[r_rows,r_cols] = torch.randint(0, self.MASK, r_rows.size())

        return seq,tgt,to_predict

    def link_prediction(self, head,rel):
        '''
        For link prediction, input will always be
            src: Head, Rel, <MASK>
        '''
        src = torch.full((3,len(head)), self.MASK) # 3 x B
        src[0, torch.arange(src.size(1))] = head
        src[1, torch.arange(src.size(1))] = rel

        preds = self.predict(src) # 3 x B x d
        return preds[-1]

    def predict(self, src):
        src = src.to(self.device)
        src = self.pe(self.vector_store(src))

        pred = self.transformer.forward(src)
        pred = self.out(pred)
        return pred

    def forward(self, sequences):
        '''
        Expect sequence of node ids
        S x B
        '''
        src,tgt,to_pred = self._mask_seq(sequences)
        preds = self.predict(src)

        preds = preds[to_pred]
        tgt = tgt.to(self.device)
        loss = self.loss(preds, tgt)
        return loss
