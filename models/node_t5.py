import torch
from torch import nn

from .large_node_model import PositionalEncoding

class NodeT5(nn.Module):
    def __init__(self, dict_size, device='cpu', hidden_size=768, layers=12, mask_rate=0.25, sentinal_count=32):
        super().__init__()

        self.device=device
        self.mask_rate = mask_rate
        self.sentinal_count = sentinal_count

        self.args = (dict_size,)
        self.kwargs = dict(
            hidden_size=hidden_size,
            layers=layers, mask_rate=mask_rate,
            sentinal_count=sentinal_count
        )

        # Special tokens
        self.SENTINAL = dict_size
        self.END = self.SENTINAL+sentinal_count + 1
        self.PAD = self.END + 1

        self.vector_store = nn.Embedding(self.PAD+1, hidden_size, device=device)
        self.pe = PositionalEncoding(hidden_size, device=device)

        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=hidden_size // 64,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=hidden_size*4,
            device=device
        )
        self.out = nn.Linear(
            hidden_size,
            self.PAD+1,
            device=device
        )

        self.loss = nn.CrossEntropyLoss()

    def _mask_seq(self, seq):
        mask = torch.rand(seq.size()) < self.mask_rate # B x S
        srcs, tgts = [],[]

        offset = 0
        longest_src = 0
        longest_tgt = 0
        for row in range(mask.size(0)):
            last_was_masked = False
            src,tgt = [],[]

            # Remove spans of masked nodes from src and add them
            # to dst after a tag marking which ones are which
            for col in range(mask.size(1)):
                if mask[row,col]:
                    if not last_was_masked:
                        src.append(self.SENTINAL+offset)
                        tgt.append(self.SENTINAL+offset)
                        last_was_masked = True
                        offset = (offset+1) % self.sentinal_count

                    tgt.append(seq[row,col].item())
                else:
                    src.append(seq[row,col].item())
                    last_was_masked = False

            # Add final <end> token to dst
            tgt.append(self.END)

            # Append list to list of lists
            longest_src = max(len(src), longest_src)
            longest_tgt = max(len(tgt), longest_tgt)

            srcs.append(src)
            tgts.append(tgt)

        # Convert to tensors
        src = torch.full((longest_src, len(srcs)), self.PAD)
        tgt = torch.full((longest_tgt, len(tgts)), self.PAD)

        for i,s in enumerate(srcs):
            src[torch.arange(len(s)), i] = torch.tensor(s)
        for i,t in enumerate(tgts):
            tgt[torch.arange(len(t)), i] = torch.tensor(t)

        return src,tgt

    def link_prediction(self, head,rel):
        '''
        For link prediction, input will always be
            src: Head,       Rel,   <sentinal>
            dst: <sentinal>, <end>, <pad>
            tgt: <sentinal>, Tail,  <end>
        '''
        src = torch.full((3,len(head)), self.SENTINAL) # 3 x B
        src[0, torch.arange(src.size(1))] = head
        src[1, torch.arange(src.size(1))] = rel

        dst = torch.tensor([[self.SENTINAL, self.PAD]]).T.repeat(1, len(head))
        preds = self.predict(src,dst) # 3 x B x d
        return preds[1]

    def predict(self, src,dst):
        src = src.to(self.device)
        dst = dst.to(self.device)

        src = self.pe(self.vector_store(src))
        dst = self.pe(self.vector_store(dst))

        mask = nn.Transformer.generate_square_subsequent_mask(dst.size(0))
        pred = self.transformer.forward(
            src.to(self.device),
            dst.to(self.device),
            tgt_mask=mask,
            tgt_is_causal=True
        )
        pred = self.out(pred)
        return pred

    def forward(self, sequences):
        '''
        Expect sequence of node ids
        S x B
        '''
        src,tgt = self._mask_seq(sequences)
        preds = self.predict(src,tgt)

        preds = preds.view(-1, preds.size(-1))
        tgt = tgt.flatten().to(self.device)
        loss = self.loss(preds, tgt)
        return loss
