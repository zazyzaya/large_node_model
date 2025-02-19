import torch
from torch import nn

from .large_node_model import PositionalEncoding

class NodeT5(nn.Module):
    def __init__(self, dict_size, device='cpu', hidden_dim=4, inner_dim=16, heads=6, layers=6, mask_rate=0.25, sentinal_count=32):
        super().__init__()

        self.device=device
        self.mask_rate = mask_rate
        self.sentinal_count = sentinal_count

        self.args = (dict_size,)
        self.kwargs = dict(
            hidden_dim=hidden_dim,
            inner_dim=inner_dim, heads=heads,
            layers=layers, mask_rate=mask_rate,
            sentinal_count=sentinal_count
        )

        # Special tokens
        self.SENTINAL = dict_size
        self.END = self.SENTINAL+sentinal_count + 1
        self.PAD = self.END + 1

        self.vector_store = nn.Embedding(self.PAD+1, hidden_dim*heads, device=device)
        self.pe = PositionalEncoding(hidden_dim*heads, device=device)

        self.transformer = nn.Transformer(
            d_model=hidden_dim*heads,
            nhead=heads,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=inner_dim*heads,
            device=device
        )
        self.out = nn.Linear(
            hidden_dim*heads,
            self.PAD+1,
            device=device
        )

        self.loss = nn.CrossEntropyLoss()

    def _mask_seq(self, seq):
        mask = torch.rand(seq.size()) < self.mask_rate # B x S
        srcs, dsts, tgts = [],[],[]

        offset = 0
        longest_src = 0
        longest_dst = 0
        for row in range(mask.size(0)):
            last_was_masked = False
            src,dst,tgt = [],[],[]

            # Remove spans of masked nodes from src and add them
            # to dst after a tag marking which ones are which
            for col in range(mask.size(1)):
                if mask[row,col]:
                    if not last_was_masked:
                        dst.append(self.SENTINAL+offset)
                        src.append(self.SENTINAL+offset)
                        tgt.append(self.SENTINAL+offset)
                        last_was_masked = True
                        offset = (offset+1) % self.sentinal_count

                    tgt.append(seq[row,col].item())
                else:
                    src.append(seq[row,col].item())
                    last_was_masked = False

            # Add final <end> token to dst
            dst.append(self.END)
            tgt.append(self.END)

            # Append list to list of lists
            longest_src = max(len(src), longest_src)
            longest_dst = max(len(dst), len(tgt), longest_dst)

            srcs.append(src)
            dsts.append(dst)
            tgts.append(tgt)

        # Convert to tensors
        src = torch.full((longest_src, len(srcs)), self.PAD)
        dst = torch.full((longest_dst, len(dsts)), self.PAD)
        tgt = torch.full((longest_dst, len(tgts)), self.PAD)

        for i,s in enumerate(srcs):
            src[torch.arange(len(s)), i] = torch.tensor(s)
        for i,d in enumerate(dsts):
            dst[torch.arange(len(d)), i] = torch.tensor(d)
        for i,t in enumerate(tgts):
            tgt[torch.arange(len(t)), i] = torch.tensor(t)

        return src,dst,tgt

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
        src,dst,tgt = self._mask_seq(sequences)
        preds = self.predict(src,tgt)

        preds = preds.view(-1, preds.size(-1))
        tgt = tgt.flatten().to(self.device)
        loss = self.loss(preds, tgt)
        return loss