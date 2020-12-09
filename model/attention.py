import math
import torch
import torch.nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, is_encoder_decoder=False, mask=None, **kwargs):    
        super(ScaledDotProductAttention, self).__init__()
        self.query_matrix = nn.Linear(d_model, d_k, bias=False)
        self.key_matrix = nn.Linear(d_model, d_k, bias=False)
        self.value_matrix = nn.Linear(d_model, d_v, bias=False)
        self.softmax = nn.Softmax()
        self.divider = math.sqrt(d_k)
        self.is_encoder_decoder = is_encoder_decoder
        
    # In the case of encoder-decoder attention, the key and value come from the
    # encoder outputs. Mask is either a src_mask used in encoder for preventing
    # attending <PAD> tokens (and in encoder-decoder attention in decoder), 
    # or a tgt_mask in decoder self-attention
    def forward(self, x, mask=None, encoder_output=None):
        Q = self.query_matrix(x)
        K = self.key_matrix(encoder_output) if encoder_output is not None else self.key_matrix(x)
        V = self.value_matrix(encoder_output) if encoder_output is not None else self.value_matrix(x)
        # swap last and second-to-last dimensions while ignoring dim 0 (batch-size)
        scores = Q @ K.transpose(-2, -1) / self.divider
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)
        return self.softmax(scores) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention(d_model, d_k, d_v, **kwargs) for _ in range(n_heads)])
        self.weight_matrix = nn.Linear(n_heads * d_k, d_model, bias=False)
        
    def forward(self, x, **kwargs):
        outputs = [head(x, **kwargs) for head in self.heads]
        return self.weight_matrix(torch.cat(outputs, dim=-1))