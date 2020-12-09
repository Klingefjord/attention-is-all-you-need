import torch
import torch.nn

from model.attention import MultiHeadAttention
from model.layers import FeedForwardLayer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, sub_layers):
        super(DecoderLayer, self).__init__()
        self.sub_layers = nn.ModuleList(sub_layers)
        self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.sub_layers])
        self.dropout = nn.ModuleList([nn.Dropout(p=0.1) for _ in self.sub_layers])
    
    def forward(self, x, **kwargs):
        for sublayer, layer_norm, dropout in zip(self.sub_layers, self.layer_norm_layers, self.dropout):        
            x = layer_norm(x + dropout(sublayer(x, **kwargs)))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, N):
        super(Decoder, self).__init__()
        layer = self._decoder_layer(d_model, d_k, d_v, n_heads)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    
    def _decoder_layer(self, d_model, d_k, d_v, n_heads):
        return DecoderLayer(d_model, [
            MultiHeadAttention(d_model, d_k, d_v, n_heads), # self-attention
            MultiHeadAttention(d_model, d_k, d_v, n_heads), # encoder-decoder attention
            FeedForwardLayer(d_model, d_hidden)    
        ])
        
    def forward(self, x, src_mask, tgt_mask, encoder_output):
        for i, layer in enumerate(self.layers):
            if i == 0:
                # self-attention
                x = layer(x, mask=tgt_mask)
            elif i == 1: 
                # encoder-decoder attention
                x = layer(x, mask=src_mask, encoder_output=encoder_output)
            else:
                # linear
                x = layer(x)
        return x