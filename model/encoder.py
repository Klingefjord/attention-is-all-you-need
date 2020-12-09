import torch.nn
from model.attention import MultiHeadAttention
from model.layers import FeedForwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, sub_layers):
        super(EncoderLayer, self).__init__()
        self.sub_layers = nn.ModuleList(sub_layers)
        self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.sub_layers])
        self.dropout = nn.ModuleList([nn.Dropout(p=0.1) for _ in self.sub_layers])
        
    def forward(self, x, **kwargs):
        for sublayer, layer_norm, dropout in zip(self.sub_layers, self.layer_norm_layers, self.dropout):
            x = layer_norm(x + dropout(sublayer(x, **kwargs)))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, N):
        super(Encoder, self).__init__()
        layer = self._encoder_layer(d_model, d_k, d_v, n_heads)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def _encoder_layer(d_model, d_k, d_v, n_heads):
        return EncoderLayer(d_model, [
            MultiHeadAttention(d_model, d_k, d_v, n_heads),
            PositionWiseFeedForwardNet(d_model, d_hidden)
        ])
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return x