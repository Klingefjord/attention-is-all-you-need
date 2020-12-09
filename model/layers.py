import torch.nn
import numpy as np

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_hidden, **kwargs):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x, **kwargs):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, weights=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if weights is not None: self.embedding.weight = weights
        
    def forward(self, x):
        return self.embedding(x)

class PositionalEncodings(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncodings, self).__init__()
        self.register_buffer('pos_encodings', self._get_positional_encodings(d_model).unsqueeze(0))

    
    def _sinusoid(self, pos, i, d_model):
        return pos / (10000**(2*i/d_model))

    def _get_positional_encodings(self, d_model):
        encodings = torch.zeros(d_model, d_model)
        for pos_idx, _ in enumerate(encodings[:,0]):
            for i, _ in enumerate(encodings[0,:]):
                encodings[pos_idx, i] = np.cos(self._sinusoid(pos_idx, i, d_model)) if i % 2 == 0 else np.sin(self._sinusoid(pos_idx, i, d_model))
            
    return encodings
        
    def forward(self, x):
        return x + self.pos_encodings[:, :x.shape[1], :x.shape[2]].clone().detach()