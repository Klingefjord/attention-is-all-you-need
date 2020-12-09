import torch.nn
from model.layers import Embedding, PositionalEncodings
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, d_model, d_hidden, n_heads, N):
        """Main class for the transformer model"""
        super(Transformer, self).__init__()
        d_k = d_model//n_heads
        self.linear = nn.Linear(d_model, decoder_vocab_size, bias=False)
        # Share the weights between output linear layer and input & output embeddings
        self.input_embeddings = Embedding(encoder_vocab_size, d_model, self.linear.weight)
        self.output_embeddings = Embedding(decoder_vocab_size, d_model, self.linear.weight)
        self.positional_encodings_input = PositionalEncodings(d_model)
        self.positional_encodings_output = PositionalEncodings(d_model)
        self.encoder = Encoder(d_model, d_k, d_k, n_heads, N)
        self.decoder = Decoder(d_model, d_k, d_k, n_heads, N)
        
        # init using glorot rather than default kaiming since we're not using ReLUs
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, x, src_mask):
        """Only forward through the encoder"""
        x = self.positional_encodings_input(self.input_embeddings(x))
        return self.encoder(x, src_mask)
    
    def decode(self, x, src_mask, tgt_mask, encoder_output):
        """Only forward through the decoder (not including the final linear layer)"""
        x = self.positional_encodings_input(self.output_embeddings(x))
        return self.decoder(x, src_mask, tgt_mask, encoder_output)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, src_mask, tgt_mask, encoder_output)
        # Return logits here - final softmax is included in the loss function.
        return self.linear(decoder_output)