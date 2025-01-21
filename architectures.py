# Import libraries
from imports import *

# Transformer Model Components
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len=500):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, src, src_mask):
        embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len=500):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, output_dim)
    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]
        output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)  # [batch_size, seq_len, output_dim]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        memory = self.encoder(src, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask)
        return output

# Function to create masks for transformer
def create_src_key_padding_mask(seq, pad_idx):
    return (seq == pad_idx)

def create_tgt_masks(tgt_seq, pad_idx):
    tgt_len = tgt_seq.size(1)
    # Corrected usage of size for mask creation
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_seq.device)
    tgt_key_padding_mask = (tgt_seq == pad_idx)
    return tgt_mask, tgt_key_padding_mask