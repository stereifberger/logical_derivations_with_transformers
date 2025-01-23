# Import necessary libraries
from imports import *  # Ensure that the necessary libraries like torch and nn modules are included

# Transformer Encoder Component
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len=500):
        super(TransformerEncoder, self).__init__()
        # Embedding layer to convert input tokens into embeddings of dimension `emb_dim`
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # Positional encoding to add position information to embeddings
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        # Creating a single encoder layer with specified parameters
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim, dropout=dropout)
        # Stacking the encoder layers to form a complete transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_mask):
        # Add positional encoding to the input embeddings
        embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        # Transpose the dimensions to match the expected input format for transformer (seq_len, batch_size, emb_dim)
        embedded = embedded.permute(1, 0, 2)
        # Pass the input through the transformer encoder with masking
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_mask)
        return output

# Transformer Decoder Component
class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len=500):
        super(TransformerDecoder, self).__init__()
        # Embedding layer to convert target tokens into embeddings of dimension `emb_dim`
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Positional encoding to add position information to embeddings
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        # Creating a single decoder layer with specified parameters
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim, dropout=dropout)
        # Stacking the decoder layers to form a complete transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Fully connected layer to map the decoder output to the final output dimension
        self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        # Add positional encoding to the target embeddings
        embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        # Transpose the dimensions to match the expected input format for transformer (seq_len, batch_size, emb_dim)
        embedded = embedded.permute(1, 0, 2)
        # Pass the combined input through the transformer decoder with masking
        output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # Pass the decoder output through the fully connected layer and transpose back dimensions
        output = self.fc_out(output)
        return output.permute(1, 0, 2)  # [batch_size, seq_len, output_dim]

# Sequence-to-sequence Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = encoder  # Encoder part of the transformer
        self.decoder = decoder  # Decoder part of the transformer
        self.device = device    # Device on which to deploy the model (CPU/GPU)

    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        # Pass the input sequence through the encoder to obtain memory representation
        memory = self.encoder(src, src_key_padding_mask)
        # Pass the target sequence and encoded memory through the decoder to obtain output
        output = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask)
        return output

# Function to create source key padding masks for the input sequences
def create_src_key_padding_mask(seq, pad_idx):
    # Returns a mask for the sequence with `True` where the tokens are padding tokens
    return (seq == pad_idx)

# Function to create target masks for the decoder input
def create_tgt_masks(tgt_seq, pad_idx):
    tgt_len = tgt_seq.size(1)
    # Generate subsequent mask for the target to prevent using future tokens
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_seq.device)
    # Create padding mask for the target sequence
    tgt_key_padding_mask = (tgt_seq == pad_idx)
    return tgt_mask, tgt_key_padding_mask