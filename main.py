import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, hidden_size, num_layers, dropout):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        # x is a tensor of shape (sequence_length, batch_size)
        embedding = self.embedding(x)
        positional_encoding = self.positional_encoding(embedding)
        transformer_output = self.transformer_encoder(positional_encoding)
        output = self.fc(transformer_output)

        # output is a tensor of shape (sequence_length, batch_size, vocab_size)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_sequence_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(max_sequence_length, embedding_size)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        positional_encoding[:, 0::2] = sin_term
        positional_encoding[:, 1::2] = cos_term
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0), :]
        x = self.dropout(x)
        return x
