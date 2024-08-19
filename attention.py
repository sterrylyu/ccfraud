import torch
import torch.nn as nn
import torch.nn.functional as F

# svm before 
class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, dropout=0.1):
        super(AttentionModel, self).__init__()

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)

        # Feedforward layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation action and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # shape of x  (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Transposing input for matching the requirement of Multihead Attention (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)

        # Attention layer output
        attn_output, _ = self.attention(x, x, x)

        # Transferring output of attention layer to original shape (batch_size, seq_len, input_dim)
        attn_output = attn_output.permute(1, 0, 2)

        # Pooling output of attention layer
        pooled_output = torch.mean(attn_output, dim=1)

        # Feedforward network
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
