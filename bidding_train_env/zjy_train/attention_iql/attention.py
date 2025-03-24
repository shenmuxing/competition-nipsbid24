# bidding_train_env.zjy_train.attention_iql.attention
# 用来将原始的不同长度的表征进行变换，得到同一长度的表征

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from copy import deepcopy

class TimeEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TimeEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If d_model is odd, pad the last dimension
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        # if time_step_index < self.pe.size(0):
        encoding = self.pe[:seq_len]  # [seq_len, 1, d_model]
        # else:
            # If time_step_index exceeds max_len, repeat the encoding
            # encoding = self.pe[-1].repeat(seq_len, 1, 1)
        return x + encoding

class AttentionIQL(nn.Module):
    def __init__(self, d_hist, d_curr, d_model, n_heads=8, dropout=0.1):
        """
        d_hist: Dimension of historical information
        d_curr: Dimension of current information
        d_model: Dimension of the model (output dimension)
        n_heads: Number of attention heads
        dropout: Dropout rate
        """
        super(AttentionIQL, self).__init__()
        self.d_model = d_model
        self.time_encoding = TimeEncoding(d_hist, max_len=48)

        # Define separate linear layers for Q, K, V
        self.linear_q = nn.Linear(d_curr, d_model, bias=False)
        self.linear_k = nn.Linear(d_hist, d_model, bias=False)
        self.linear_v = nn.Linear(d_hist, d_model, bias=False)

        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)

        # LayerNorm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP for regression
        self.mlp = nn.Sequential(
            # nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # Assuming regression output is single-dimensional
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear_q.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_k.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_v.weight, nonlinearity='relu')
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_attention_vector(self, history_informations, current_informations, time_step_index):
        """
        history_informations: Tensor of shape [T, 1, D_hist]
        current_informations: Tensor of shape [1, 1, D_curr]
        time_step_index: int, current time step index for time encoding
        """
        # Apply time encoding
        history_encoded = self.time_encoding(history_informations)  # [T, 1, D_model]
        current_encoded = current_informations  # Assuming already encoded or no encoding needed

        # Project to Q, K, V
        Q = self.linear_q(current_encoded)  # [1, 1, D_model]
        K = self.linear_k(history_encoded)  # [T, 1, D_model]
        V = self.linear_v(history_encoded)  # [T, 1, D_model], 注意：T是时间的意思，所以是[L,N,D]而不是[n,L,D]的形式。

        # Apply LayerNorm before attention (Pre-LN)
        Q = self.norm1(Q)
        K = self.norm1(K)
        V = self.norm1(V)

        # Compute attention
        attn_output, attn_weights = self.multihead_attn(Q, K, V)  # attn_output: [batch, seq, D_model]


        return attn_output  # [1,1,D_model]

    def forward(self, history_informations, current_informations, time_step_index):
        """
        history_informations: Tensor of shape [T, 1, D_hist]
        current_informations: Tensor of shape [1, 1, D_curr]
        time_step_index: int, current time step index for time encoding
        """
        attn_vector = self.get_attention_vector(history_informations, current_informations, time_step_index)  # [1,1,D_model]
        
        # Apply LayerNorm before MLP (Post-Attention LayerNorm with residual)
        attn_vector_norm = self.norm2(attn_vector)
        
        # Pass through MLP for regression
        action = self.mlp(attn_vector_norm)  # [1,1,1]
        
        return action.squeeze(-1)  # [1,1]

    def save_net(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        state_dict = {
            'time_encoding': self.time_encoding.state_dict(),
            'linear_q': self.linear_q.state_dict(),
            'linear_k': self.linear_k.state_dict(),
            'linear_v': self.linear_v.state_dict(),
            'multihead_attn': self.multihead_attn.state_dict(),
            'norm1': self.norm1.state_dict(),
            'norm2': self.norm2.state_dict(),
            'mlp': self.mlp.state_dict()
        }
        torch.save(state_dict, os.path.join(save_path, "attention_iql_model.pth"))

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        state_dict = torch.load(os.path.join(load_path, "attention_iql_model.pth"), map_location=device)
        
        self.time_encoding.load_state_dict(state_dict['time_encoding'])
        self.linear_q.load_state_dict(state_dict['linear_q'])
        self.linear_k.load_state_dict(state_dict['linear_k'])
        self.linear_v.load_state_dict(state_dict['linear_v'])
        self.multihead_attn.load_state_dict(state_dict['multihead_attn'])
        self.norm1.load_state_dict(state_dict['norm1'])
        self.norm2.load_state_dict(state_dict['norm2'])
        self.mlp.load_state_dict(state_dict['mlp'])