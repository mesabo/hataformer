#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/15/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import torch
import torch.nn as nn
class HATAFormerPositionalEncoding(nn.Module):
    """
    Implements learnable positional encoding with optional sinusoidal fallback.

    Args:
        d_model (int): Dimensionality of the embedding space.
        d_t (int): Number of temporal features (e.g., hour, day, month).
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
        encoding_type (str): One of {'learnable', 'sinusoidal', 'temporal'}.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1, encoding_type="sinusoidal", d_t=9):
        super(HATAFormerPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type
        self.d_t = d_t
        self.d_model = d_model

        if encoding_type == "learnable":
            self.position_embeddings = nn.Embedding(max_len, d_model)

        elif encoding_type == "sinusoidal":
            self.register_buffer("pe", self._generate_sinusoidal_pe(d_model, max_len))

        elif encoding_type == "temporal":
            # Ensure W1 correctly projects temporal features to d_model space
            self.W1 = nn.Parameter(torch.randn(d_t, d_model))  # (d_t, d_model) instead of (d_model, d_t)
            self.W2 = nn.Parameter(torch.randn(1, 1, d_model))  # Bias term for broadcasting

    def _generate_sinusoidal_pe(self, d_model, max_len):
        """
        Generate sinusoidal positional encodings.

        Args:
            d_model (int): Dimensionality of the embedding space.
            max_len (int): Maximum sequence length.

        Returns:
            torch.Tensor: Positional encodings of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Add positional encoding or embeddings to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        seq_len = x.size(1)

        if self.encoding_type == "learnable":
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(position_ids)

        elif self.encoding_type == "sinusoidal":
            if seq_len > self.pe.size(1):
                raise ValueError(f"Input sequence length ({seq_len}) exceeds the maximum allowed length.")
            x = x + self.pe[:, :seq_len, :]

        elif self.encoding_type == "temporal":
            # Extract the last d_t columns as temporal features
            temporal_features = x[:, :, -self.d_t:]  # Shape: (batch_size, seq_len, d_t=(hour,day,month))

            assert temporal_features.shape[-1] == self.d_t, "Mismatch in temporal feature dimensions."

            # Compute Learnable Positional Encoding: LPE = φ(τ) W1^T + W2
            lpe = torch.einsum("btd,df->btf", temporal_features, self.W1) + self.W2  # Fixed einsum shape
            x = x + lpe

        return self.dropout(x)

