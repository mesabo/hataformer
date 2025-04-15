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
import math


class HATAFormerPositionalEncoding(nn.Module):
    """
    Implements multiple positional encoding strategies, including:
    - Learnable
    - Sinusoidal
    - Temporal
    - Fourier-Light + Scalar Bias Modulation (new)
    - Normalized Temporal Projection (copyright-free alternative)

    Args:
        d_model (int): Dimensionality of the embedding space.
        d_t (int): Number of temporal features (e.g., hour, day, month).
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
        encoding_type (str): One of {'learnable', 'sinusoidal', 'temporal', 'fourier_scalar', 'temporal_proj'}.
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
            self.W1 = nn.Parameter(torch.randn(d_t, d_model))
            self.W2 = nn.Parameter(torch.randn(1, 1, d_model))

        elif encoding_type == "fourier_scalar":
            self.freqs = nn.Parameter(torch.randn(1, 1, d_t, 2))
            self.bias_proj = nn.Linear(2 * d_t, d_model)
            self.scalar_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        elif encoding_type == "temporal_proj":
            self.W1 = nn.Parameter(torch.randn(d_t, d_model))
            self.W2 = nn.Parameter(torch.randn(1, 1, d_model))

    def _generate_sinusoidal_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)

        if self.encoding_type == "learnable":
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(position_ids)

        elif self.encoding_type == "sinusoidal":
            if seq_len > self.pe.size(1):
                raise ValueError(f"Input sequence length ({seq_len}) exceeds the maximum allowed length.")
            x = x + self.pe[:, :seq_len, :]

        elif self.encoding_type == "temporal":
            temporal_features = x[:, :, -self.d_t:]
            assert temporal_features.shape[-1] == self.d_t, "Mismatch in temporal feature dimensions."
            lpe = torch.einsum("btd,df->btf", temporal_features, self.W1) + self.W2
            x = x + lpe

        elif self.encoding_type == "fourier_scalar":
            temporal_features = x[:, :, -self.d_t:]
            sincos = torch.cat([
                torch.sin(temporal_features * self.freqs[..., 0]),
                torch.cos(temporal_features * self.freqs[..., 1])
            ], dim=-1)
            scalar_mod = self.bias_proj(sincos) + self.scalar_bias
            x = x + scalar_mod

        elif self.encoding_type == "temporal_proj":
            temporal = x[:, :, -self.d_t:]
            normed = (temporal - temporal.mean(dim=1, keepdim=True)) / (temporal.std(dim=1, keepdim=True) + 1e-5)
            temp_proj = torch.tanh(normed @ self.W1)
            temp_bias = self.W2.expand(temp_proj.size())
            x = x + temp_proj + temp_bias

        return self.dropout(x)



