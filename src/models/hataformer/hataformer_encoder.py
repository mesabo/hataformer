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

import torch.nn as nn

from src.models.hataformer.hataformer_feed_forward import HATAFormerFFN
from src.models.hataformer.hataformer_multi_head_attention import HATAFormerMultiHeadAttention
from src.models.hataformer.hataformer_positional_encoding import HATAFormerPositionalEncoding


class HATAFormerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, local_window_size=None):
        super().__init__()
        self.attn = HATAFormerMultiHeadAttention(d_model, n_heads, dropout, local_window_size)
        self.feed_forward = HATAFormerFFN(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, metric_weights=None):
        residual = x
        x, _ = self.attn(x, x, x, metric_weights)
        x = self.dropout(x)
        x = self.layer_norm1(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm2(x + residual)
        return x


class HATAFormerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, max_len=5000, dropout=0.1,
                 encoding_type="temporal", local_window_size=None):
        super().__init__()
        self.positional_encoding = HATAFormerPositionalEncoding(d_model, max_len, dropout, encoding_type)
        self.encoder_layers = nn.ModuleList([
            HATAFormerEncoderBlock(d_model, n_heads, d_ff, dropout, local_window_size)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.layer_norm(x)


