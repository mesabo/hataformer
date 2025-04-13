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


class HATAFormerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, local_window_size=None, bias_type="learned"):
        super().__init__()
        self.attn = HATAFormerMultiHeadAttention(d_model, n_heads, dropout, local_window_size, bias_type)
        self.cross_attn = HATAFormerMultiHeadAttention(d_model, n_heads, dropout, local_window_size, bias_type)
        self.feed_forward = HATAFormerFFN(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, metric_weights=None):
        residual = tgt
        x, self_attn_weights = self.attn(tgt, tgt, tgt, metric_weights)
        x = self.dropout(x)
        x = self.layer_norm1(x + residual)

        residual = x
        x, cross_attn_weights = self.cross_attn(x, memory, memory, metric_weights)
        x = self.dropout(x)
        x = self.layer_norm2(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm3(x + residual)

        return x, self_attn_weights, cross_attn_weights


class HATAFormerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, max_len=5000, dropout=0.1,
                 encoding_type="temporal", local_window_size=None, bias_type="learned"):
        super().__init__()
        self.positional_encoding = HATAFormerPositionalEncoding(d_model, max_len, dropout, encoding_type)
        self.decoder_layers = nn.ModuleList([
            HATAFormerDecoderBlock(d_model, n_heads, d_ff, dropout, local_window_size, bias_type) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, metric_weights=None):
        tgt = self.positional_encoding(tgt)
        all_self_attn, all_cross_attn = [], []

        for layer in self.decoder_layers:
            tgt, self_attn, cross_attn = layer(tgt, memory, metric_weights)
            all_self_attn.append(self_attn)
            all_cross_attn.append(cross_attn)

        return self.layer_norm(tgt), all_self_attn, all_cross_attn
