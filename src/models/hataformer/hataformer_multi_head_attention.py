#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/02/2025
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


class HATAFormerScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, bias_mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if bias_mask is not None:
            scores = scores + bias_mask

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class HATAFormerMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, local_window_size=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.local_window_size = local_window_size

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = HATAFormerScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.alpha_gate = nn.Parameter(torch.randn(1, n_heads, 1, 1))

        self.local_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))

    def _get_soft_local_bias(self, q_len, k_len, device):
        bias = torch.zeros(q_len, k_len, device=device)
        for i in range(q_len):
            start = max(0, i - self.local_window_size)
            end = min(k_len, i + self.local_window_size + 1)
            bias[i, start:end] = 1.0
        bias = bias.unsqueeze(0).unsqueeze(0)
        return bias

    def forward(self, query, key, value):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)

        query = self.w_q(query).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, v_len, self.n_heads, self.d_v).transpose(1, 2)

        if self.local_window_size and self.local_window_size > 1:
            soft_local_mask = self._get_soft_local_bias(q_len, k_len, query.device)
            local_bias = soft_local_mask * torch.sigmoid(self.local_bias)
        else:
            local_bias = None

        attn_output, attn_weights = self.attention(query, key, value, local_bias)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.w_o(attn_output)
        output = self.dropout(output)

        residual = query.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.layer_norm(output + residual)
        return output, attn_weights







