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

from src.models.hataformer.hataformer_decoder import HATAFormerDecoder
from src.models.hataformer.hataformer_encoder import HATAFormerEncoder


class HATAFormerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, d_ff, num_encoder_layers, num_decoder_layers,
                 max_len=5000, dropout=0.1, encoding_type="temporal", local_window_size=None,
                 forecast_horizon=1, bias_type="learned"):
        super(HATAFormerModel, self).__init__()
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(1, d_model)

        self.encoder = HATAFormerEncoder(
            d_model, n_heads, d_ff, num_encoder_layers, max_len, dropout,
            encoding_type, local_window_size, bias_type
        )
        self.decoder = HATAFormerDecoder(
            d_model, n_heads, d_ff, num_decoder_layers, max_len, dropout,
            encoding_type, local_window_size, bias_type
        )

        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, tgt, metric_weights=None):
        # Encode inputs
        src = self.input_embedding(src)
        memory = self.encoder(src, metric_weights=metric_weights)

        # Embed targets
        tgt = tgt.unsqueeze(-1)
        tgt = self.output_embedding(tgt)

        # Decode with memory
        output, self_attn_weights, cross_attn_weights = self.decoder(tgt, memory, metric_weights=metric_weights)

        # Final projection
        predictions = self.fc_out(output).squeeze(-1)
        return predictions, self_attn_weights, cross_attn_weights
