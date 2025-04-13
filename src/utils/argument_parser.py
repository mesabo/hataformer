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

import argparse

from distutils.util import strtobool


def get_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Time-series forecasting with Transformers."
    )

    # Task choices
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "synthetic",
            "tetuancity_data",
            "energy_data",
            "electricity_data",
            "household_data",
            "etth1_data",
            "etth2_data",
            "ettm1_data",
            "ettm2_data",
        ],
        default="synthetic",
        help="Task type (e.g., time_series, classification, regression).",
    )

    # Test case choices
    parser.add_argument(
        "--test_case",
        type=str,
        choices=[
            "generate_data",
            "preprocess_data",
            # -----------------
            "train_transformer",
            "evaluate_transformer",
            "train_tsformer",
            "evaluate_tsformer",
            "train_hatformer",
            "evaluate_hatformer",
            "train_hataformer",
            "evaluate_hataformer",
            # -----------------
            "tuning_train_tsformer",
            "tuning_evaluate_tsformer",
            "tuning_train_hatformer",
            "tuning_evaluate_hatformer",
            "tuning_evaluate_hataformer",
            # -----------------
            "ablation_train_tsformer",
            "ablation_evaluate_tsformer",
            "ablation_train_hatformer",
            "ablation_evaluate_hatformer",
            "ablation_evaluate_hataformer",
            # -----------------
            "convert_predictions",
            # =====================
        ],
        default="generate_data",
        help="Test case to run.",
    )

    # Dataset and paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/time_series_data.csv",
        help="Path to dataset.",
    )
    parser.add_argument(
        "--target_name", type=str, default="conso", help="Dataset target name"
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save models and outputs."
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        choices=["minmax", "revnorm", "standard"],
        default="revnorm",
        help="Normalization method to use.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        choices=["minutely", "hourly", "daily", "weekly", "monthly"],
        default="hourly",
        help="Frequency of the time series.",
    )
    parser.add_argument(
        "--add_temporal_features",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Add temporal features.",
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        choices=["temporal", "learnable", "sinusoidal"],
        default="sinusoidal",
        help="Whether to use learnable positional embeddings or fixed sinusoidal encoding.",
    )
    parser.add_argument(
        "--enable_ci",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Enable Channel Independence (CI) in the model.",
    )
    parser.add_argument(
        "--enable_cd",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Enable Channel Dependency (CD) in the model.",
    )
    parser.add_argument(
        "--enable_gating",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Enable gating mechanisms in the decoder.",
    )
    parser.add_argument(
        "--compute_metric_weights",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Enable computaion of metric weight.",
    )
    parser.add_argument(
        "--visualize_attention",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Enable attention weights visualization",
    )
    parser.add_argument(
        "--stochastic_depth_prob",
        type=float,
        default=0.0,  # Provide reasonable defaults based on research
        choices=[0.0, 0.1, 0.2, 0.3, 0.5],  # Allowable values
        help="List of probabilities for applying stochastic depth to encoder/decoder layers. Choose from: 0.0, 0.1, 0.2, 0.3, 0.5",
    )
    parser.add_argument(
        "--local_window_size",
        type=int,
        required=True,
        default=0,
        # choices=[10, 15, 20, 25, 30, 35, 40, 45, 50],  # Allowable values
        help="In a sequence of length T, a local_window_size of w ensures that each timestep attends only to w timesteps before and after it, instead of attending to the entire sequence.",
    )
    parser.add_argument(
        "--num_channels",
        nargs="+",  # Accept multiple values as a list
        type=int,
        default=1,  # Default to a single channel
        choices=[1, 2, 3, 4, 5],
        help="List of input/output channels for multivariate time series (e.g., --channels 1 2 3).",
    )
    # Model and hyperparameters
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "tsformer", "hatformer", "hataformer"],
        default="transformer",
        help="Model type.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[16, 32, 64, 96, 128, 256,512],
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        choices=[2, 5, 10, 15, 20, 25, 50, 60, 100],
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        choices=[0.001, 0.0001, 0.0005, 0.00001, 0.000001],
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        choices=[5, 8, 10, 20, 25, 50, 100, 200],
        default=50,
        help="Patience for early stopping.",
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        choices=[0.0, 0.01, 0.001, 0.0001, 0.00001,  0.000001],
        default=0.000001,
        help="Minimum change in validation loss to qualify as an improvement for early stopping.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        choices=[0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        default=1e-5,
        help="L2 regularization term (default: 0.0).",
    )
    parser.add_argument(
        "--lookback_window",
        type=int,
        choices=[1, 7, 14, 21, 30, 60, 90, 96, 120],
        default=30,
        help="Lookback window size.",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        choices=[1, 7, 12, 14, 21, 24, 30, 48, 60, 90, 96, 120, 192, 336, 720],
        default=7,
        help="Forecast horizon.",
    )

    # Transformer-specific parameters
    parser.add_argument(
        "--d_model",
        type=int,
        choices=[8, 16, 32, 64, 128, 256, 512],
        default=64,
        help="Model dimensionality.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        choices=[1, 2, 4, 6, 8, 16],
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        choices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        default=128,
        help="Feed-forward dimensionality.",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        choices=[1, 2, 4, 6, 8],
        default=2,
        help="Number of encoder layers.",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        choices=[1, 2, 4, 6, 8],
        default=2,
        help="Number of decoder layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3, 0.5],
        default=0.1,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        choices = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        default=0.0,
        help="Weighting factor for local vs global attention.",
    )

    # Computation device
    parser.add_argument(
        "--bias_type",
        type=str,
        choices=["learned", "scheduled"],
        default="learned",
        help="Bias type for global attention.",
    )

    # Computation device
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default="cuda",
        help="Device to use.",
    )

    # Event type for logging
    parser.add_argument(
        "--event",
        type=str,
        choices=["default", "common", "hyperparam", "ablation"],
        default="default",
        help="Event type for logging.",
    )

    return parser.parse_args()
