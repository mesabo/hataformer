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

from pathlib import Path
import time
import torch
import psutil
import os
import numpy as np
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go


def load_predictions_and_targets(preds_path):
    """
    Loads predictions and targets from a saved .npz file.

    Args:
        preds_path (str or Path): Path to the .npz file containing predictions and targets.

    Returns:
        tuple: A tuple containing:
            - predictions (np.ndarray): Predicted values, shape (samples, steps).
            - targets (np.ndarray): Actual values, shape (samples, steps).
    """
    preds_path = Path(preds_path)

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {preds_path}")

    data = np.load(preds_path)
    predictions = data["predictions"]
    targets = data["targets"]

    print(f"Loaded predictions and targets from {preds_path}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    return predictions, targets


class TrainingProfiler:
    """
    A utility class to measure elapsed time and memory utilization during training or evaluation.
    """

    def __init__(self, save_path=None):
        self.start_time = None
        self.end_time = None
        self.max_memory = 0
        self.save_path = save_path

    def start(self):
        """
        Starts the timer and initializes memory tracking.
        """
        self.start_time = time.time()
        self.max_memory = 0

    def stop(self):
        """
        Stops the timer and calculates the elapsed time.
        """
        self.end_time = time.time()

    def record_memory(self):
        """
        Tracks the peak memory usage.
        """
        # Get the memory usage for the current process in MB
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 ** 2)
        self.max_memory = max(self.max_memory, current_memory)

    def elapsed_time(self):
        """
        Returns the elapsed time in seconds.
        """
        return self.end_time - self.start_time if self.end_time and self.start_time else None

    def report(self):
        """
        Returns a report of the elapsed time and peak memory usage.
        """
        return {
            "elapsed_time": self.elapsed_time(),
            "max_memory_mb": self.max_memory,
        }

    def save_report(self):
        """
        Saves the profiling report to a JSON file if a save path is provided.
        """
        if self.save_path:
            report = self.report()
            os.makedirs(Path(self.save_path).parent, exist_ok=True)
            with open(self.save_path, "w") as f:
                json.dump(report, f, indent=4)
            print(f"Profiler report saved to {self.save_path}")


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path to save the best model.
        verbose (bool): Whether to print messages about the early stopping process.
    """
    def __init__(self, patience=10, delta=0.0, path="checkpoint.pt", verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_optimizer(model, optimizer_name, learning_rate, weight_decay=0):
    """
    Returns an optimizer for the model.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_name (str): Name of the optimizer. Options: ["adam", "sgd"].
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization term (default: 0).

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def save_attention_maps(attention_maps, output_path):
    """
    Saves attention maps in a consistent format for later use.

    Args:
        attention_maps (dict): Dictionary containing masked and cross attention weights.
            Format: {
                "masked": [list of masked attention maps per layer],
                "cross": [list of cross attention maps per layer]
            }
        output_path (str): Directory path to save the serialized attention maps.
    """
    os.makedirs(output_path, exist_ok=True)

    for attn_type, attn_map_list in attention_maps.items():
        for layer_idx, layer_attn_maps in enumerate(attn_map_list):
            # layer_attn_maps shape: (batch_size, n_heads, seq_len_query, seq_len_key)
            layer_path = os.path.join(output_path, f"{attn_type}_layer_{layer_idx + 1}")
            os.makedirs(layer_path, exist_ok=True)

            for batch_idx, batch_attn_map in enumerate(layer_attn_maps):
                file_path = os.path.join(layer_path, f"batch_{batch_idx + 1}.npy")
                np.save(file_path, batch_attn_map)

    print(f"Attention maps saved to {output_path}")


def visualize_positional_encoding(
    pe_path,
    method='pca',
    color_by='position',
    save_fig=True,
    num_batch = None,
    encoding_type=None,
    point_size=6,
    lookback=96,
    forecast_horizon=96,
    width=600,
    height=450,
    output_format="both"  # 'png', 'pdf', or 'both'
):
    """
    Visualize positional encodings from a saved .pt tensor using Plotly.

    Args:
        pe_path (str): Path to base directory containing 'data' and 'image' subdirs.
        method (str): 'pca' or 'tsne'.
        color_by (str): 'position' or 'batch'.
        save_fig (bool): Whether to save the figure.
        show_fig (bool): Whether to show the plot.
        encoding_type (str): E.g., 'temporal_proj'.
        point_size (int): Marker size.
        lookback (int): Lookback window size (for plot title).
        forecast_horizon (int): Forecast horizon size (for plot title).
        plot_samples (int): Used for legend text.
        width (int): Width of plot in px.
        height (int): Height of plot in px.
        output_format (str): 'png', 'pdf', or 'both'.
    """
    if encoding_type is None:
        raise ValueError("You must specify an encoding_type (e.g. 'temporal_proj')")

    # Directories
    load_pe_data = os.path.join(pe_path, "data")
    save_pe_plot = os.path.join(pe_path, "image")
    os.makedirs(load_pe_data, exist_ok=True)
    os.makedirs(save_pe_plot, exist_ok=True)

    # Load tensor
    pe_file = os.path.join(load_pe_data, f"batch{num_batch}_pe_{encoding_type}.pt")
    if not os.path.isfile(pe_file):
        raise FileNotFoundError(f"Encoding file not found: {pe_file}")
    pe = torch.load(pe_file, weights_only=True)

    if pe.ndim != 3:
        raise ValueError(f"Expected [B, T, D] tensor, got shape {pe.shape}")

    B, T, D = pe.shape
    flat_pe = pe.view(-1, D).cpu().numpy()

    # Dimensionality reduction
    reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    reduced = reducer.fit_transform(flat_pe)

    # Color mapping
    if color_by == 'position':
        colors = np.tile(np.arange(T), B)
        color_label = 'Seq Pos'
    elif color_by == 'batch':
        colors = np.repeat(np.arange(B), T)
        color_label = 'Batch Index'
    else:
        raise ValueError("color_by must be 'position' or 'batch'")

    # Create scatter plot
    scatter = go.Scattergl(
        x=reduced[:, 0],
        y=reduced[:, 1],
        mode='markers',
        marker=dict(
            size=point_size,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title=color_label),
            showscale=True
        ),
        name="Encoding",
        text=[f"Index {i}" for i in range(len(colors))]
    )

    fig = go.Figure(data=[scatter])

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"l={lookback}/p={forecast_horizon} — PE Projection ({encoding_type}, batch {num_batch}) — {method.upper()}",
            font=dict(size=20, family="Arial", color="black")
        ),
        xaxis=dict(
            title=dict(
                text="Component 1",
                font=dict(size=20, family="Arial", color="black")
            ),
            tickfont=dict(size=20, family="Arial")
        ),
        yaxis=dict(
            title=dict(
                text="Component 2",
                font=dict(size=20, family="Arial", color="black")
            ),
            tickfont=dict(size=20, family="Arial")
        ),
        legend=dict(
            font=dict(size=10, family="Arial")
        ),
        template="plotly_white",
        width=width,
        height=height,
        autosize=False
    )

    # Save
    if save_fig:
        base_filename = f"batch{num_batch}_pe_{encoding_type}_{method}"
        save_base_path = os.path.join(save_pe_plot, base_filename)
        os.makedirs(save_pe_plot, exist_ok=True)

        if output_format in ("png", "both"):
            fig.write_image(f"{save_base_path}.png", scale=4)
        if output_format in ("pdf", "both"):
            fig.write_image(f"{save_base_path}.pdf", scale=4)

        print(f"[Saved] {save_base_path}.(png/pdf)")
    
    del fig
    

def visualize_positional_encoding_grid(
    pe_path,
    method='pca',
    color_by='position',
    save_fig=True,
    encoding_type=None,
    point_size=6,
    lookback=96,
    forecast_horizon=96,
    width=2000,
    height=900,
    output_format="both"
):
    """
    Loads and visualizes the first 10 positional encoding batch files in a 2x5 grid.

    Args:
        pe_path (str): Path to directory with 'data' and 'image' subfolders.
        method (str): 'pca' or 'tsne'.
        color_by (str): 'position' or 'batch'.
        save_fig (bool): Save the figure.
        encoding_type (str): e.g. 'temporal_proj'
        point_size (int): Size of each plotted point.
        lookback (int): Used in title.
        forecast_horizon (int): Used in title.
        width (int): Figure width.
        height (int): Figure height.
        output_format (str): 'png', 'pdf', or 'both'.
    """
    if encoding_type is None:
        raise ValueError("You must specify an encoding_type (e.g. 'temporal_proj')")

    load_pe_data = os.path.join(pe_path, "data")
    save_pe_plot = os.path.join(pe_path, "image")
    os.makedirs(save_pe_plot, exist_ok=True)

    fig = make_subplots(rows=2, cols=5, subplot_titles=[f"Batch {i:02d}" for i in range(10)],
                        horizontal_spacing=0.03, vertical_spacing=0.1)

    for i in range(10):
        batch_file = os.path.join(load_pe_data, f"batch{i}_pe_{encoding_type}.pt")
        if not os.path.isfile(batch_file):
            print(f"[Skipped] {batch_file} not found.")
            continue

        pe = torch.load(batch_file, weights_only=True)
        if pe.ndim != 3:
            raise ValueError(f"Expected shape [B, T, D], got {pe.shape}")
        B, T, D = pe.shape
        flat_pe = pe.view(-1, D).cpu().numpy()

        # Reduce
        reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
        reduced = reducer.fit_transform(flat_pe)

        # Colors
        if color_by == 'position':
            colors = np.tile(np.arange(T), B)
            color_label = 'Seq Pos'
        elif color_by == 'batch':
            colors = np.repeat(np.arange(B), T)
            color_label = 'Batch Index'
        else:
            raise ValueError("color_by must be 'position' or 'batch'")

        # Add to subplot
        row, col = divmod(i, 5)
        scatter = go.Scattergl(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors,
                colorscale='Viridis',
                showscale=(i == 0),  # Only show colorbar once
                colorbar=dict(title=color_label) if i == 0 else None
            ),
            name=f"Batch {i:02d}",
            showlegend=False
        )
        fig.add_trace(scatter, row=row+1, col=col+1)

    # Global layout
    fig.update_layout(
        title=dict(
            text=f"{lookback} lookback / {forecast_horizon} ahead — PE Projection Grid ({encoding_type}, {method.upper()})",
            font=dict(size=26, family="Arial", color="black")
        ),
        height=height,
        width=width,
        template="plotly_white"
    )

    # Save
    if save_fig:
        filename_base = f"grid_pe_{encoding_type}_{method}"
        path_base = os.path.join(save_pe_plot, filename_base)
        if output_format in ("png", "both"):
            fig.write_image(f"{path_base}.png", scale=4)
        if output_format in ("pdf", "both"):
            fig.write_image(f"{path_base}.pdf", scale=4)
        print(f"[Saved] {path_base}.(png/pdf)")

    del fig