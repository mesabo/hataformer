#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/19/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multi_step_predictions(actual, predicted, output_path, qty=0.25):
    """
    Plots multi-step forecasting predictions for a subset of the data.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Directory path to save the plots.
        qty (float): Fraction of the total rows to plot (e.g., 0.25 for the first 25% of rows).
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    n_steps = actual.shape[1]

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    for step in range(n_steps):
        plt.figure(figsize=(12, 6))
        plt.plot(actual[:length, step], label=f"Actual (Step {step + 1})", color="blue")
        plt.plot(predicted[:length, step], label=f"Predicted (Step {step + 1})", linestyle="dashed", color="orange")

        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.title(f"Multi-Step Forecasting Predictions (Step {step + 1})")
        plt.legend()
        plt.tight_layout()

        # Save each step's plot with a unique name
        step_output_path = os.path.join(output_path, str("predicted_vs_actual_step"), f"{step + 1}.png")
        os.makedirs(os.path.dirname(step_output_path), exist_ok=True)
        plt.savefig(step_output_path)
        plt.close()


def plot_aggregated_steps(actual, predicted, output_path, qty=0.25):
    """
    Plots aggregated multi-step forecasting predictions by flattening the arrays.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Path to save the plot.
    """

    length = int(len(actual.flatten()) * qty)  # Number of rows to include in the plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual.flatten()[-length:], label="Actual", alpha=0.7)
    plt.plot(predicted.flatten()[-length:], label="Predicted", alpha=0.7, linestyle="dashed")
    plt.xlabel("Flattened Samples")
    plt.ylabel("Values")
    plt.title("Aggregated Multi-Step Forecasting Predictions")
    plt.legend()
    plt.tight_layout()
    aggr_output_path = os.path.join(output_path, "aggregated_steps.png")
    os.makedirs(os.path.dirname(aggr_output_path), exist_ok=True)
    plt.savefig(aggr_output_path)
    plt.close()


def plot_error_heatmap(actual, predicted, output_path, qty=0.25):
    """
    Plots an error heatmap for multi-step forecasting predictions.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Path to save the heatmap.
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    errors = np.abs(actual[:length] - predicted[:length])
    plt.figure(figsize=(12, 8))
    sns.heatmap(errors, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Error Heatmap (Multi-Step Forecasting)")
    plt.xlabel("Forecast Steps")
    plt.ylabel("Samples")
    plt.tight_layout()
    ehm_output_path = os.path.join(output_path, "error_heatmap.png")
    os.makedirs(os.path.dirname(ehm_output_path), exist_ok=True)
    plt.savefig(ehm_output_path)
    plt.close()


def plot_residuals(actual, predicted, output_path, qty=0.25):
    """
    Plots residuals (actual - predicted) for each forecasting step.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Path to save the residual plots.
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    n_steps = actual.shape[1]
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, 10), sharex=True)
    for step in range(n_steps):
        residuals = actual[:length, step] - predicted[:length, step]
        axes[step].plot(residuals, label=f"Residuals Step {step + 1}")
        axes[step].axhline(0, color="black", linestyle="--", alpha=0.7)
        axes[step].legend()

    plt.xlabel("Samples")
    plt.suptitle("Residuals Analysis (Multi-Step Forecasting)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    res_output_path = os.path.join(output_path, "residuals.png")
    os.makedirs(os.path.dirname(res_output_path), exist_ok=True)
    plt.savefig(res_output_path)
    plt.close()

def plot_multi_step_predictions_with_uncertainty(actual, predicted, lower_bound, upper_bound, output_path, qty=0.25):
    """
    Plots multi-step predictions with confidence intervals for a subset of the data.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        lower_bound (np.ndarray): Lower confidence bound, shape (samples, steps).
        upper_bound (np.ndarray): Upper confidence bound, shape (samples, steps).
        output_path (str): Directory path to save the plots.
        qty (float): Fraction of total rows to plot.
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    n_steps = actual.shape[1]

    os.makedirs(output_path, exist_ok=True)

    for step in range(n_steps):
        plt.figure(figsize=(12, 6))
        plt.plot(actual[:length, step], label=f"Actual (Step {step + 1})", color="blue")
        plt.plot(predicted[:length, step], label=f"Predicted (Step {step + 1})", linestyle="dashed", color="orange")
        plt.fill_between(
            np.arange(len(lower_bound[:length, step])),
            lower_bound[:length, step],
            upper_bound[:length, step],
            color="gray",
            alpha=0.3,
            label="Confidence Interval",
        )

        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.title(f"Multi-Step Forecasting Predictions with Uncertainty (Step {step + 1})")
        plt.legend()
        plt.tight_layout()

        step_output_path = os.path.join(output_path, "with_uncertainty", f"{step + 1}.png")
        os.makedirs(os.path.dirname(step_output_path), exist_ok=True)
        plt.savefig(step_output_path)
        plt.close()

def plot_attention_maps(attention_maps, output_path):
    """
    Plots attention maps for masked and cross-attention weights.

    Args:
        attention_maps (dict): Dictionary containing masked and cross-attention weights.
            Format: {
                "masked": [list of layers with attention maps],
                "cross": [list of layers with attention maps]
            }
        output_path (str): Directory path to save the plots.
    """
    os.makedirs(output_path, exist_ok=True)

    for attn_type, layers in attention_maps.items():
        for layer_idx, layer_maps in enumerate(layers):
            # layer_maps: (batch_size, num_heads, seq_len, seq_len)
            for batch_idx, batch_map in enumerate(layer_maps):
                for head_idx, head_attn_map in enumerate(batch_map):
                    if head_attn_map.ndim == 4:
                        # Reduce batch dimension for visualization; use first batch
                        head_attn_map = head_attn_map[0]

                    if head_attn_map.ndim == 3:
                        # Reduce head dimension for visualization; select one head
                        head_attn_map = head_attn_map[0]  # Example: visualize first head

                    if head_attn_map.ndim != 2:
                        raise ValueError(
                            f"Expected a 2D attention map but got shape {head_attn_map.shape} "
                            f"for {attn_type} at layer {layer_idx + 1}, batch {batch_idx + 1}, head {head_idx + 1}."
                        )

                    # Plot the attention map
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(head_attn_map, cmap="viridis", cbar=True)
                    plt.title(
                        f"{attn_type.capitalize()} Attention\nLayer: {layer_idx + 1}, Batch: {batch_idx + 1}, Head: {head_idx + 1}"
                    )
                    plt.xlabel("Key Positions")
                    plt.ylabel("Query Positions")
                    plt.tight_layout()

                    # Save the plot
                    save_path = os.path.join(
                        output_path, f"{attn_type}_layer{layer_idx + 1}_batch{batch_idx + 1}_head{head_idx + 1}.png"
                    )
                    plt.savefig(save_path)
                    plt.close()

    print(f"Attention maps successfully plotted and saved to {output_path}")

def plot_last_layer_attention_maps(attention_maps, output_path):
    """
    Plots attention maps for the last layer only, in a grid format for masked and cross-attention weights.

    Args:
        attention_maps (dict): Dictionary containing masked and cross-attention weights.
            Format: {
                "masked": [list of layers with attention maps],
                "cross": [list of layers with attention maps]
            }
        output_path (str): Directory path to save the plots.
    """
    os.makedirs(output_path, exist_ok=True)

    for attn_type, layers in attention_maps.items():
        # Select the last layer
        last_layer_maps = np.array(layers[-1])

        # Check if there is an extra dimension
        if last_layer_maps.ndim == 5:
            # Merge the first two dimensions (batch size + another dim)
            last_layer_maps = last_layer_maps.reshape(-1, *last_layer_maps.shape[2:])

        if last_layer_maps.ndim != 4:
            raise ValueError(
                f"Expected attention maps with shape (batch_size, num_heads, seq_len, seq_len) "
                f"but got {last_layer_maps.shape} for {attn_type} at the last layer."
            )

        batch_size, num_heads, seq_len, _ = last_layer_maps.shape

        for batch_idx in range(batch_size):
            # Create a grid of subplots: heads as rows
            fig, axes = plt.subplots(
                nrows=num_heads,
                ncols=1,
                figsize=(10, num_heads * 3),
                sharex=True,
                sharey=True,
            )

            for head_idx in range(num_heads):
                # Extract attention map for this batch, layer, and head
                head_attn_map = last_layer_maps[batch_idx, head_idx, :, :]

                ax = axes[head_idx] if num_heads > 1 else axes  # Handle single-row grids
                sns.heatmap(
                    head_attn_map,
                    cmap="viridis",
                    cbar=True,
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False,
                )
                ax.set_title(
                    f"Head {head_idx + 1}", fontsize=10
                )
                ax.set_xlabel("Key Positions")
                ax.set_ylabel("Query Positions")

            # Set a global title and save the plot
            fig.suptitle(
                f"{attn_type.capitalize()} Attention - Last Layer, Batch {batch_idx + 1}",
                fontsize=14,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            save_path = os.path.join(
                output_path,
                f"{attn_type}_last_layer_batch{batch_idx + 1}.png",
            )
            plt.savefig(save_path)
            plt.close()

    print(f"Attention maps for the last layer successfully plotted and saved to {output_path}")








