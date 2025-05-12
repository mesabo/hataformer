#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/05/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch

# Add the `src` directory to `PYTHONPATH`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

class ConvertPredictions:
    """
    Reads predictions stored in .npz format, reverts normalization using stored parameters, 
    takes the last step in multi-step predictions, and converts them into a CSV file.
    """

    def __init__(self, args):
        self.args = args
        self.common_name = f"{args.optimizer}_{args.weight_decay}_{args.learning_rate}"
        self.preds_path = self.args.output_paths["predictions"] / f"{self.common_name}.npz"
        self.profile_path = self.args.output_paths["profiles"] / f"{self.common_name}.csv"
        self.csv_output_path = self.args.output_paths["predictions"] / f"{self.common_name}.csv"
        self.plot_output_path = self.args.output_paths["predictions"] / f"{self.common_name}.pdf"

        data_path = Path(args.data_path).resolve()
        self.processed_dir = data_path / f"lookback{args.lookback_window}_forecast{self.args.forecast_horizon}"
        self.norm_info_path = self.processed_dir / "normalization_info.json"

    def revert_normalization(self, data, norm_info):
        """
        Reverts normalization using stored parameters. Supports:
        - MinMax Scaling
        - Reversible Normalization (RevNorm)
        """
        if "target_mean" in norm_info and "target_std" in norm_info:
            # Reversible Normalization (RevNorm)
            return (data * norm_info["target_std"]) + norm_info["target_mean"]
        elif "scaler_min_" in norm_info and "scaler_scale_" in norm_info:
            # MinMax Scaling
            return (data - norm_info["scaler_min_"]) / norm_info["scaler_scale_"]
        else:
            raise ValueError("Invalid normalization information detected.")

    def convert(self):
        """
        Reads the .npz file, extracts predictions and targets, takes the last step in multi-step forecasts, 
        reverts normalization, and saves them as a CSV file.
        """
        if not self.preds_path.exists():
            self.args.logger.error(f"Prediction file not found at {self.preds_path}")
            return

        if not self.norm_info_path.exists():
            self.args.logger.error(f"Normalization info file not found at {self.norm_info_path}")
            return

        # Load normalization info
        with open(self.norm_info_path, "r") as f:
            norm_info = json.load(f)

        # Load predictions
        data = np.load(self.preds_path)
        predictions = data["predictions"][:, -1]  # Last step in forecast
        targets = data["targets"][:, -1]  # Last step in forecast

        # Revert normalization
        predictions = self.revert_normalization(predictions, norm_info)
        targets = self.revert_normalization(targets, norm_info)

        # Save to CSV
        df_combined = pd.DataFrame({"Actual": targets, "Predicted": predictions})
        df_combined.to_csv(self.csv_output_path, index=True)

        self.args.logger.info(f"Predictions successfully converted and saved to {self.csv_output_path}")

        # Generate plot
        self.plot_aggregated_actual_vs_predicted(targets, predictions, self.args.lookback_window, self.args.forecast_horizon, percentage=25)
        
        # Process CSV file
        #self.process_csv(self.profile_path)

    def hot_revert_norm(self, predictions, targets):
        """
        Reverts normalization on-the-fly using self.revert_normalization and norm_info.json.

        Inputs:
            predictions: torch.Tensor or np.ndarray (normalized)
            targets:     torch.Tensor or np.ndarray (normalized)

        Returns:
            Tuple (targets_denorm, predictions_denorm) as np.ndarrays
        """
        if not self.norm_info_path.exists():
            self.args.logger.error(f"Normalization info not found at {self.norm_info_path}")
            return None, None

        with open(self.norm_info_path, "r") as f:
            norm_info = json.load(f)

        # Ensure NumPy arrays (if torch)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Revert normalization using internal logic
        predictions = self.revert_normalization(predictions, norm_info)
        targets = self.revert_normalization(targets, norm_info)

        return targets, predictions

    def plot_aggregated_actual_vs_predicted(self, actual, predicted, lookback, forecast_horizon, percentage=100):
        """
        Plots actual vs. predicted values using aggregated steps.
        """
        num_samples = len(actual)
        plot_samples = 40  # Limits number of plotted samples

        aggregated_actual = actual[:lookback + forecast_horizon].tolist()
        aggregated_predicted = predicted[:lookback + forecast_horizon].tolist()

        for i in range(lookback + forecast_horizon, num_samples, forecast_horizon):
            aggregated_actual.append(actual[i])
            aggregated_predicted.append(predicted[i])

        aggregated_actual = aggregated_actual#[-plot_samples:]
        aggregated_predicted = aggregated_predicted#[-plot_samples:]

        plt.figure(figsize=(6, 6))
        plt.plot(aggregated_actual, label="Actual", color="blue")
        plt.plot(aggregated_predicted, label="Predicted", linestyle="dashed", color="orange")
        
        plt.xlabel("Samples")
        plt.ylabel("Power Consumption (KWh)")
        plt.title(f"Actual vs Predicted ({plot_samples} last Samples)")
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.plot_output_path), exist_ok=True)
        plt.savefig(self.plot_output_path)
        plt.close()
        
        self.args.logger.info(f"Aggregated plot saved to {self.plot_output_path}")

    def process_csv(file_path):
        """
        Reads a CSV file containing epoch, time elapsed, and memory usage.
        Computes the difference per step for time and memory, then saves the updated data.

        Parameters:
        file_path (str or Path): Path to the CSV file.

        Returns:
        pd.DataFrame: Processed DataFrame with time_per_step and memory_per_step.
        """

        # Load CSV file
        df = pd.read_csv(file_path)

        # Ensure correct column names
        df.columns = ["epoch", "time_elapsed", "memory_used_mb"]

        # Compute time per step and memory per step
        df["time_per_step"] = df["time_elapsed"].diff().fillna(0)
        df["memory_per_step"] = df["memory_used_mb"].diff().fillna(0)

        df.to_csv(file_path, index=False)

        print(f"Processed data saved to {file_path}")

        return df








