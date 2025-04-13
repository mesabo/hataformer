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
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from src.utils.temporal_feature_extraction import add_temporal_and_holiday_features

# Add the `src` directory to `PYTHONPATH`
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.append(str(PROJECT_ROOT))

def preprocess_time_series_data(args):
    """
    Preprocesses raw time-series data and splits it into train, validation, and test sets.
    Handles downsampling based on frequency and generates sliding window datasets for each split.
    Supports Min-Max normalization and Reversible Normalization (Rev-Norm).
    """
    try:
        # Define the project root directory
        project_root = Path(__file__).resolve().parents[2]
        raw_data_path = project_root / f"data/raw/{args.task or 'time_series'}.csv"

        # Ensure output directory exists
        processed_dir = project_root / (args.data_path or "data/processed/time_series") / str(
            f"lookback{args.lookback_window}_forecast{args.forecast_horizon}"
        )
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Check if processed files already exist
        train_file = processed_dir / "train_sliding.csv"
        val_file = processed_dir / "val_sliding.csv"
        test_file = processed_dir / "test_sliding.csv"

        if train_file.exists() and val_file.exists() and test_file.exists():
            args.logger.info("Processed files already exist. Skipping preprocessing.")
            return

        # Load raw data
        data = pd.read_csv(raw_data_path, parse_dates=["date"])
        # Fill NaN or null values with 0
        data = data.fillna(0)

        # Optionally add temporal covariates
        if args.add_temporal_features:
            data = add_temporal_and_holiday_features(data=data)

        # Resample data based on frequency
        frequency_map = {
            "minutely": "T",   # T for minutes
            "hourly": "H",     # H for hours
            "daily": "D",      # D for days
            "weekly": "W",     # W for weeks
            "monthly": "M",    # M for months
        }
        if "frequency" in args and args.frequency:
            freq = frequency_map.get(args.frequency.lower())
            if not freq:
                raise ValueError(f"Invalid frequency: {args.frequency}")
            data = data.resample(freq, on="date").mean().dropna()

        # Infer feature columns
        feature_columns = [col for col in data.columns if col != args.target_name and col != "date"]
        target_column = args.target_name

        # Choose normalization method
        normalization_info = {}
        if args.normalization_method == "minmax":
            # Min-Max normalization
            scaler = MinMaxScaler(feature_range=(-1,1))
            scaled_features = scaler.fit_transform(data[feature_columns])
            scaled_target = scaler.fit_transform(data[[target_column]])
            normalization_info = {
                "scaler_min_": scaler.min_.tolist(),
                "scaler_scale_": scaler.scale_.tolist(),
                "data_min_": scaler.data_min_.tolist(),
                "data_max_": scaler.data_max_.tolist(),
            }

        elif args.normalization_method == "revnorm":
            # Reversible normalization (mean and std)
            feature_means = data[feature_columns].mean()
            feature_stds = data[feature_columns].std()
            target_mean = data[target_column].mean()
            target_std = data[target_column].std()

            scaled_features = (data[feature_columns] - feature_means) / feature_stds
            scaled_target = (data[target_column] - target_mean) / target_std

            normalization_info = {
                "feature_means": feature_means.tolist(),
                "feature_stds": feature_stds.tolist(),
                "target_mean": target_mean,
                "target_std": target_std,
            }

        else:
            raise ValueError("Unsupported normalization method. Use 'minmax' or 'revnorm'.")

        # Combine scaled features and target
        normalized_data = pd.DataFrame(scaled_features, columns=feature_columns)
        normalized_data[args.target_name] = scaled_target

        # Train/validation/test split
        train_split = int(len(normalized_data) * 0.7)
        val_split = int(len(normalized_data) * 0.85)
        train_data = normalized_data.iloc[:train_split]
        val_data = normalized_data.iloc[train_split:val_split]
        test_data = normalized_data.iloc[val_split:]

        # Save normalization info for reversible transformation
        normalization_info_path = processed_dir / "normalization_info.json"
        with open(normalization_info_path, "w") as f:
            json.dump(normalization_info, f, indent=4)

        train_data.to_csv(processed_dir / "train.csv", index=False)
        val_data.to_csv(processed_dir / "val.csv", index=False)
        test_data.to_csv(processed_dir / "test.csv", index=False)

        args.logger.info("Preprocessing complete. Train/Val/Test splits saved.")

        # Generate sliding window datasets
        create_sliding_window_data(train_data, args.lookback_window, args.forecast_horizon, train_file, args.target_name)
        args.logger.info("Creating sliding window datasets for train split complete.")
        create_sliding_window_data(val_data, args.lookback_window, args.forecast_horizon, val_file, args.target_name)
        args.logger.info("Creating sliding window datasets for validation split complete.")
        create_sliding_window_data(test_data, args.lookback_window, args.forecast_horizon, test_file, args.target_name)
        args.logger.info("Creating sliding window datasets for test split complete.")

    except Exception as e:
        args.logger.error(f"Error during preprocessing: {traceback.format_exc()}")
        raise
def create_sliding_window_data(data, lookback, forecast, output_path, target_name):
    """
    Converts time-series data into sliding window format.

    Args:
        data (pd.DataFrame): Input time-series data.
        lookback (int): Number of past time steps used as input.
        forecast (int): Number of future time steps to predict as output.
        output_path (str): Path to save the sliding window formatted dataset.
        target_name (str): Name of the target column in the dataset.

    Returns:
        None: Saves the sliding window dataset to a CSV file.
    """
    features = data.drop(columns=[target_name]).values  # Exclude target column
    target = data[target_name].values  # Use target column
    x, y = [], []

    for i in range(len(features) - lookback - forecast + 1):
        x.append(features[i: i + lookback])  # Shape: (lookback, num_features)
        y.append(target[i + lookback: i + lookback + forecast])  # Shape: (forecast,)

    sliding_df = pd.DataFrame({
        "x": [xi.tolist() for xi in x],  # Convert arrays to lists for storage
        "y": [yi.tolist() for yi in y],
    })

    try:
        sliding_df.to_csv(output_path, index=False)
    except Exception as e:
        raise Exception(f"Failed to save sliding window data to {output_path}. Error: {e}")

class TimeSeriesDataset(Dataset):
    """
    Dataset class for handling sliding window time-series data.

    Args:
        csv_path (str): Path to the sliding window formatted CSV file.
        lookback_window (int): Number of past time steps used as input.
        forecast_horizon (int): Number of future time steps to predict as output.
    """
    def __init__(self, csv_path, lookback_window, forecast_horizon):
        data = pd.read_csv(csv_path)
        self.x = np.array([np.array(eval(xi)) for xi in data["x"]])  # Shape: (num_samples, lookback, num_features)
        self.y = np.array([np.array(eval(yi)) for yi in data["y"]])  # Shape: (num_samples, forecast)

        # Ensure that y has the correct dimensions
        if len(self.y.shape) == 1:
            self.y = self.y[:, np.newaxis]

        if len(self.x.shape) != 3 or len(self.y.shape) != 2:
            raise ValueError(f"Invalid dimensions for X or Y. Shapes: X-{self.x.shape}, Y-{self.y.shape}")

        if self.x.shape[1] != lookback_window or self.y.shape[1] != forecast_horizon:
            raise ValueError("Mismatch in lookback_window or forecast_horizon.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
