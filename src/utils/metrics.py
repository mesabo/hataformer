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

import numpy as np

from src.utils.convert_predictions import ConvertPredictions


class Metrics:
    """
    Metrics calculation class for evaluating forecasting models.

    This class provides various methods to calculate statistical error metrics
    commonly used to evaluate forecasting models against actual data. The methods
    include calculation of mean absolute error, mean squared error, root mean
    squared error, R-squared score, mean absolute percentage error, symmetric
    mean absolute percentage error, mean absolute scaled error, and weighted
    absolute percentage error. The `seasonality` parameter, set during
    initialization, is required for calculating certain metrics like mean absolute
    scaled error.

    :ivar seasonality: Specifies the seasonal period required as input for certain
        metrics, like mean absolute scaled error.
    :type seasonality: int
    """
    def __init__(self, args):
        self.args = args
        self.seasonality = args.forecast_horizon or 1

    def calculate_all(self, actual, predicted):
        """
        Calculate all metrics.
        """
        ConvertPredictions(self.args).hot_revert_norm(actual, predicted)
        return {
            "MAE": float(f"{self.mean_absolute_error(actual, predicted):6f}"),
            "MSE": float(f"{self.mean_squared_error(actual, predicted):6f}"),
            "RMSE": float(f"{self.root_mean_squared_error(actual, predicted):6f}"),
            "R2": float(f"{self.r2_score(actual, predicted):6f}"),
            "MAPE": float(f"{self.mean_absolute_percentage_error(actual, predicted):6f}"),
            "sMAPE": float(f"{self.symmetric_mean_absolute_percentage_error(actual, predicted):6f}"),
            "MASE": float(f"{self.mean_absolute_scaled_error(actual, predicted):6f}"),
            "WAPE": float(f"{self.weighted_absolute_percentage_error(actual, predicted):6f}")
        }

    def mean_absolute_error(self, actual, predicted):
        return np.mean(np.abs(actual - predicted))

    def mean_squared_error(self, actual, predicted):
        return np.mean((actual - predicted) ** 2)

    def root_mean_squared_error(self, actual, predicted):
        return np.sqrt(self.mean_squared_error(actual, predicted))

    def r2_score(self, actual, predicted):
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot)

    def mean_absolute_percentage_error(self, actual, predicted):
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        actual = np.where(actual == 0, epsilon, actual)
        mape = np.nanmean(np.abs((actual - predicted) / actual)) * 100
        return mape

    def symmetric_mean_absolute_percentage_error(self, actual, predicted):
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + epsilon)) * 100
        return smape

    def mean_absolute_scaled_error(self, actual, predicted):
        if self.seasonality >= len(actual):
            raise ValueError("Seasonality must be less than the length of the actual array.")
        naive_forecast = actual[:-self.seasonality]
        naive_forecast = np.where(naive_forecast == 0, np.nan, naive_forecast)  # Replace zeros with NaN
        mase = np.nanmean(
            np.abs(actual[self.seasonality:] - predicted[self.seasonality:]) / np.abs(naive_forecast))
        return mase

    def weighted_absolute_percentage_error(self, actual, predicted):
        sum_actual = np.sum(np.abs(actual))
        if sum_actual == 0:
            raise ValueError("Sum of actual values must not be zero for WAPE calculation.")
        return np.sum(np.abs(actual - predicted)) / sum_actual * 100