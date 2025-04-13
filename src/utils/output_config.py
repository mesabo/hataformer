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

import os
from pathlib import Path


def get_output_paths(args):
    """
    Generate paths for logs, models, results, and metrics based on configuration.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: A dictionary containing paths for logs, models, results, and metrics.
    """
    # Determine event_type for paths other than models
    event_type = args.event if hasattr(args, "event") else "default"

    # Determine the configuration-based subdirectory
    ablation_subdir_parts = []
    if args.compute_metric_weights:
        ablation_subdir_parts.append(f"cmw{args.compute_metric_weights}")
    subdir = "_".join(ablation_subdir_parts) if ablation_subdir_parts else "default_ablation"

    tuning_subdir_parts = []
    if args.compute_metric_weights:
        tuning_subdir_parts.append("cmw")
    if args.d_model:
        tuning_subdir_parts.append(f"dm{args.d_model}")
    if args.n_heads:
        tuning_subdir_parts.append(f"nh{args.n_heads}")
    if args.d_ff:
        tuning_subdir_parts.append(f"dff{args.d_ff}")
    if args.num_encoder_layers:
        tuning_subdir_parts.append(f"el{args.num_encoder_layers}")
    if args.num_decoder_layers:
        tuning_subdir_parts.append(f"dl{args.num_decoder_layers}")
    if args.dropout:
        tuning_subdir_parts.append(f"drop{args.dropout}")
    if args.local_window_size:
        tuning_subdir_parts.append(f"lws{args.local_window_size}")
    subdir = "_".join(tuning_subdir_parts) if tuning_subdir_parts else "default_tuning"

    # Base directory setup
    base_dir = Path(
        "output/") / args.device / args.model / f"batch{args.batch_size}" / args.encoding_type / args.task / f"epoch{args.epochs}" / f"alpha{args.alpha}"

    # Append directories based on event type
    if args.event == "common":
        base_dir = base_dir / Path("common/") / subdir
    elif args.event == "ablation":
        base_dir = base_dir / Path("ablation/") / subdir
    elif args.event == "hyperparam":
        base_dir = base_dir / Path("tuning/") / subdir
    else:
        base_dir = base_dir / Path("default/") / subdir

    event_case = "default"
    if args.test_case in ["train_hataformer","tuning_train_hataformer", "ablation_train_hataformer"]:
        event_case = "training"
    elif args.test_case in ["evaluate_hataformer", "tuning_evaluate_hataformer", "ablation_evaluate_hataformer"]:
        event_case = "testing"
    else:
        event_case = "common"

    paths = {
        "logs": base_dir / "logs" / event_case / args.frequency /
                f"lookback{args.lookback_window}_forecast{args.forecast_horizon}.log",
        "profiles": base_dir / "profiles" / event_case / args.frequency /
                    f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "models": base_dir / "models" / "common" / args.frequency /
                  f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "params": base_dir / "params" / "common" / args.frequency /
                  f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "results": base_dir / "results" / event_case / args.frequency /
                   f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "metrics": base_dir / "metrics" / event_case / args.frequency /
                   f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "visuals": base_dir / "visuals" / event_case / args.frequency /
                   f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "predictions": base_dir / "predictions" / event_case / args.frequency /
                       f"lookback{args.lookback_window}_forecast{args.forecast_horizon}",
    }

    # Ensure directories only for logs when in "generate_data" or "preprocess_data"
    for key, path in paths.items():
        if args.test_case in ["generate_data", "preprocess_data"]:
            if key == "logs":
                os.makedirs(path.parent, exist_ok=True)
                print(f"Log path created: {path}")
            continue  # Skip creating other paths
        else:
            if key == "predictions":
                os.makedirs(path, exist_ok=True)
            elif path.suffix:  # If the path has a file extension
                os.makedirs(path.parent, exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)

    return paths
