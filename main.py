import sys
from pathlib import Path

# Import test cases (to be implemented in subsequent steps)
from src.data_processing.generate_data import generate_multivariate_time_series
from src.data_processing.prepare_data import preprocess_time_series_data
from src.training.hataformer_training_testing import TrainHATAFormer
from src.utils.argument_parser import get_arguments
from src.utils.convert_predictions import ConvertPredictions
from src.utils.device_utils import setup_device
from src.utils.logger_config import setup_logger
from src.utils.output_config import get_output_paths

# Add the `src` directory to `PYTHONPATH`
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.append(str(PROJECT_ROOT))

import torch

# Map test cases to functions
TEST_CASES = {
    # ------------------ DATA MANAGEMENT
    "generate_data": generate_multivariate_time_series,
    "preprocess_data": preprocess_time_series_data,
    # ================== NORMAL TRAINING
    "train_hataformer": lambda args: TrainHATAFormer(args).train(),
    "evaluate_hataformer": lambda args: TrainHATAFormer(args).evaluate(Path(args.data_path)),
    # ------------------
    # ================== TUNNING
    "tuning_train_hataformer": lambda args: TrainHATAFormer(args).train(),
    "tuning_evaluate_hataformer": lambda args: TrainHATAFormer(args).evaluate(Path(args.data_path)),
    # ------------------
    # ================== ABLATION
    "ablation_train_hataformer": lambda args: TrainHATAFormer(args).train(),
    "ablation_evaluate_hataformer": lambda args: TrainHATAFormer(args).evaluate(Path(args.data_path)),
    # ------------------
    # ================== ABLATION
    "convert_predictions": lambda args: ConvertPredictions(args).convert(),
}


def main():
    # Parse arguments
    args = get_arguments()

    if args.test_case in ["convert_predictions"]:
        # Setup device
        args.device = "cuda"
        args.test_case = "evaluate_tsformer"
        # Get output paths
        output_paths = get_output_paths(args)
        args.test_case = "convert_predictions"
    else:
        # Setup device
        device = setup_device(args)
        args.device = device.type  # Add device information to args

        # Get output paths
        output_paths = get_output_paths(args)

    # Setup logger
    logger = setup_logger(args)
    args.logger = logger
    args.output_paths = output_paths

    # Log the configuration details
    logger.info(f"Task: {args.task}")
    logger.info(f"Test Case: {args.test_case}")
    logger.info(f"Event: {args.event}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Frequency: {args.frequency}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Lookback Window: {args.lookback_window}")
    logger.info(f"Forecast Horizon: {args.forecast_horizon}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Weight Decay: {args.weight_decay}")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"d_model: {args.d_model}")
    logger.info(f"n_heads: {args.n_heads}")
    logger.info(f"d_ff: {args.d_ff}")
    logger.info(f"Encoder Layers: {args.num_encoder_layers}")
    logger.info(f"Decoder Layers: {args.num_decoder_layers}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Stochastic Depth Probability: {args.stochastic_depth_prob}")
    logger.info(f"Add Temporal Features: {args.add_temporal_features}")
    logger.info(f"Learnable Positional Encoding: {args.encoding_type}")
    logger.info(f"Enable CI: {args.enable_ci}")
    logger.info(f"Enable CD: {args.enable_cd}")
    logger.info(f"Compute Metric Weights: {args.compute_metric_weights}")
    logger.info(f"Visualize Attention: {args.visualize_attention}")
    logger.info(f"Local Window Size: {args.local_window_size}")
    logger.info(f"Bias type: {args.bias_type}")
    logger.info(f"Log Path: {output_paths['logs']}")
    logger.info(f"Model Path: {output_paths['models']}")
    logger.info(f"Results Path: {output_paths['results']}")
    logger.info(f"Metrics Path: {output_paths['metrics']}")
    logger.info(f"Visuals Path: {output_paths['visuals']}")

    # Execute the selected test case
    if args.test_case in TEST_CASES:
        logger.info(f"{10 * '🌟'} Running {args.test_case} {10 * '🌟'}")
        try:
            TEST_CASES[args.test_case](args)  # Pass arguments to the function
        except Exception as e:
            logger.error(f"Error while running test case '{args.test_case}': {e}")
            logger.error("Skipping to the next test case...")
    else:
        logger.warning(f"Invalid test case: {args.test_case}. Skipping execution.")
        logger.error(f"Available test cases: {list(TEST_CASES.keys())}")

    logger.info(f"{10 * '🏁'} ALL EXECUTIONS DONE! {10 * '🏁'}")


if __name__ == "__main__":
    # Set the seed
    torch.manual_seed(99)  # For CPU
    torch.cuda.manual_seed(99)  # For GPU (if available)

    main()
