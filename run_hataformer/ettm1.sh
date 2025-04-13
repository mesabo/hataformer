#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14

# Load user environment
source ~/.bashrc
hostname

# Detect computation device
DEVICE="cpu"
if [[ -n "$CUDA_VISIBLE_DEVICES" && $(nvidia-smi | grep -c "GPU") -gt 0 ]]; then
    DEVICE="cuda"
elif [[ "$(uname -s)" == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    DEVICE="mps"
fi

# Activate Conda environment
ENV_NAME="forecastnet"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    if ! command -v conda &>/dev/null; then
        echo "Error: Conda not found. Please install Conda and ensure it's in your PATH."
        exit 1
    fi
    source activate "$ENV_NAME" || { echo "Error: Could not activate Conda environment '$ENV_NAME'."; exit 1; }
fi

# Configurations
TASKS=("ettm1_data")
TEST_CASES=("train_hataformer" "evaluate_hataformer")
TARGET="conso"
FREQUENCIES=("hourly")
MODELS=("hataformer")
BATCH_SIZES=("32")
LOOKBACK_WINDOWS=("96")
FORECAST_HORIZONS=("336")
LEARNING_RATES=("0.0005")
WEIGHT_DECAYS=("0.000001")
OPTIMIZERS=("adam")
PATIENCE=("10")
EPOCHS=("10")
D_MODELS=("16")
N_HEADS=("1")
D_FFS=("16")
ENCODER_LAYERS=("1")
DECODER_LAYERS=("1")
DROPOUTS=("0.2")
BIAS_TYPES=("learned" "scheduled")
ADD_TEMPORAL_FEATURES_VALUES=("True")
ENCODING_TYPES=("sinusoidal")
COMPUTE_METRIC_WEIGHTS_VALUES=("True")
VISUALIZE_ATTENTION_VALUES=("False")

# Grid Search Execution
for TASK in "${TASKS[@]}"; do
  for TEST_CASE in "${TEST_CASES[@]}"; do

    case "$TEST_CASE" in
      "train_hataformer" | "evaluate_hataformer") EVENT="common" ;;
      "tuning_train_hataformer" | "tuning_evaluate_hataformer") EVENT="hyperparam" ;;
      "ablation_train_hataformer" | "ablation_evaluate_hataformer") EVENT="ablation" ;;
      *) EVENT="default" ;;
    esac

    echo "Processing $TEST_CASE as EVENT=$EVENT"

    for EPOCH in "${EPOCHS[@]}"; do
      for FREQUENCY in "${FREQUENCIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
          for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            for LOOKBACK_WINDOW in "${LOOKBACK_WINDOWS[@]}"; do
              for FORECAST_HORIZON in "${FORECAST_HORIZONS[@]}"; do
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                  for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
                      for PATIENCE in "${PATIENCE[@]}"; do
                        for D_MODEL in "${D_MODELS[@]}"; do
                          for N_HEAD in "${N_HEADS[@]}"; do
                            for D_FF in "${D_FFS[@]}"; do
                              for ENCODER_LAYER in "${ENCODER_LAYERS[@]}"; do
                                for DECODER_LAYER in "${DECODER_LAYERS[@]}"; do
                                  for DROPOUT in "${DROPOUTS[@]}"; do
                                    for ADD_TEMPORAL_FEATURES in "${ADD_TEMPORAL_FEATURES_VALUES[@]}"; do
                                      for ENCODING_TYPE in "${ENCODING_TYPES[@]}"; do
                                        for COMPUTE_METRIC_WEIGHTS in "${COMPUTE_METRIC_WEIGHTS_VALUES[@]}"; do
                                          for VISUALIZE_ATTENTION in "${VISUALIZE_ATTENTION_VALUES[@]}"; do
                                            for BIAS_TYPE in "${BIAS_TYPES[@]}"; do

                                              LOCAL_WINDOW_SIZE=$(( FORECAST_HORIZON * 15 / 100 ))
                                              if [[ $LOCAL_WINDOW_SIZE -gt $LOOKBACK_WINDOW ]]; then
                                                LOCAL_WINDOW_SIZE=$LOOKBACK_WINDOW
                                              fi

                                              echo "Running Configuration:"
                                              echo "  Task: $TASK"
                                              echo "  Test Case: $TEST_CASE"
                                              echo "  Event: $EVENT"
                                              echo "  Target: $TARGET"
                                              echo "  Frequency: $FREQUENCY"
                                              echo "  Model: $MODEL"
                                              echo "  Add Temporal Features: $ADD_TEMPORAL_FEATURES"
                                              echo "  Encoding Type: $ENCODING_TYPE"
                                              echo "  Compute Metric Weights: $COMPUTE_METRIC_WEIGHTS"
                                              echo "  Visualize Attention: $VISUALIZE_ATTENTION"
                                              echo "  Batch Size: $BATCH_SIZE"
                                              echo "  Epochs: $EPOCH"
                                              echo "  Lookback Window: $LOOKBACK_WINDOW"
                                              echo "  Forecast Horizon: $FORECAST_HORIZON"
                                              echo "  Learning Rate: $LEARNING_RATE"
                                              echo "  Weight Decay: $WEIGHT_DECAY"
                                              echo "  Optimizer: $OPTIMIZER"
                                              echo "  Patience: $PATIENCE"
                                              echo "  d_model: $D_MODEL"
                                              echo "  n_heads: $N_HEAD"
                                              echo "  d_ff: $D_FF"
                                              echo "  Encoder Layers: $ENCODER_LAYER"
                                              echo "  Decoder Layers: $DECODER_LAYER"
                                              echo "  Dropout: $DROPOUT"
                                              echo "  Bias Type: $BIAS_TYPE"
                                              echo "  Local Window Size: $LOCAL_WINDOW_SIZE"
                                              echo "  Device: $DEVICE"

                                              python -u main.py \
                                                --task "$TASK" \
                                                --test_case "$TEST_CASE" \
                                                --model "$MODEL" \
                                                --frequency "$FREQUENCY" \
                                                --add_temporal_features "$ADD_TEMPORAL_FEATURES" \
                                                --encoding_type "$ENCODING_TYPE" \
                                                --compute_metric_weights "$COMPUTE_METRIC_WEIGHTS" \
                                                --visualize_attention "$VISUALIZE_ATTENTION" \
                                                --data_path "data/processed/$TASK/$FREQUENCY" \
                                                --target_name "$TARGET" \
                                                --batch_size "$BATCH_SIZE" \
                                                --epochs "$EPOCH" \
                                                --patience "$PATIENCE" \
                                                --learning_rate "$LEARNING_RATE" \
                                                --weight_decay "$WEIGHT_DECAY" \
                                                --optimizer "$OPTIMIZER" \
                                                --lookback_window "$LOOKBACK_WINDOW" \
                                                --forecast_horizon "$FORECAST_HORIZON" \
                                                --d_model "$D_MODEL" \
                                                --n_heads "$N_HEAD" \
                                                --d_ff "$D_FF" \
                                                --num_encoder_layers "$ENCODER_LAYER" \
                                                --num_decoder_layers "$DECODER_LAYER" \
                                                --dropout "$DROPOUT" \
                                                --bias_type "$BIAS_TYPE" \
                                                --local_window_size "$LOCAL_WINDOW_SIZE" \
                                                --device "$DEVICE" \
                                                --event "$EVENT"

                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "🌟 Execution Complete 🌟"