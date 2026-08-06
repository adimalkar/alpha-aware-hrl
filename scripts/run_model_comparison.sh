#!/bin/bash
# Run model comparison experiments
# This script trains multiple model architectures and compares their performance

cd "/home/aditya/Downloads/Alpha-Aware Hierarchical Reinforcement Learning/alpha-aware-hrl"
source venv/bin/activate

echo "=============================================="
echo "LOB Model Comparison Experiments"
echo "=============================================="
echo ""

# Create checkpoints directory
mkdir -p checkpoints

# Common settings
EPOCHS=30
BATCH_SIZE=64
SEQ_LEN=50
D_MODEL=128
N_LAYERS=4
MAX_SAMPLES=100000  # Use 100k samples for reasonable training time

echo "Settings: epochs=$EPOCHS, batch=$BATCH_SIZE, seq_len=$SEQ_LEN, d_model=$D_MODEL"
echo ""

# Train LSTM
echo "=============================================="
echo "1. Training LSTM..."
echo "=============================================="
python scripts/train_lob_classifier.py \
    --model lstm \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --max_samples $MAX_SAMPLES

echo ""

# Train BiLSTM
echo "=============================================="
echo "2. Training BiLSTM..."
echo "=============================================="
python scripts/train_lob_classifier.py \
    --model bilstm \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --max_samples $MAX_SAMPLES

echo ""

# Train TCN
echo "=============================================="
echo "3. Training TCN..."
echo "=============================================="
python scripts/train_lob_classifier.py \
    --model tcn \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --max_samples $MAX_SAMPLES

echo ""

# Train Transformer
echo "=============================================="
echo "4. Training Transformer..."
echo "=============================================="
python scripts/train_lob_classifier.py \
    --model transformer \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --max_samples $MAX_SAMPLES

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Check checkpoints/ for saved models"
echo "=============================================="
