#!/usr/bin/env bash
# Automated training pipeline: generate -> train -> acceptance evaluation, looped
#
# Flow (AlphaZero style):
#   1. generate: self-play with the accepted best weight to produce data
#   2. learner.py: train to produce candidate weight current+1
#   3. eval_with_winner: candidate vs best; if win rate (draws count as 0.5) > UPDATE_THRESHOLD, best is updated,
#      otherwise current reverts to best and the next round retrains from best
#   4. After each evaluation match, both ratings are updated with the standard Elo formula,
#      persisted to elo.txt and appended to eval_result.log
#
# Environment variables:
#   WORK_DIR     training work dir (contains the train_and_eval binary and data/weights), default build
#   BIN          train_and_eval binary path, default ./train_and_eval
#   NUM_CONTEST  acceptance evaluation game count, default 20
#   BATCH_ID     starting batch id, default 0
#   MAX_ITERS    max iterations, default 1000
#   PYTHON       python interpreter, default python3
set -euo pipefail

WORK_DIR="${WORK_DIR:-build}"
BIN="${BIN:-./train_and_eval}"
NUM_CONTEST="${NUM_CONTEST:-10}"
BATCH_ID="${BATCH_ID:-0}"
MAX_ITERS="${MAX_ITERS:-1000}"
PYTHON="${PYTHON:-python3}"
# When using the load-dynamic CUDA onnxruntime on Colab, uncomment and point to libonnxruntime.so
# export ORT_LIB_LOCATION=/path/to/libonnxruntime.so
# Run acceptance evaluation every CHECK_FREQ rounds (1 = every round); rounds in between skip
# evaluation and accept the candidate directly
# Colab T4 sessions are time-limited; evaluate every other round by default to shorten each round
CHECK_FREQ="${CHECK_FREQ:-2}"
# Each round generates NUM_2_SELF_PLAY games (src/configuration.rs, currently 16);
# keep the batch id step in sync with it
STEP="${STEP:-16}"

cd "$WORK_DIR"

# absolute training work dir used by learner.py (current dir if unset), aligned with WORK_DIR
BUILD_DIR="${BUILD_DIR:-$PWD}"
export BUILD_DIR

for ((iter=1; iter<=MAX_ITERS; iter++)); do
    echo "===== iter $iter: generate batch $BATCH_ID ====="
    "$BIN" generate "$BATCH_ID"
    BATCH_ID=$((BATCH_ID + STEP))

    echo "===== iter $iter: train ====="
    (cd ../train && "$PYTHON" learner.py train)

    echo "===== iter $iter: eval vs best ($NUM_CONTEST games) ====="
    if (( iter % CHECK_FREQ == 0 )); then
        "$BIN" eval_with_winner "$NUM_CONTEST"
    else
        echo "skip eval (CHECK_FREQ=$CHECK_FREQ): accept candidate directly"
        if read -r CUR _ < current_and_best_weight.txt; then
            printf '%s %s\n' "$CUR" "$CUR" > current_and_best_weight.txt
        fi
    fi
done
