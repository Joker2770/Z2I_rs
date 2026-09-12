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
# Environment variables (all overridable from the shell, e.g. EVAL_SIMS=384 ./train_loop.sh):
#   WORK_DIR     training work dir (contains the train_and_eval binary and data/weights), default build
#   BIN          train_and_eval binary path, default ./train_and_eval
#   PYTHON       python interpreter, default python3
#   BATCH_ID     starting batch id, default 0
#   MAX_ITERS    max iterations, default 1000
#   NUM_CONTEST  acceptance evaluation game count, default 50 (even = complete pairs)
#   CHECK_FREQ   evaluate every N rounds, default 1 (every round; no silent acceptance)
#   EVAL_SIMS    simulations per move for evaluation, default 256
#   EVAL_WORKERS games played concurrently during evaluation, default 2
#   REPLAY_INCLUDE_ARCHIVE  1 (default) lets the learner fill a short replay window from
#                data_archive/, 0 trains only on data/ + data_backup/
#   EVAL_SKIP_VERIFY=1   bypass the weight probe (diagnostics only)
#   EVAL_ALLOW_SELF_MATCH=1  evaluate a weight against itself (diagnostics only)
#   STEP         batch id step per round, default 16 (= NUM_2_SELF_PLAY in src/configuration.rs)
set -euo pipefail

WORK_DIR="${WORK_DIR:-build}"
BIN="${BIN:-./train_and_eval}"
BATCH_ID="${BATCH_ID:-0}"
MAX_ITERS="${MAX_ITERS:-1000}"
PYTHON="${PYTHON:-python3}"
# When using the load-dynamic CUDA onnxruntime on Colab, uncomment and point to libonnxruntime.so
# export ORT_LIB_LOCATION=/path/to/libonnxruntime.so

# --- acceptance evaluation -----------------------------------------------------------------
# 50 games at EVAL_SIMS=256 is about 85s of play plus ~1s of weight probing: shallow enough
# to stay cheap, deep enough not to punish a merely flat policy. See README "Evaluation cost
# knobs" for the measured numbers, and raise EVAL_SIMS (384) when a candidate's loss needs to
# be trusted as a real strength gap.
NUM_CONTEST="${NUM_CONTEST:-50}"
EVAL_SIMS="${EVAL_SIMS:-256}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
export EVAL_SIMS EVAL_WORKERS
# evaluate every round: skipping a round accepts the candidate without playing a game
CHECK_FREQ="${CHECK_FREQ:-1}"

# --- replay window -------------------------------------------------------------------------
# A short replay window is what quietly flattens a policy, so let the learner fall back to
# data_archive/ when data/ + data_backup/ cannot fill the window. In steady state the window
# is already full and this never triggers; set to 0 for a strictly recency-only window.
REPLAY_INCLUDE_ARCHIVE="${REPLAY_INCLUDE_ARCHIVE:-1}"
export REPLAY_INCLUDE_ARCHIVE

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
