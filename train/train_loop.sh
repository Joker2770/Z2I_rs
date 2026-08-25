#!/usr/bin/env bash
# 自动化训练流水线:generate -> 训练 -> 验收评估,循环执行
#
# 流程(AlphaZero 风格):
#   1. generate:用已通过验收的 best 权重自对弈生成数据
#   2. learner.py:训练得到候选权重 current+1
#   3. eval_with_winner:候选 vs best,胜率(和棋计 0.5)> UPDATE_THRESHOLD 则 best 更新,
#      否则回退 current=best,下一轮从 best 重新训练
#
# 环境变量:
#   WORK_DIR    训练工作目录(含 train_and_eval 二进制与 data/weights),默认 build
#   BIN         train_and_eval 二进制路径,默认 ./train_and_eval
#   NUM_CONTEST 验收评估局数,默认 20
#   BATCH_ID    起始批次 id,默认 0
#   MAX_ITERS   最大迭代轮数,默认 1000
#   PYTHON      python 解释器,默认 python3
set -euo pipefail

WORK_DIR="${WORK_DIR:-build}"
BIN="${BIN:-./train_and_eval}"
NUM_CONTEST="${NUM_CONTEST:-20}"
BATCH_ID="${BATCH_ID:-0}"
MAX_ITERS="${MAX_ITERS:-1000}"
PYTHON="${PYTHON:-python3}"
# 每轮生成 NUM_2_SELF_PLAY 局(src/configuration.rs),批次 id 步进与之保持一致
STEP=10

cd "$WORK_DIR"

for ((iter=1; iter<=MAX_ITERS; iter++)); do
    echo "===== iter $iter: generate batch $BATCH_ID ====="
    "$BIN" generate "$BATCH_ID"
    BATCH_ID=$((BATCH_ID + STEP))

    echo "===== iter $iter: train ====="
    (cd ../train && "$PYTHON" learner.py train)

    echo "===== iter $iter: eval vs best ($NUM_CONTEST games) ====="
    "$BIN" eval_with_winner "$NUM_CONTEST"
done
