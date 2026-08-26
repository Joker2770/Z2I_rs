#!/usr/bin/env bash
# 自动化训练流水线:generate -> 训练 -> 验收评估,循环执行
#
# 流程(AlphaZero 风格):
#   1. generate:用已通过验收的 best 权重自对弈生成数据
#   2. learner.py:训练得到候选权重 current+1
#   3. eval_with_winner:候选 vs best,胜率(和棋计 0.5)> UPDATE_THRESHOLD 则 best 更新,
#      否则回退 current=best,下一轮从 best 重新训练
#   4. 每场评估赛后按标准 Elo 公式更新双方评级,持久化到 elo.txt 并写入 eval_result.log
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
NUM_CONTEST="${NUM_CONTEST:-10}"
BATCH_ID="${BATCH_ID:-0}"
MAX_ITERS="${MAX_ITERS:-1000}"
PYTHON="${PYTHON:-python3}"
# Colab 上使用 load-dynamic 的 CUDA 版 onnxruntime 时,取消注释并指向 libonnxruntime.so
# export ORT_LIB_LOCATION=/path/to/libonnxruntime.so
# 每 CHECK_FREQ 轮做一次验收评估(1=每轮评估);中间轮跳过评估、直接接纳候选
# Colab T4 会话有时限,默认隔轮评估以缩短单轮耗时
CHECK_FREQ="${CHECK_FREQ:-2}"
# 每轮生成 NUM_2_SELF_PLAY 局(src/configuration.rs,当前 16),批次 id 步进与之保持一致
STEP="${STEP:-16}"

cd "$WORK_DIR"

# learner.py 使用的训练工作目录绝对路径(未设置时取当前目录),与 WORK_DIR 对齐
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
