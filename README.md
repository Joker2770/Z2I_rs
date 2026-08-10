# Z2I_rs

Z2I 的 Rust 重写版本。Z2I 是一个基于神经网络和 Monte Carlo Tree Search（MCTS）的 Gomoku/Renju AI；本项目将棋盘规则、MCTS、ONNX Runtime 推理和 Gomocup/Piskvork 引擎协议整合为一个 Rust 控制台程序。

本项目适合与 [qpiskvork](https://github.com/Joker2770/qpiskvork) 这类 Gomoku manager 配合使用：manager 负责启动引擎、发送棋局命令和管理对局，`pbrain-Z2I_rs` 负责通过标准输入输出计算并返回落子。

相关项目：

- [Joker2770/Z2I](https://github.com/Joker2770/Z2I)：原始 Z2I 项目。
- [Joker2770/qpiskvork](https://github.com/Joker2770/qpiskvork)：Gomoku manager，可用于人机对战、引擎对战和棋局管理。

## 功能

- MCTS 搜索结合 ONNX 神经网络策略和值函数。
- ONNX Runtime 推理支持批处理和后台推理 worker。
- CPU 构建默认可用；CUDA 作为可选 Cargo feature。
- 支持 FreeStyle、Standard、Renju、Caro 及 Standard+Caro 规则标志。
- 实现 Gomocup/Piskvork 风格的控制台协议：`START`、`BEGIN`、`TURN`、`BOARD`、`INFO`、`ABOUT`、`END`。
- 支持通过 `config.toml` 配置模型路径和 MCTS 模拟次数。
- 包含自对弈数据生成、模型评估和随机 MCTS 对手评估程序。

## 构建

需要 Rust stable toolchain 和 Cargo。

### CPU

CPU 是默认构建方式：

```bash
cargo build --release --bin pbrain-Z2I_rs
```

### CUDA

编译 CUDA provider 支持：

```bash
cargo build --release --features cuda --bin pbrain-Z2I_rs
```

CUDA 运行还需要主机安装与 ONNX Runtime/CUDA provider 匹配的 CUDA 和相关运行库。没有 GPU 的主机请使用默认 CPU 构建。

## 模型与配置

程序会在当前工作目录和可执行文件所在目录查找 `config.toml`。没有配置文件时使用源码中的默认模型路径和 MCTS 参数。

示例：

```toml
[model]
default_model = "models/free-style_15x15_889.onnx"
free_style_model = "models/free-style_15x15_889.onnx"
renju_model = "models/renju_15x15_592.onnx"
standard_model = "models/standard_15x15_535.onnx"
caro_model = "models/caro_15x15_532.onnx"
standard_caro_model = "models/standard_caro_15x15_533.onnx"

[MCTS]
num_mct_sims = 500
```

模型文件必须放在配置指定的位置。当前 `NeuralNetwork` 的输入张量固定为 `3x15x15`，仓库中的 ONNX 模型也按 15x15 棋盘导出；如需支持其他尺寸，需要同时修改模型输入、张量转换和棋盘配置。

### Provider 选择

当前 provider 初始化由 Cargo feature 控制：默认使用 CPU；使用 CUDA 构建时可在 `src/ortcommon.rs` 中启用 CUDA provider。若部署到 CPU-only 主机，直接使用不带 `--features cuda` 的构建即可。

## 与 qpiskvork 配合使用

`qpiskvork` 作为 manager 时，应将编译出的引擎配置为：

```text
pbrain-Z2I_rs
```

程序是控制台进程，通过 stdin 接收命令，通过 stdout 输出响应。建议使用绝对路径配置引擎和模型文件，因为 manager 可能会修改引擎的工作目录。

示例启动：

```bash
./target/release/pbrain-Z2I_rs
```

手工测试协议：

```text
START 15
BEGIN
TURN 7,7
END
```

正常情况下，`START` 返回 `OK`，`BEGIN` 和 `TURN` 返回 `x,y` 格式的落子坐标。

## 支持的协议命令

| 命令 | 作用 |
| --- | --- |
| `START size` | 创建指定尺寸的棋盘并初始化引擎。 |
| `BEGIN` | AI 先手时请求第一步。 |
| `TURN x,y` | 告知对手落子，并请求 AI 落子。 |
| `BOARD` | 开始发送完整棋盘；以 `DONE` 结束后请求 AI 落子。 |
| `INFO rule value` | 设置规则标志。 |
| `ABOUT` | 返回引擎名称和版本。 |
| `END` | 结束进程。 |

协议细节参见仓库中的 [Gomoku AI protocol.html](Gomoku%20AI%20protocol.html)。该协议要求引擎及时刷新 stdout；本项目在处理命令后会刷新输出。

## 规则标志

`INFO rule value` 使用 bitflags 数值：

| 规则 | 值 |
| --- | ---: |
| FreeStyle | `0` |
| Standard | `1` |
| Renju | `4` |
| Caro | `8` |

例如：

```text
INFO rule 4
```

表示使用 Renju 规则。规则标志的组合由 Rust 端解析。

## 训练与评估

训练/评估使用单独的 `train_and_eval` 二进制：

```bash
cargo run --release --bin train_and_eval -- prepare
cargo run --release --bin train_and_eval -- generate 0
cargo run --release --bin train_and_eval -- eval_with_winner 10
cargo run --release --bin train_and_eval -- eval_with_random 10
```

命令说明：

- `prepare`：创建 `data/`、`weights/` 和权重状态文件。
- `generate <batch_id>`：加载当前权重并生成自对弈训练数据。
- `eval_with_winner <games>`：评估当前权重与最佳权重。
- `eval_with_random <games>`：评估当前权重与无神经网络的随机 MCTS 对手。

Python 训练脚本位于 `python/`，依赖可通过以下命令安装：

```bash
python -m pip install -r python/requirements.txt
```

训练流程需要 Python 侧生成初始 ONNX 权重，然后 Rust 端才能进行带模型的自对弈和评估。

## 推理与吞吐量

MCTS simulation 会将推理请求发送到后台 ONNX worker。worker 会在达到 `batch_size` 或短暂等待窗口结束后批量调用 `session.run()`，再按请求进入队列的 FIFO 顺序通过各自的 response channel 返回结果。

因此，增大 batch size 可能提高 GPU 吞吐量，但会增加单个请求的等待时间和显存占用。MCTS 模拟次数由 `num_mct_sims` 控制。

## 开发检查

```bash
cargo fmt --all
cargo check --all-targets
cargo test --all-targets
```

## License

本项目使用 MIT License，详见 [LICENSE](LICENSE)。