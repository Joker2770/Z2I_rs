# Z2I_rs

[![Release](https://github.com/Joker2770/Z2I_rs/actions/workflows/release.yml/badge.svg)](https://github.com/Joker2770/Z2I_rs/actions/workflows/release.yml)
[![Rust](https://github.com/Joker2770/Z2I_rs/actions/workflows/rust.yml/badge.svg)](https://github.com/Joker2770/Z2I_rs/actions/workflows/rust.yml)
[![Snap](https://github.com/Joker2770/Z2I_rs/actions/workflows/snap.yml/badge.svg)](https://github.com/Joker2770/Z2I_rs/actions/workflows/snap.yml)
[![z2i-rs](https://snapcraft.io/z2i-rs/badge.svg)](https://snapcraft.io/z2i-rs)

A Rust rewrite of Z2I. Z2I is a Gomoku/Renju AI based on a neural network and Monte Carlo Tree Search (MCTS); this project integrates board rules, MCTS, ONNX Runtime inference and the Gomocup/Piskvork engine protocol into a single Rust console program.

This project works well with Gomoku managers such as [qpiskvork](https://github.com/Joker2770/qpiskvork): the manager starts the engine, sends game commands and manages games, while `pbrain-Z2I_rs` computes and returns moves through standard input/output.

Related projects:

- [Joker2770/Z2I](https://github.com/Joker2770/Z2I): the original Z2I project.
- [Joker2770/qpiskvork](https://github.com/Joker2770/qpiskvork): Gomoku manager for human-vs-engine play, engine-vs-engine matches and game management.

## Features

- MCTS search combined with ONNX neural network policy and value functions.
- ONNX Runtime inference with batching and a background inference worker.
- CPU build available by default; CUDA as an optional Cargo feature.
- Supports FreeStyle, Standard, Renju, Caro and Standard+Caro rule flags.
- Implements the Gomocup/Piskvork-style console protocol: `START`, `BEGIN`, `TURN`, `BOARD`, `INFO`, `ABOUT`, `END`.
- Model path and MCTS simulation count configurable via `config.toml`.
- Includes programs for self-play data generation, model evaluation and evaluation against a random MCTS opponent.

## Building

Requires the Rust stable toolchain and Cargo.

### CPU

CPU is the default build:

```bash
cargo build --release --bin pbrain-Z2I_rs
```

### CUDA

Build with CUDA provider support:

```bash
cargo build --release --features cuda --bin pbrain-Z2I_rs
```

CUDA builds also require the host to have CUDA and related runtime libraries matching the ONNX Runtime/CUDA provider. Hosts without a GPU should use the default CPU build.

## Model & configuration

The program searches for `config.toml` in the following order and uses the first one found:

1. User config dir: Linux `$XDG_CONFIG_HOME/Z2I_rs/` (falling back to `~/.config/Z2I_rs/` when unset), macOS `~/Library/Application Support/Z2I_rs/`, Windows `%APPDATA%\Z2I_rs\`;
2. current working directory;
3. directory of the executable.

When none exists, the default model path and MCTS parameters in the source are used. For configs placed in the user directory it is recommended to write the model path as absolute (relative paths are still resolved against the current working dir / executable dir).

Example:

```toml
[model]
default_model = "models/free-style_15x15_889.onnx"
free_style_model = "models/free-style_15x15_889.onnx"
renju_model = "models/renju_15x15_592.onnx"
standard_model = "models/standard_15x15_535.onnx"
caro_model = "models/caro_15x15_532.onnx"
standard_caro_model = "models/standard_caro_15x15_533.onnx"

[MCTS]
# Total number of MCTS simulations for self-play and evaluation.
# For protocol games, this is the upper bound when time permits.
num_mct_sims = 500
# Number of simulations submitted and evaluated as one inference batch.
# Larger values can improve throughput but increase latency and memory use.
num_sim_per_batch = 8
# Print periodic search information while thinking.
open_mind = true
# Run background MCTS batches while waiting for the opponent.
enable_ponder = true
# Minimum remaining time (milliseconds) before starting a full batch.
# Tune for the target machine and model; increase it on slower systems.
time_reserve_ms = 1800
# Minimum remaining time (milliseconds) before starting one final simulation.
# Increase it if a single simulation is slow or the system is heavily loaded.
single_sim_reserve_ms = 400
# Time kept for applying and reporting the selected move (milliseconds).
final_move_reserve_ms = 100

[ONNXRUNTIME]
# Number of intra-op ONNX Runtime threads used for each inference session.
num_intra_thread = 4
```

Model files must be placed where the config specifies. The `NeuralNetwork` input tensor is currently fixed at `3x15x15`, and the ONNX models in the repo are exported for a 15x15 board; supporting other sizes requires changing the model input, the tensor conversion and the board configuration together.

### Provider selection

Provider initialization is currently controlled by Cargo features: CPU by default; with a CUDA build you can enable the CUDA provider in `src/ortcommon.rs`. For CPU-only hosts, simply use the build without `--features cuda`.

## Using with qpiskvork

When `qpiskvork` acts as the manager, configure the built engine as:

```text
pbrain-Z2I_rs
```

The program is a console process: it receives commands on stdin and writes responses to stdout. Use absolute paths for the engine and model files, since the manager may change the engine's working directory.

Example launch:

```bash
./target/release/pbrain-Z2I_rs
```

Manual protocol test:

```text
START 15
BEGIN
TURN 7,7
END
```

Normally `START` replies `OK`, and `BEGIN`/`TURN` reply a move coordinate in `x,y` format.

## Supported protocol commands

| Command | Meaning |
| --- | --- |
| `START size` | Create a board of the given size and initialize the engine. |
| `BEGIN` | Request the first move when the AI plays first. |
| `TURN x,y` | Inform the opponent's move and request the AI's move. |
| `BOARD` | Start receiving the full board; after `DONE`, request the AI's move. |
| `INFO rule value` | Set the rule flag. |
| `ABOUT` | Return the engine name and version. |
| `END` | End the process. |

Protocol details are in [Gomoku AI protocol.html](Gomoku%20AI%20protocol.html). The protocol requires the engine to flush stdout promptly; this project flushes output after processing a command.

## Rule flags

`INFO rule value` uses bitflag values:

| Rule | Value |
| --- | ---: |
| FreeStyle | `0` |
| Standard | `1` |
| Renju | `4` |
| Caro | `8` |

For example:

```text
INFO rule 4
```

selects the Renju rule. Rule flag combinations are parsed by the Rust side.

## Training & evaluation

Training/evaluation uses the separate `train_and_eval` binary:

```bash
cargo run --release --bin train_and_eval -- prepare
cargo run --release --bin train_and_eval -- generate 0
cargo run --release --bin train_and_eval -- eval_with_winner 10
cargo run --release --bin train_and_eval -- eval_with_random 10
```

Command descriptions:

- `prepare`: create `data/`, `weights/` and the weight state file.
- `generate <batch_id>`: load the current weight and generate self-play training data.
- `eval_with_winner <games>`: evaluate the current weight against the best weight.
- `eval_with_random <games>`: evaluate the current weight against a random MCTS opponent without a neural network.

Python training scripts are in `train/`; install dependencies with:

```bash
python3 -m pip install -r train/requirements.txt
```

The training flow needs the Python side to produce initial ONNX weights before the Rust side can run self-play and evaluation with a model.

### ORT Training

The Rust training entry needs ONNX Runtime Training artifacts, not ordinary inference models. The artifact directory must contain:

```text
training_model.onnx
eval_model.onnx
optimizer_model.onnx
checkpoint
```

After generating the artifacts, the Rust training entry can read the Rust self-play data in `data/`:

```bash
cargo run --features training --bin ort_train -- \
	artifacts data weights/1.onnx artifacts/checkpoint.updated 128 1
```

Argument order: `artifact_dir data_dir output_model checkpoint batch_size epochs`. Default training input names are `board`, `target_p`, `target_v`; if the training graph uses other names, pass three more name arguments. Ordinary `weights/*.onnx` supports inference only and cannot be used directly as ORT Training artifacts.

## Inference & throughput

Each MCTS simulation sends its inference request to the background ONNX worker. The worker batches `session.run()` calls once the batch reaches `batch_size` or the short wait window ends, then returns results in FIFO request order through each response channel.

Therefore, a larger batch size may improve GPU throughput, but increases the per-request latency and memory usage. The MCTS simulation count is controlled by `num_mct_sims`.

## Snap

A snap named `z2i-rs` is published on snapcraft.io; the packaging lives in `snap/snapcraft.yaml`. It ships the engine binary only and builds from the GitHub main branch. No weights are bundled — after installing, copy `snap/config.toml` to `~/snap/z2i-rs/current/.config/Z2I_rs/config.toml` and point the model paths at your own ONNX weight.

Build locally (requires lxd):

```bash
sudo snap install lxd && sudo lxd init --auto
snapcraft pack
```

Or build in destructive mode (not recommended for release):

```bash
snapcraft pack --destructive-mode
```

## Development checks

```bash
cargo fmt --all
cargo check --all-targets
cargo test --all-targets
```

## License

This project uses the MIT License; see [LICENSE](LICENSE).