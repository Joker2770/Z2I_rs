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
- Tic-Tac-Toe mode is available through the `tic-tac-toe` Cargo feature.
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

### Tic-Tac-Toe

Build with the 3x3 board and three-in-a-row defaults:

```bash
cargo build --release --features tic-tac-toe --bin pbrain-Z2I_rs
```

The same feature must be used for `train_and_eval` and `ort_train` so that
self-play, evaluation and training tensors use the matching `3x3`/`9` shapes.
The ONNX model must also be exported for a 3x3 board.

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

Model files must be placed where the config specifies. The default build expects
`4x15x15` input tensors and the `tic-tac-toe` build expects `4x3x3`; the ONNX
model must be exported for the selected board size. The four channels are
current player's stones, opponent's stones, last-move marker, and a constant
side-to-move color plane (+1 Black / -1 White), respectively.

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

Normally `START` replies `OK`, and `BEGIN`/`TURN` reply a move coordinate in `x,y` format. Commands that ask for a move but cannot be served are answered with an `ERROR <reason>` line on stdout, so a manager waiting for a reply never hangs. Besides responses, stdout can carry `MESSAGE`/`ERROR`/`DEBUG` lines that are not tied to a request; a manager should skip lines it does not understand.

## Supported protocol commands

| Command | Meaning |
| --- | --- |
| `START size` | Create a board of the given size and initialize the engine. |
| `BEGIN` | Request the first move when the AI plays first. |
| `TURN x,y` | Inform the opponent's move and request the AI's move. |
| `BOARD` | Start receiving the full board; after `DONE`, request the AI's move. |
| `INFO rule value` | Set the rule flag (bit 1 = exactly-five, bit 2 = continuous game, bit 4 = Renju, bit 8 = Caro). |
| `INFO timeout_turn ms` | Per-move time limit; the engine stops searching once the deadline is reached. |
| `INFO time_left ms` | Remaining match time announced by the manager; it caps the move as well and is authoritative for the move that follows it. |
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

## Continuous game (self-play, `INFO rule` bit 2)

`INFO rule` bit 2 puts the engine into a *continuous game*: it plays both colors itself, without an opponent, and reports every move to the manager on stdout. This suits custom managers and board viewers that consume a plain coordinate stream.

```text
INFO rule 2
START 15
BEGIN
```

Behavior:

- `INFO rule 2` enables the mode; the other bits still select the win rule, so `INFO rule 2` alone keeps FreeStyle, and `INFO rule 10` selects Caro *and* continuous game. An `INFO rule` value without bit 2 turns the mode off again.
- `BEGIN` opens the move stream; the engine replies with the first coordinate and then keeps playing both colors until the game is over. While the stream runs, each reply is one `x,y` line on stdout, and `MESSAGE continuous game finished` is written to stdout when the game ends.
- After the game is over the engine waits. A new `START` starts a new game and the stream resumes without another `BEGIN`; a `BEGIN` sent while no continuous game is in progress is refused with `ERROR cannot begin: no continuous game in progress` on stdout.
- `TURN` is ignored during the mode (the engine has no opponent), and pondering is disabled because the engine is always the side to move. The refusal is reported as `ERROR TURN ignored during continuous game` on stdout, so a manager that sent `TURN` gets an answer instead of waiting forever.
- Because no manager refreshes the clock per move, the engine deducts its own thinking time from the last `INFO time_left` value; `INFO timeout_turn` still caps each move. An `INFO time_left` announcement received before a move is authoritative and skips that deduction for that move, so a manager that refreshes the clock before every move and one that never refreshes it both get correct accounting.
- Announcements are handled in command order, so a manager that reacts to a reported move reaches the engine while the *next* move is already being searched: the update applies from the move after that one. The engine never waits for an announcement, and the manager's value always replaces the locally deducted clock.
- In `TURN`-driven mode the engine never deducts locally: the `INFO time_left` a manager sends before each move is used as-is, so every move gets the announced budget.
- stdout carries everything a manager has to react to: the move coordinates, `OK` for a successful `START`, `ERROR <reason>` when a command that expects a move cannot be served (`START` with an unsupported size, a `BEGIN` that cannot open the game, an unplayable `TURN` or `BOARD` position, a `TURN` ignored during a continuous game), `MESSAGE continuous game finished` when the self-play stream reaches the end of the game, `ERROR continuous game stopped` when the engine cannot continue it, and optional `DEBUG thinking ...` lines when `open_mind` is enabled in the configuration. stderr carries only diagnostics such as the configuration and model loading messages.

Note that spontaneous multi-line output is outside the strict request/response pattern of the piskvork protocol, so a stock manager such as qpiskvork will lose sync. Use this mode with a manager built for it.

An `INFO rule` value carrying bit 2 switches the mode on immediately, even in the middle of an ordinary game that already has stones on the board. Such a manager then gets `ERROR TURN ignored during continuous game` for the very next `TURN` instead of a move. Send `INFO rule` without bit 2 (for example `INFO rule 1`) to hand the game back to `TURN` control; the win rule itself still changes only from the next `START`.

## Training & evaluation

Training/evaluation uses the separate `train_and_eval` binary:

```bash
cargo run --release --bin train_and_eval -- prepare
cargo run --release --bin train_and_eval -- generate 0
cargo run --release --bin train_and_eval -- eval_with_winner 10
cargo run --release --bin train_and_eval -- eval_with_random 10
```

Command descriptions:

- `prepare`: create `data/`, `weights/`, the weight state file and a default `openings.txt`.
- `generate <batch_id>`: load the current weight and generate self-play training data.
- `eval_with_winner <games>`: evaluate the current weight against the best weight.
- `eval_with_random <games>`: evaluate the current weight against a random MCTS opponent without a neural network.
- `verify_weight <id>`: run the weight probe on `weights/<id>.onnx` and exit 1 if it fails.

### Weight probe (`verify_weight`)

A destroyed weight is not a rare accident: a training round on corrupt targets, a structure
conversion whose distillation went wrong, or a stale ONNX companion file all produce a
model that loads and plays while being no better than a random policy. Such a model loses
every evaluation game, which is expensive to discover and easy to mistake for a real
regression.

The probe asks a weight three questions with **one forward pass per position** (about a
second in total, no search):

```
$ train_and_eval verify_weight 1205
weight probe: PASS (0.9s, policy sharpness 0.772)
  outputs            ok    6 position(s), |v| max 1.000
  win in one         ok    3/3 solved: horizontal gap p=1.000, vertical gap p=0.998, diagonal gap p=0.999
  block the four     ok    top1 156 p=0.833 (warn only)
  value (winning)    ok    v=+1.000
  value (losing)     ok    v=-1.000
```

- **win in one** (3 shapes): the side to move has exactly one immediate five; the raw
  policy top-1 must be it. A uniform policy answers index 0 everywhere and solves none.
- **block the four**: the opponent threatens five and the block is the only move that does
  not lose at once. Advisory only -- a policy preferring a counter-threat is weak, not
  broken, so this never rejects a candidate on its own.
- **value signs**: a won position must evaluate positive and a lost one negative, and
  decisively so (|v| >= 0.5), which catches a flat or inverted value head.
- **policy sharpness** (the number in the headline) is the mean top-1 probability over the
  probes. It is the one figure that separates "weak" from "destroyed": a destroyed weight
  returns the uniform 1/225, while a merely weaker weight keeps its tactics with visibly
  lower confidence (a sharp model answers the block probe near 0.8, a diffuse one near 0.2).
  The per-shape probabilities are always printed, so a worse candidate can be compared to
  best instead of being guessed at.
- Every probe position is parity-correct (`b == w` on Black's turn, `b == w + 1` on White's
  turn) with a harmless Black stone played last, and always White to move: White has no
  forbidden moves under any supported rule, so the expected move is rule-agnostic.

`eval_with_winner` runs the probe on the candidate automatically and **rejects the
candidate before playing any games** when it fails; it also probes best and prints its
headline every round, so the candidate's sharpness can be read against the incumbent's. Set
`EVAL_SKIP_VERIFY=1` to bypass the gate, e.g. to time an evaluation or to study a known-bad
weight.

Note that a 128-sim screen is shallow (only 8 inference batches per move), so it punishes a
diffuse policy harder than a real game would: a candidate that passes the probe but scores
far below best should be re-checked at a higher `EVAL_SIMS` (256 or 384 still costs only
seconds per pair) before its weakness is treated as a real regression.

### Colour-paired evaluations

Acceptance evaluation always gives Black the first move, so a rule can be colour
symmetric and still not be colour symmetric *as a sampled distribution*: the reachable
positions are only closed under a colour swap inside the mirror universe where White
moves first. `eval_with_winner` therefore starts its games from an opening book and
plays every opening **twice with the colours exchanged**:

- `openings.txt` in the training work dir holds one opening per line: comma or
  whitespace separated board indices (`row * board_size + col`), colours alternating
  Black, White, ... so the count must be even. `#` starts a comment and unusable lines
  are reported and skipped.
- The even ply count is what makes the colour-swapped twin a legal "Black to move"
  position, so the first-move advantage cancels *inside* a pair and the pair score is a
  fair strength estimate. A file with no usable opening falls back to the built-in book.
- The book is spread over the eight board symmetries, so a small book still yields many
  distinct pairs. Odd `games` values leave the last game unpaired: it still counts in the
  win rate but is excluded from the pair statistic.
- Each evaluation logs the per-pair scores plus a two-sided exact sign test over the
  decisive pairs, which is a much sharper statement than the raw win rate over the same
  number of independent games. Promotion still uses the existing
  `UPDATE_THRESHOLD` rule on the overall score, so the log reports both.

### Evaluation cost knobs

`eval_with_winner` / `eval_with_random` print and log a cost line that separates session
creation from actual play, e.g.

```
eval cost: 25 pairs / 50 games, load 0.8s + games 98.4s = 99.2s, 4.0s per pair total / 3.9s per pair of play, 128 sims/move (EVAL_SIMS), 2 worker(s)
```

Use it to budget a session instead of guessing:

- **`load` is per-process and does not scale with the game count**, so it dominates short
  evaluations and disappears in long ones. A large `load` is a cold ONNX Runtime/CUDA
  start or a heavy weight file (loading the runtime or the weights from a network drive
  is a common cause) — not an expensive search. `games` is the part that grows with
  `games`/`EVAL_SIMS`.
- **`EVAL_SIMS=<n>`**: pin the simulations per move. Without it the evaluation follows
  `sims_for_weight`, which grows to `SIMS_CAP` — a late-generation evaluation then costs
  up to 3x a generation-0 one for the same games. Both sides always get the same value.
  Note that the MCTS floor is one batch (`DEFAULT_SIM_PER_BATCH_NUM`), so budgets below
  that batch size all cost the same.
- **`EVAL_WORKERS=<n>`** (default 2): play `n` games concurrently. Workers are independent
  and each owns its own ONNX sessions, so the merged result does not depend on `n` — it is
  a pure wall-clock lever. Board rendering is disabled when `n > 1` (interleaved boards are
  unreadable); set `EVAL_WORKERS=1` to keep them. Measure both settings with the cost line
  before fixing a value.
- Evaluating a weight against itself (`current == best`, which happens on a re-run right
  after a rollback) is skipped instead of burning a round and jittering the shared Elo
  rating with search noise. `EVAL_ALLOW_SELF_MATCH=1` runs it anyway: a self-match is the
  cheapest control that the harness is sound, because it must score ~0.5 at any budget.
- A failed evaluation (missing or unloadable weight, a weight whose input channels do not
  match `INPUT_CHANNEL_SIZE`, or a game that could not be played) reports zero games,
  rejects the candidate and leaves best in place. The failure is contained in the worker,
  so a bad weight no longer kills the process mid-run.

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

Argument order: `artifact_dir data_dir output_model checkpoint batch_size epochs`. The
default training input names are `board`, `target_p`, `target_v`, matching the
Python model; exported inference outputs are `P` and `V`. If the training graph
uses other names, pass three more name arguments. Each complete self-play file
is expanded with the same eight rotations/flips as the Python learner.
Incomplete or corrupted files are skipped. Ordinary `weights/*.onnx` supports
inference only and cannot be used directly as ORT Training artifacts.

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