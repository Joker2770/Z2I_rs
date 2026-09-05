#![deny(deprecated)]

mod configuration;

use configuration::cfg;
use ndarray::{Array2, Array4};
use ort::{memory::Allocator, session::Session, training::Trainer, value::Tensor};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, Read},
    mem,
    path::{Path, PathBuf},
};

const BOARD_SIZE: usize = cfg::BOARD_SIZE as usize;
const ACTION_SIZE: usize = BOARD_SIZE * BOARD_SIZE;
const BOARD_INPUT_NAME: &str = "board";
const POLICY_TARGET_NAME: &str = "target_p";
const VALUE_TARGET_NAME: &str = "target_v";
const POLICY_OUTPUT_NAME: &str = "P";
const VALUE_OUTPUT_NAME: &str = "V";

struct Sample {
    board: [i32; ACTION_SIZE],
    policy: [f32; ACTION_SIZE],
    value: f32,
    current_player: i32,
    last_action: i32,
}

fn read_i32(file: &mut File) -> io::Result<i32> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_f32(file: &mut File) -> io::Result<f32> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_samples(path: &Path) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let count = read_i32(&mut file)?;
    if count <= 0 {
        return Err(format!("invalid sample count {} in {}", count, path.display()).into());
    }
    let bytes_per_sample = (ACTION_SIZE * 2 + 3)
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or("sample size overflow")?;
    let expected_size = 4usize
        .checked_add(
            (count as usize)
                .checked_mul(bytes_per_sample)
                .ok_or("file size overflow")?,
        )
        .ok_or("file size overflow")?;
    let file_size = file.metadata()?.len();
    if file_size < expected_size as u64 {
        return Err(format!(
            "incomplete data file {}: step={}, size={}, expected={}",
            path.display(),
            count,
            file_size,
            expected_size
        )
        .into());
    }

    let mut boards = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut board = [0; ACTION_SIZE];
        for cell in &mut board {
            *cell = read_i32(&mut file)?;
        }
        boards.push(board);
    }

    let mut policies = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut policy = [0.0; ACTION_SIZE];
        for probability in &mut policy {
            *probability = read_f32(&mut file)?;
        }
        policies.push(policy);
    }

    let mut values = Vec::with_capacity(count as usize);
    let mut players = Vec::with_capacity(count as usize);
    let mut actions = Vec::with_capacity(count as usize);
    for _ in 0..count {
        values.push(read_i32(&mut file)? as f32);
    }
    for _ in 0..count {
        players.push(read_i32(&mut file)?);
    }
    for _ in 0..count {
        actions.push(read_i32(&mut file)?);
    }

    let mut samples = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        samples.push(Sample {
            board: boards[index],
            policy: policies[index],
            value: values[index],
            current_player: players[index],
            last_action: actions[index],
        });
    }
    Ok(samples)
}

fn load_data(directory: &Path) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut paths = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    paths.sort();

    let mut samples = Vec::new();
    for path in paths {
        match read_samples(&path) {
            Ok(file_samples) => {
                for sample in file_samples {
                    samples.extend(symmetries(sample));
                }
            }
            Err(error) => eprintln!("skip corrupted data file {}: {error}", path.display()),
        }
    }
    Ok(samples)
}

fn transformed_position(position: usize, rotation: usize, flip: bool) -> usize {
    let mut row = position / BOARD_SIZE;
    let mut column = position % BOARD_SIZE;
    for _ in 0..rotation {
        (row, column) = (BOARD_SIZE - 1 - column, row);
    }
    if flip {
        column = BOARD_SIZE - 1 - column;
    }
    row * BOARD_SIZE + column
}

fn symmetries(sample: Sample) -> Vec<Sample> {
    let mut result = Vec::with_capacity(8);
    for rotation in 1..=4 {
        for flip in [false, true] {
            let mut board = [0; ACTION_SIZE];
            let mut policy = [0.0; ACTION_SIZE];
            for position in 0..ACTION_SIZE {
                let transformed = transformed_position(position, rotation, flip);
                board[transformed] = sample.board[position];
                policy[transformed] = sample.policy[position];
            }
            let last_action = if sample.last_action >= 0 {
                transformed_position(sample.last_action as usize, rotation, flip) as i32
            } else {
                -1
            };
            result.push(Sample {
                board,
                policy,
                value: sample.value,
                current_player: sample.current_player,
                last_action,
            });
        }
    }
    result
}

#[inline]
fn player_channels(current_player: i32, stone: i32) -> (f32, f32) {
    // `current_player` and `stone` are both +1 (Black) / -1 (White) / 0 (empty),
    // so their product is positive when the stone belongs to the current player
    // and negative when it belongs to the opponent.
    let own = if current_player * stone > 0 { 1.0 } else { 0.0 };
    let opponent = if current_player * stone < 0 { 1.0 } else { 0.0 };
    (own, opponent)
}

fn make_batch(
    samples: &[Sample],
) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), Box<dyn Error>> {
    let mut states = Array4::<f32>::zeros((samples.len(), 3, BOARD_SIZE, BOARD_SIZE));
    let mut policies = Array2::<f32>::zeros((samples.len(), ACTION_SIZE));
    let mut values = Array2::<f32>::zeros((samples.len(), 1));

    for (batch_index, sample) in samples.iter().enumerate() {
        for position in 0..ACTION_SIZE {
            // Channel 0 holds the current player's own stones and channel 1 the
            // opponent's stones, matching the inference-side transform in
            // ortopt.rs and the Python trainer (_data_convert): the board is
            // always encoded from the mover's perspective.
            let (own_channel, opponent_channel) =
                player_channels(sample.current_player, sample.board[position]);
            let row = position / BOARD_SIZE;
            let column = position % BOARD_SIZE;
            states[[batch_index, 0, row, column]] = own_channel;
            states[[batch_index, 1, row, column]] = opponent_channel;
            policies[[batch_index, position]] = sample.policy[position];
        }
        if (0..ACTION_SIZE as i32).contains(&sample.last_action) {
            let position = sample.last_action as usize;
            states[[batch_index, 2, position / BOARD_SIZE, position % BOARD_SIZE]] = 1.0;
        }
        values[[batch_index, 0]] = sample.value;
    }

    Ok((
        Tensor::from_array(states)?,
        Tensor::from_array(policies)?,
        Tensor::from_array(values)?,
    ))
}

fn train(
    artifact_dir: &Path,
    data_dir: &Path,
    output_model: &Path,
    checkpoint_output: &Path,
    batch_size: usize,
    epochs: usize,
    board_name: &str,
    policy_name: &str,
    value_name: &str,
) -> Result<(), Box<dyn Error>> {
    let _ = ort::init().commit();
    // ort registers `release_env_on_exit` in `.fini_array` so that `ReleaseEnv`
    // runs when the process exits. When the runtime is loaded dynamically
    // (ORT_DYLIB_PATH / load-dynamic) the shared library is dlopen'd after the
    // executable, so `_dl_fini` tears down the library's C++ statics *before*
    // our `.fini_array` entry runs, and calling `ReleaseEnv` on the torn-down
    // library segfaults at process exit. Leaking one `Arc` keeps the reference
    // count above zero so the environment is never released here; the OS
    // reclaims it when the process exits anyway.
    mem::forget(ort::environment::Environment::current()?);
    let trainer = Trainer::new_from_artifacts(
        Session::builder()?,
        Allocator::default(),
        artifact_dir,
        None,
    )?;
    let samples = load_data(data_dir)?;
    if samples.is_empty() {
        return Err(format!("no training samples found in {}", data_dir.display()).into());
    }

    let batch_size = batch_size.max(1).min(samples.len());
    for epoch in 0..epochs {
        let mut batches = 0;
        for batch in samples.chunks(batch_size) {
            let (board, policy, value) = make_batch(batch)?;
            // ORT's `TrainStep` expects exactly one OrtValue per user input
            // (here `board`, `target_p`, `target_v`). ort 2.0.0-rc.13's
            // `Trainer::step` concatenates the expanded entries of `inputs`
            // and `labels` when both are maps, which would produce 2N entries,
            // so pass every user input in the labels map and an empty inputs
            // slice: the ValueSlice+ValueMap branch yields exactly N entries
            // matched by name.
            let inputs: &[ort::session::SessionInputValue<'_>] = &[];
            let labels = ort::inputs! {
                board_name => board,
                policy_name => policy,
                value_name => value,
            };
            let outputs = trainer.step(inputs, labels)?;
            trainer.optimizer().reset_grad()?;
            trainer.optimizer().step()?;
            batches += 1;
            if let Some(loss) = outputs.values().next() {
                let loss_value = loss
                    .try_extract_tensor::<f32>()
                    .ok()
                    .and_then(|(_, data)| data.first().copied())
                    .unwrap_or(f32::NAN);
                println!("epoch={} batch={} loss={loss_value}", epoch + 1, batches);
            }
        }
        println!("epoch={} batches={}", epoch + 1, batches);
    }

    trainer.checkpoint().save(checkpoint_output, true)?;
    trainer.export(output_model, [POLICY_OUTPUT_NAME, VALUE_OUTPUT_NAME])?;
    Ok(())
}

fn usage(program: &str) {
    eprintln!(
        "usage: {program} <artifact-dir> <data-dir> <output-model> <checkpoint> [batch-size] [epochs] [board-name] [policy-name] [value-name]"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 5 {
        usage(&args[0]);
        return Err("missing training arguments".into());
    }

    let batch_size = args
        .get(5)
        .map_or(Ok(128usize), |value| value.parse::<usize>())?;
    let epochs = args
        .get(6)
        .map_or(Ok(1usize), |value| value.parse::<usize>())?;
    train(
        &PathBuf::from(&args[1]),
        &PathBuf::from(&args[2]),
        &PathBuf::from(&args[3]),
        &PathBuf::from(&args[4]),
        batch_size,
        epochs,
        args.get(7).map_or(BOARD_INPUT_NAME, String::as_str),
        args.get(8).map_or(POLICY_TARGET_NAME, String::as_str),
        args.get(9).map_or(VALUE_TARGET_NAME, String::as_str),
    )
}

#[cfg(test)]
mod tests {
    use super::{player_channels, symmetries, Sample, ACTION_SIZE, BOARD_SIZE};

    #[test]
    fn channel_zero_holds_own_stones_for_both_players() {
        // Black to move: own = Black (+1), opponent = White (-1).
        assert_eq!(player_channels(1, 1), (1.0, 0.0));
        assert_eq!(player_channels(1, -1), (0.0, 1.0));
        assert_eq!(player_channels(1, 0), (0.0, 0.0));

        // White to move: own = White (-1), opponent = Black (+1).
        assert_eq!(player_channels(-1, -1), (1.0, 0.0));
        assert_eq!(player_channels(-1, 1), (0.0, 1.0));
        assert_eq!(player_channels(-1, 0), (0.0, 0.0));
    }

    #[test]
    fn generates_python_compatible_eight_symmetries() {
        let mut board = [0; ACTION_SIZE];
        let mut policy = [0.0; ACTION_SIZE];
        board[0] = 1;
        policy[0] = 1.0;
        let variants = symmetries(Sample {
            board,
            policy,
            value: 1.0,
            current_player: 1,
            last_action: 0,
        });

        assert_eq!(variants.len(), 8);
        let bottom_left = (BOARD_SIZE - 1) * BOARD_SIZE;
        assert_eq!(variants[0].board[bottom_left], 1);
        assert_eq!(variants[0].last_action, bottom_left as i32);
        assert_eq!(variants[1].board[ACTION_SIZE - 1], 1);
        assert_eq!(variants[1].last_action, (ACTION_SIZE - 1) as i32);
        assert_eq!(variants[6].board[0], 1);
        assert_eq!(variants[6].last_action, 0);
    }
}
