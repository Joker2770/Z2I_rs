// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use crate::{
    configuration::cfg,
    gomoku::Gomoku,
    ortcommon,
    rule::{Board, Color},
};

use ort::{
    session::{self, Session, SessionOutputs},
    value::TensorRef,
};

use ndarray::Array;
use std::{
    error,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Duration,
};

pub(crate) type InferenceOutput = Result<(Vec<f64>, f64), String>;

struct InferenceTask {
    state: Vec<f32>,
    response: Sender<InferenceOutput>,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    request_sender: Sender<InferenceTask>,
    batch_size: Arc<AtomicUsize>,
}

impl NeuralNetwork {
    pub fn new(model_path: &Path, bs: usize) -> Result<Self, Box<dyn error::Error>> {
        // Register EPs based on feature flags - this isn't crucial for usage and can be removed.
        ortcommon::init()?;

        let session = Session::builder()?
            .with_optimization_level(session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(cfg::INTRA_THREAD_NUM as usize)?
            .commit_from_file(model_path)?;
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|item| item.name().to_string())
            .collect();

        let batch_size = if bs <= cfg::MAX_BATCH_SIZE as usize && bs >= cfg::MIN_BATCH_SIZE as usize
        {
            bs
        } else {
            cfg::DEFAULT_BATCH_SIZE as usize
        };

        let output_names = vec![
            cfg::OUTPUT_0_NAME.to_string(),
            cfg::OUTPUT_1_NAME.to_string(),
        ];
        let batch_size = Arc::new(AtomicUsize::new(batch_size));
        let worker_batch_size = Arc::clone(&batch_size);
        let (request_sender, request_receiver) = mpsc::channel();
        thread::Builder::new()
            .name("onnx-inference".to_string())
            .spawn(move || {
                inference_loop(
                    session,
                    input_names,
                    output_names,
                    worker_batch_size,
                    request_receiver,
                )
            })?;

        let nn = NeuralNetwork {
            request_sender,
            batch_size,
        };

        Ok(nn)
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size.load(Ordering::Relaxed)
    }

    pub fn transform_board_2_tensor(
        &self,
        board: &Board,
        last_move: i16,
        cur_color: &Color,
    ) -> Vec<f32> {
        let mut input_tensor_values =
            vec![
                0.0;
                cfg::CHANNEL_SIZE as usize * cfg::BOARD_SIZE as usize * cfg::BOARD_SIZE as usize
            ];
        let mut first = 0;
        let mut second = 0;
        if *cur_color == Color::Black {
            second = 1;
        } else {
            first = 1;
        }
        for r in 0..cfg::BOARD_SIZE {
            for c in 0..cfg::BOARD_SIZE {
                match board[r as usize][c as usize] {
                    Color::Black => {
                        input_tensor_values[first as usize
                            * cfg::BOARD_SIZE as usize
                            * cfg::BOARD_SIZE as usize
                            + r as usize * cfg::BOARD_SIZE as usize
                            + c as usize] = 1.0;
                    }
                    Color::White => {
                        input_tensor_values[second as usize
                            * cfg::BOARD_SIZE as usize
                            * cfg::BOARD_SIZE as usize
                            + r as usize * cfg::BOARD_SIZE as usize
                            + c as usize] = 1.0
                    }
                    _ => {}
                }
            }
            if last_move >= 0 {
                input_tensor_values[2 * cfg::BOARD_SIZE as usize * cfg::BOARD_SIZE as usize
                    + last_move as usize] = 1.0;
            }
        }
        input_tensor_values
    }

    pub fn transform_gomoku_2_tensor(&self, gomoku: &Gomoku) -> Vec<f32> {
        self.transform_board_2_tensor(
            gomoku.get_board(),
            gomoku.get_last_move(),
            gomoku.get_cur_color(),
        )
    }

    pub fn commit(&self, gomoku: &Gomoku) -> Result<Receiver<InferenceOutput>, String> {
        let state = self.transform_gomoku_2_tensor(gomoku);
        let (response_sender, response_receiver) = mpsc::channel();
        self.request_sender
            .send(InferenceTask {
                state,
                response: response_sender,
            })
            .map_err(|error| error.to_string())?;
        Ok(response_receiver)
    }

    pub fn set_batch_size(&mut self, bs: usize) -> bool {
        if bs <= cfg::MAX_BATCH_SIZE as usize && bs >= cfg::MIN_BATCH_SIZE as usize {
            self.batch_size.store(bs, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

fn inference_loop(
    mut session: Session,
    input_node_names: Vec<String>,
    output_names: Vec<String>,
    batch_size: Arc<AtomicUsize>,
    request_receiver: Receiver<InferenceTask>,
) {
    loop {
        let first = match request_receiver.recv() {
            Ok(task) => task,
            Err(_) => return,
        };
        let mut tasks = vec![first];

        let max_batch_size = batch_size.load(Ordering::Relaxed);
        while tasks.len() < max_batch_size {
            match request_receiver
                .recv_timeout(Duration::from_micros(cfg::INFER_TASK_WAIT_US as u64))
            {
                Ok(task) => tasks.push(task),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        let states = tasks
            .iter()
            .flat_map(|task| task.state.iter().copied())
            .collect();
        let result = infer_batch(
            &mut session,
            &input_node_names,
            &output_names,
            states,
            tasks.len(),
        );
        match result {
            Ok(outputs) => {
                for (task, output) in tasks.into_iter().zip(outputs) {
                    let _ = task.response.send(Ok(output));
                }
            }
            Err(error) => {
                for task in tasks {
                    let _ = task.response.send(Err(error.clone()));
                }
            }
        }
    }
}

fn infer_batch(
    session: &mut Session,
    input_node_names: &[String],
    output_names: &[String],
    state_all: Vec<f32>,
    batch_size: usize,
) -> Result<Vec<(Vec<f64>, f64)>, String> {
    let input_arr = Array::from_shape_vec(
        (
            batch_size,
            cfg::CHANNEL_SIZE as usize,
            cfg::BOARD_SIZE as usize,
            cfg::BOARD_SIZE as usize,
        ),
        state_all,
    )
    .map_err(|error| error.to_string())?;
    let inputs = ort::inputs! {
        &input_node_names[0] => TensorRef::from_array_view(&input_arr)
            .map_err(|error| error.to_string())?
    };

    let outputs: SessionOutputs = session.run(inputs).map_err(|error| error.to_string())?;
    let v_arr = outputs[output_names[0].as_str()]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?
        .into_owned();
    let p_arr = outputs[output_names[1].as_str()]
        .try_extract_array::<f32>()
        .map_err(|error| error.to_string())?
        .into_owned();
    let v_vec = v_arr.iter().collect::<Vec<&f32>>();
    let p_vec = p_arr.iter().collect::<Vec<&f32>>();
    let action_size = cfg::BOARD_SIZE as usize * cfg::BOARD_SIZE as usize;

    if v_vec.len() < batch_size || p_vec.len() < batch_size * action_size {
        return Err(format!(
            "invalid model output size: V={}, P={}, expected at least V={}, P={}",
            v_vec.len(),
            p_vec.len(),
            batch_size,
            batch_size * action_size
        ));
    }

    let mut results = Vec::with_capacity(batch_size);
    for (index, value) in v_vec.iter().enumerate().take(batch_size) {
        let probabilities = p_vec
            .iter()
            .skip(index * action_size)
            .take(action_size)
            .map(|value| (**value as f64).exp())
            .collect();
        results.push((probabilities, **value as f64));
    }
    Ok(results)
}
