from collections import deque
from os import path
import os
import random
import sys

import numpy as np
import torch

from common import config
from neural_network import NeuralNetWorkWrapper

# training work dir: build/ under the repo root by default, overridable via BUILD_DIR;
# derived from the script's own location, independent of the runtime cwd
REPO_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
BUILD_DIR = os.environ.get('BUILD_DIR') or path.join(REPO_ROOT, 'build')


def parse_batch_id(file_name):
    """Parse batch_id from a data file name `data_{batch_id}_{hex}`
       returns None on failure
    """
    parts = path.basename(file_name).split('_')
    if len(parts) >= 3 and parts[0] == 'data':
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def select_replay_files(data_dir, backup_dir, window_files):
    """Collect data files from data/ and data_backup/, sort by (mtime, batch_id)
       descending, and take the newest window_files as the replay window
       returns (selected, obsolete) file path lists
    """
    candidates = []
    for folder in (data_dir, backup_dir):
        if not path.isdir(folder):
            continue
        for file_name in os.listdir(folder):
            file_path = path.join(folder, file_name)
            if not path.isfile(file_path):
                continue
            try:
                mtime = os.path.getmtime(file_path)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, parse_batch_id(file_name), file_path))
    # descending by mtime then batch_id; unparsable ids rank lowest
    candidates.sort(
        key=lambda item: (item[0], item[1] if item[1] is not None else -1),
        reverse=True,
    )
    selected = [item[2] for item in candidates[:window_files]]
    obsolete = [item[2] for item in candidates[window_files:]]
    return selected, obsolete


class Learner():
    def __init__(self, config):
        """Create the trainer from the shared training configuration."""
        self.n = config['n']
        self.n_in_row = config['n_in_row']
        self.action_size = config['action_size']

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']

        self.examples_buffer = deque([], maxlen=config['examples_buffer_max_len'])

        self.use_GPU = config['train_use_gpu']

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], config['n'],
                                         self.action_size, config['input_channel_size'])

    def learn(self, model_dir, model_id):
        """Train one model generation and archive its consumed self-play data."""
        model_path = path.join(model_dir, str(model_id))
        model_file = model_path + '.pkl'
        if not path.exists(model_file):
            raise FileNotFoundError(f"{model_file} does not exist")
        print(f"loading {model_id}-th model")
        self.nnet.load_model(model_path)

        # learning rate decays in steps by model generation
        lr = config['lr']
        for milestone in config['lr_milestones']:
            if model_id >= milestone:
                lr *= config['lr_gamma']
        self.nnet.set_learning_rate(lr)
        print(f"learning rate: {lr}")

        # replay window: train on the most recent N iterations of data from data/ and
        # data_backup/ combined (AlphaZero replay)
        data_path = path.join(BUILD_DIR, 'data')
        data_backup_path = path.join(path.dirname(data_path), 'data_backup')
        data_archive_path = path.join(path.dirname(data_path), 'data_archive')
        window_files = config['examples_buffer_max_len'] * config['games_per_iter']
        replay_files, _ = select_replay_files(data_path, data_backup_path, window_files)
        print(f"replay window: {config['examples_buffer_max_len']} iters x "
              f"{config['games_per_iter']} games = {window_files} files, "
              f"selected {len(replay_files)}")
        train_data = self.load_samples(replay_files)
        if not train_data:
            raise RuntimeError("no valid training samples found in the replay window")
        random.shuffle(train_data)

        # train neural network
        epochs = self.epochs * (len(train_data) + self.batch_size - 1) // self.batch_size
        self.nnet.train(train_data, min(self.batch_size, len(train_data)), epochs)

        model_path = path.join(model_dir, str(model_id+1))
        self.nnet.save_model(model_path)
        if self.use_GPU:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # post-training archiving: move the newly generated files from data/ to data_backup/
        # for later replay; then filter all of data_backup/ by the replay window, moving
        # out-of-window history to data_archive/ for retention
        os.makedirs(data_backup_path, exist_ok=True)
        os.makedirs(data_archive_path, exist_ok=True)
        for file_name in os.listdir(data_path):
            try:
                os.rename(path.join(data_path, file_name),
                          path.join(data_backup_path, file_name))
            except OSError:
                pass
        print(f"moved training data to: {data_backup_path}")
        _, obsolete_files = select_replay_files(data_path, data_backup_path, window_files)
        for file_path in obsolete_files:
            try:
                os.rename(file_path,
                          path.join(data_archive_path, path.basename(file_path)))
            except OSError as error:
                print(f"skip archiving {file_path}: {error}")
        print(f"archived {len(obsolete_files)} files beyond replay window to: {data_archive_path}")

    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert (len(pi) == self.action_size)  # 1 for pass

        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        if(last_action != -1):
            last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                newAction = np.rot90(last_action_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newAction = np.fliplr(newAction)
                l += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return l

    def load_samples(self, files):
        """load self.examples_buffer
           files: data file path list (selected by the replay window)
        """
        BOARD_SIZE = self.n
        N2 = BOARD_SIZE * BOARD_SIZE
        # bytes per sample: board (N2 i32) + prob (N2 f32) + v/color/last_action (3 i32)
        bytes_per_step = N2 * 4 + N2 * 4 + 3 * 4
        train_examples = []
        for file_path in files:
            if not path.isfile(file_path):
                continue
            try:
                file_size = path.getsize(file_path)
            except OSError as error:
                print(f"skip unreadable data file {file_path}: {error}")
                continue
            try:
                with open(file_path, 'rb') as binfile:
                    step = int().from_bytes(binfile.read(4), byteorder='little', signed=True)
                    expected_size = 4 + step * bytes_per_step
                    # the self-play process may still be writing, or a previous interrupted
                    # run left a partial file; skip size-mismatched files to avoid ValueError
                    # from reshape
                    if step <= 0 or file_size < expected_size:
                        print(f"skip incomplete data file {file_path}: "
                              f"step={step}, size={file_size}, expected={expected_size}")
                        continue
                    # bulk read to avoid element-wise Python-level IO
                    board = np.frombuffer(binfile.read(step * N2 * 4), dtype='<i4').reshape(step, BOARD_SIZE, BOARD_SIZE)
                    prob = np.frombuffer(binfile.read(step * N2 * 4), dtype='<f4').reshape(step, N2)
                    v = np.frombuffer(binfile.read(step * 4), dtype='<i4')
                    color = np.frombuffer(binfile.read(step * 4), dtype='<i4')
                    last_action = np.frombuffer(binfile.read(step * 4), dtype='<i4')

                    for i in range(step):
                        sym = self.get_symmetries(board[i], prob[i], last_action[i])
                        for b, p, a in sym:
                            train_examples.append([b, a, color[i], p, v[i]])
            except (ValueError, OSError) as error:
                print(f"skip corrupted data file {file_path}: {error}")
                continue
        print(f"loaded {len(train_examples)} samples from {len(files)} files")
        return train_examples


if __name__ == '__main__':
    model_dir = path.join(BUILD_DIR, "weights")
    le = Learner(config)
    if len(sys.argv) <= 1 or sys.argv[1] == "prepare":
        print("save 0-th model !!")
        le.nnet.save_model(path.join(model_dir,'0'))
        print("done !")
    else:
        assert sys.argv[1] == "train", sys.argv[1]
        weight_file = path.join(BUILD_DIR, "current_and_best_weight.txt")
        with open(weight_file, 'r') as f:
            current_id, best_id =  f.readline().split(" ")
            current_id = int(current_id)
        le.learn(model_dir=model_dir, model_id=current_id)
        with open(weight_file, 'w') as f:
            f.write(str(int(current_id)+1) + " "+ str(best_id))
        
