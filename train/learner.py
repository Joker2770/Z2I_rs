from os import path
import os
import random
import shutil
import struct
import sys
import tempfile

# The training stack is only needed to actually train: the replay-window selection below is
# plain file IO, and keeping the import optional lets `learner.py --self-test` check that
# selection on a machine without train/requirements.txt installed.
try:
    import numpy as np
    import torch
    from neural_network import NeuralNetWorkWrapper
except ImportError:  # pragma: no cover - only without the training dependencies
    np = None
    torch = None
    NeuralNetWorkWrapper = None

from common import config

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


def read_data_rule(file_path):
    """Rule flag stored in a data file header.

    Returns None when the file cannot be used as a training sample at all: too short to hold
    the header, or smaller than its own step count claims (a `.part` file a running self-play
    process has not renamed yet, or an interrupted round). Files written before the rule
    field existed carry no rule and default to FreeStyle (0), exactly as `load_samples`
    reads them.
    """
    n = config['n']
    N2 = n * n
    bytes_per_step = N2 * 4 + N2 * 4 + 3 * 4
    try:
        file_size = path.getsize(file_path)
        with open(file_path, 'rb') as binfile:
            header = binfile.read(4)
            if len(header) < 4:
                return None
            step = int.from_bytes(header, byteorder='little', signed=True)
            if step <= 0 or file_size < 4 + step * bytes_per_step:
                return None
            if file_size >= 4 + step * bytes_per_step + 4:
                return int.from_bytes(binfile.read(4), byteorder='little', signed=True)
            return 0
    except OSError:
        return None


def _collect_candidates(folder):
    """(mtime, batch_id, path) for every file in `folder` (unsorted)."""
    items = []
    if not path.isdir(folder):
        return items
    for file_name in os.listdir(folder):
        file_path = path.join(folder, file_name)
        if not path.isfile(file_path):
            continue
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0.0
        items.append((mtime, parse_batch_id(file_name), file_path))
    return items


def _newest_first(items):
    """Sort by mtime then batch_id, descending; unparsable ids rank lowest."""
    items.sort(
        key=lambda item: (item[0], item[1] if item[1] is not None else -1),
        reverse=True,
    )
    return items


def select_replay_files(data_dir, backup_dir, window_files, archive_dir=None, expected_rule=None):
    """Collect data files from data/ and data_backup/, sort by (mtime, batch_id)
       descending, and take the newest window_files as the replay window
       returns (selected, obsolete, stats)

       `expected_rule` filters on the rule flag in each file header **before** the window is
       filled. The order matters on a work dir copied from another rule (the `*_lead_by_*`
       hot-start flow): the parent's files are the newest ones there, so filling the window
       first spends every slot on files `load_samples` then discards, leaving training with
       only the handful of files the round just generated -- a starved window that never
       trips the short-window warning. Files from another rule are returned in `obsolete` so
       the caller archives them out of the way (they are retained in data_archive/, never
       deleted); unreadable or incomplete files are dropped from both lists, so a `.part`
       file a running self-play process is still writing is never moved.

       `archive_dir` is only consulted when data/ + data_backup/ cannot fill the window
       (a fresh or restored work dir): the self-play samples are independent of the model
       that generated them, so month-old archived games are still usable training data,
       and training on a starved window is what quietly flattens a policy.

       `stats` counts {"foreign": n, "unreadable": n} for the caller's log.
    """
    candidates = []
    foreign = []
    unreadable = 0
    for folder in (data_dir, backup_dir):
        for item in _collect_candidates(folder):
            if expected_rule is None:
                candidates.append(item)
                continue
            rule = read_data_rule(item[2])
            if rule is None:
                unreadable += 1
            elif rule != expected_rule:
                foreign.append(item)
            else:
                candidates.append(item)
    _newest_first(candidates)
    _newest_first(foreign)
    selected = [item[2] for item in candidates[:window_files]]

    if len(selected) < window_files and archive_dir and path.isdir(archive_dir):
        archived = _collect_candidates(archive_dir)
        if expected_rule is not None:
            archived = [item for item in archived
                        if read_data_rule(item[2]) == expected_rule]
        # the archived files are all older than the live window, so the fallback keeps the
        # live/new ordering at the head of the list
        selected.extend(item[2] for item in _newest_first(archived)[:window_files - len(selected)])

    obsolete = [item[2] for item in candidates[window_files:]]
    obsolete.extend(item[2] for item in foreign)
    return selected, obsolete, {"foreign": len(foreign), "unreadable": unreadable}


class Learner():
    def __init__(self, config):
        """Create the trainer from the shared training configuration."""
        if NeuralNetWorkWrapper is None:
            raise RuntimeError("the training dependencies are missing: "
                               "python3 -m pip install -r train/requirements.txt")
        self.n = config['n']
        self.action_size = config['action_size']
        # expected rule of the self-play samples (0 = FreeStyle); files carrying a different
        # rule in their header are filtered out of the replay window before it is filled (and
        # archived away afterwards), see select_replay_files
        self.rule = config['rule']

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
        # A starved replay window is the failure mode that quietly flattens a policy: with
        # few samples the value head still converges (and still passes a tactical probe)
        # while the policy never sharpens. Fall back to data_archive/ and say so loudly.
        include_archive = os.environ.get('REPLAY_INCLUDE_ARCHIVE', '') not in ('', '0')
        replay_files, _, replay_stats = select_replay_files(
            data_path, data_backup_path, window_files,
            archive_dir=data_archive_path if include_archive else None,
            expected_rule=self.rule,
        )
        archive_used = sum(1 for file_path in replay_files
                           if path.dirname(file_path) == data_archive_path)
        print(f"replay window: {config['examples_buffer_max_len']} iters x "
              f"{config['games_per_iter']} games = {window_files} files, "
              f"selected {len(replay_files)}"
              + (f" ({archive_used} filled from data_archive/)" if archive_used else ""))
        if replay_stats['foreign']:
            print(f"rule check: {replay_stats['foreign']} file(s) carry another rule and were "
                  f"left out of the window (expected {self.rule}); they are archived to "
                  f"{data_archive_path} after this round")
        if replay_stats['unreadable']:
            print(f"rule check: {replay_stats['unreadable']} incomplete file(s) ignored "
                  f"(left in place)")
        if len(replay_files) < window_files:
            hint = (f" {replay_stats['foreign']} file(s) in data/ + data_backup/ belong to "
                    f"another rule, and a work dir started from another rule needs its own "
                    f"{window_files} files before the window is full."
                    if replay_stats['foreign'] else "")
            print(f"WARNING: the replay window is short by {window_files - len(replay_files)} "
                  f"file(s) ({len(replay_files)}/{window_files}). Training will barely move "
                  f"the model, which shows up as a flat policy; set REPLAY_INCLUDE_ARCHIVE=1 "
                  f"to fill it from {data_archive_path}.{hint}")
        train_data = self.load_samples(replay_files)
        if not train_data:
            raise RuntimeError("no valid training samples found in the replay window")
        random.shuffle(train_data)

        # train neural network
        epochs = self.epochs * (len(train_data) + self.batch_size - 1) // self.batch_size
        print(f"training: {len(train_data)} samples, batch {min(self.batch_size, len(train_data))}"
              f", {epochs} steps ({epochs * min(self.batch_size, len(train_data))} sample draws)")
        self.nnet.train(train_data, min(self.batch_size, len(train_data)), epochs)

        model_path = path.join(model_dir, str(model_id+1))
        self.nnet.save_model(model_path)
        if config['train_use_gpu']:
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
        _, obsolete_files, _ = select_replay_files(data_path, data_backup_path, window_files,
                                                  expected_rule=self.rule)
        for file_path in obsolete_files:
            try:
                os.rename(file_path,
                          path.join(data_archive_path, path.basename(file_path)))
            except OSError as error:
                print(f"skip archiving {file_path}: {error}")
        print(f"archived {len(obsolete_files)} file(s) beyond the replay window (or from "
              f"another rule) to: {data_archive_path}")

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
        """Load samples from files selected by the replay window."""
        N2 = self.n * self.n
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
                    # header is backward compatible: files written by the current self-play
                    # carry a rule field (i32) right after step; legacy files have none and
                    # default to FreeStyle (0)
                    payload_bytes = step * bytes_per_step
                    old_expected = 4 + payload_bytes
                    new_expected = old_expected + 4
                    # the self-play process may still be writing, or a previous interrupted
                    # run left a partial file; skip size-mismatched files to avoid ValueError
                    # from reshape
                    if step <= 0 or file_size < old_expected:
                        print(f"skip incomplete data file {file_path}: "
                              f"step={step}, size={file_size}, expected={new_expected}")
                        continue
                    if file_size >= new_expected:
                        rule = int().from_bytes(binfile.read(4), byteorder='little', signed=True)
                    else:
                        rule = 0
                    if rule != self.rule:
                        print(f"skip data file {file_path}: rule={rule} != expected "
                              f"{self.rule} (mixed-rule training is unsupported)")
                        continue
                    # bulk read to avoid element-wise Python-level IO
                    board = np.frombuffer(
                        binfile.read(step * N2 * 4), dtype='<i4'
                    ).reshape(step, self.n, self.n)
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


def _write_synthetic_game(file_path, rule, plies, board_size):
    """Write one data file with zeroed features (self-test only)."""
    plane = board_size * board_size
    with open(file_path, 'wb') as binfile:
        binfile.write(struct.pack('<i', plies))
        binfile.write(struct.pack('<i', rule))
        binfile.write(struct.pack(f'<{plies * plane}i', *([0] * plies * plane)))
        binfile.write(struct.pack(f'<{plies * plane}f',
                                  *([1.0 / plane] * plies * plane)))
        binfile.write(struct.pack(f'<{plies}i', *([1] * plies)))     # v
        binfile.write(struct.pack(f'<{plies}i', *([1] * plies)))     # colour
        binfile.write(struct.pack(f'<{plies}i', *([-1] * plies)))    # last_move


def self_test():
    """Check the rule-aware replay-window selection without the training stack."""
    n = config['n']
    temp = tempfile.mkdtemp(prefix='learner_selftest_')
    try:
        folders = {name: path.join(temp, name)
                   for name in ('data', 'data_backup', 'data_archive')}
        for folder in folders.values():
            os.makedirs(folder)

        def write(folder, name, rule, mtime):
            file_path = path.join(folders[folder], name)
            _write_synthetic_game(file_path, rule, 2, n)
            os.utime(file_path, (mtime, mtime))
            return file_path

        # a work dir copied from another rule: the parent's files are the newest ones
        write('data', 'data_320_freestyle', 0, 4000)
        write('data_backup', 'data_304_freestyle', 0, 3999)
        write('data_backup', 'data_288_renju', 4, 3000)
        write('data_archive', 'data_256_renju', 4, 2999)
        # a half-written file: the shape an interrupted round (or a `.part` in flight) has
        partial = path.join(folders['data'], 'data_336_partial')
        with open(partial, 'wb') as binfile:
            binfile.write(struct.pack('<i', 4))
        os.utime(partial, (3500, 3500))

        # without the filter the window is filled by the parent rule's files: the trap
        unfiltered, _, _ = select_replay_files(folders['data'], folders['data_backup'], 2)
        assert [path.basename(p) for p in unfiltered] == [
            'data_320_freestyle', 'data_304_freestyle'], unfiltered

        selected, obsolete, stats = select_replay_files(
            folders['data'], folders['data_backup'], 2, expected_rule=4)
        assert [path.basename(p) for p in selected] == ['data_288_renju'], selected
        assert stats == {'foreign': 2, 'unreadable': 1}, stats
        # foreign files come back as obsolete (the caller archives them out of the way),
        # while the incomplete one is left where the writing process put it
        assert [path.basename(p) for p in obsolete] == [
            'data_320_freestyle', 'data_304_freestyle'], obsolete

        # the archive fallback fills a short window, and only with matching files
        filled, _, _ = select_replay_files(
            folders['data'], folders['data_backup'], 3,
            archive_dir=folders['data_archive'], expected_rule=4)
        assert [path.basename(p) for p in filled] == [
            'data_288_renju', 'data_256_renju'], filled

        # the rule is read from the header, not guessed from the file name
        assert read_data_rule(path.join(folders['data'], 'data_320_freestyle')) == 0
        assert read_data_rule(path.join(folders['data_backup'], 'data_288_renju')) == 4
        assert read_data_rule(partial) is None

        print('self-test: OK')
        return 0
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        sys.exit(self_test())
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
            current_id, best_id = f.readline().split()
            current_id = int(current_id)
        le.learn(model_dir=model_dir, model_id=current_id)
        with open(weight_file, 'w') as f:
            f.write(str(int(current_id)+1) + " "+ str(best_id))
        
