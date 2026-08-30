#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model structure conversion: resize an existing network to new layer/channel counts
while preserving its playing strength as much as possible.

Flow:
1. Load the source model A (.pkl)
2. Create network B with the target layer/channel counts
3. Initialize B via structural mapping:
   - widening channels: old channel weights copied element-wise; new rows/columns/BN gamma
     initialized with tiny 1e-3-scale random values, so the forward output is almost identical
     to A (perturbation ~1e-3) while all new parameters still receive gradients in backward
   - narrowing channels: keep the first C_B channels (some error, recovered by distillation)
   - layer count: copy the first min(K_A, K_B) residual blocks; added blocks are near-identity
     initialized (conv ~ 0 perturbation)
   - the policy/value head FC input dims don't depend on backbone channels, copied as-is
4. Distill: minimize CE(soft pi_A || pi_B) + MSE(v_A, v_B) on replay data
5. Save B's .pkl and .onnx (reusing NeuralNetWorkWrapper.save_model)

Usage example (4x256 -> 4x128, 1500 distill steps):
    python3 train/convert_model.py --src build/weights/1204 --dst build/weights/1205 \
        --layers 4 --channels 128 --steps 1500

    python3 train/convert_model.py --self-test   # run only the structural mapping self-test, no disk access
"""

import argparse
import os
import sys
from os import path

import numpy as np
import torch

from common import config
from neural_network import NeuralNetWork, NeuralNetWorkWrapper

# same training work dir derivation as learner.py
REPO_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
BUILD_DIR = os.environ.get('BUILD_DIR') or path.join(REPO_ROOT, 'build')

# perturbation scale of added parameters (relative to old weight std): near-identity
# forward, gradient flow guaranteed in backward
PERTURB = 1e-3


# ---------------------------------------------------------------- structural mapping

def copy_conv(dst, src, perturb=PERTURB):
    """Copy conv weights: overlap region copied element-wise; new output rows/input cols
       get tiny random perturbation"""
    with torch.no_grad():
        sw = src.weight
        dw = dst.weight
        dw.zero_()
        o = min(dw.shape[0], sw.shape[0])
        i = min(dw.shape[1], sw.shape[1])
        dw[:o, :i].copy_(sw[:o, :i])
        new_std = perturb * sw.std().item()
        if dw.shape[0] > o:
            dw[o:].normal_(0.0, new_std)
        if dw.shape[1] > i:
            dw[:, i:].normal_(0.0, new_std)
        if dst.bias is not None:
            dst.bias.zero_()
            if src.bias is not None:
                c = min(len(dst.bias), len(src.bias))
                dst.bias[:c].copy_(src.bias[:c])


def copy_bn(dst, src, new_gamma=PERTURB):
    """Copy BN: overlap channels copy all statistics; new channels get a small gamma
       (output ~0 but with gradient)"""
    with torch.no_grad():
        c = min(dst.weight.shape[0], src.weight.shape[0])
        dst.weight.zero_()
        dst.bias.zero_()
        dst.weight[:c].copy_(src.weight[:c])
        dst.bias[:c].copy_(src.bias[:c])
        dst.running_mean.zero_()
        dst.running_var.fill_(1.0)
        dst.running_mean[:c].copy_(src.running_mean[:c])
        dst.running_var[:c].copy_(src.running_var[:c])
        dst.num_batches_tracked.zero_()
        if c < dst.weight.shape[0]:
            dst.weight[c:].fill_(new_gamma)


def copy_fc(dst, src):
    """Copy an FC layer as-is (requires matching input/output dims)"""
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)


def init_identity(block, perturb=PERTURB):
    """Near-identity init for an added residual block: small conv perturbation, default BN params;
       output = relu(x + tiny) ≈ x (x is the previous relu output, non-negative)"""
    with torch.no_grad():
        block.conv1.weight.normal_(0.0, perturb)
        block.conv2.weight.normal_(0.0, perturb)
        block.bn1.reset_parameters()
        block.bn2.reset_parameters()
        if block.downsample:
            block.downsample_conv.weight.normal_(0.0, perturb)
            block.downsample_bn.reset_parameters()


def map_model(src_net, dst_net, perturb=PERTURB):
    """Map the source network A's structure onto target network B as initial weights
       requires: same n, action_size and input_channel on both sides
    """
    src_blocks = src_net.res_layers
    dst_blocks = dst_net.res_layers
    shared = min(len(src_blocks), len(dst_blocks))

    # shared depth: map residual blocks one by one
    for i in range(shared):
        s, d = src_blocks[i], dst_blocks[i]
        copy_conv(d.conv1, s.conv1, perturb)
        copy_bn(d.bn1, s.bn1)
        copy_conv(d.conv2, s.conv2, perturb)
        copy_bn(d.bn2, s.bn2)
        if d.downsample and s.downsample:
            copy_conv(d.downsample_conv, s.downsample_conv, perturb)
            copy_bn(d.downsample_bn, s.downsample_bn)

    # added depth: near-identity residual blocks
    for i in range(shared, len(dst_blocks)):
        init_identity(dst_blocks[i], perturb)

    # heads: 1x1 convs narrowed/widened by channel, FCs copied as-is
    # (FC inputs don't depend on backbone channels)
    copy_conv(dst_net.p_conv, src_net.p_conv, perturb)
    copy_bn(dst_net.p_bn, src_net.p_bn)
    copy_conv(dst_net.v_conv, src_net.v_conv, perturb)
    copy_bn(dst_net.v_bn, src_net.v_bn)
    copy_fc(dst_net.p_fc, src_net.p_fc)
    copy_fc(dst_net.v_fc1, src_net.v_fc1)
    copy_fc(dst_net.v_fc2, src_net.v_fc2)

    return (f"mapped {shared}/{len(dst_blocks)} blocks from source, "
            f"{len(dst_blocks) - shared} blocks identity-init")


# ---------------------------------------------------------------- data reading

def read_raw_features(file_path):
    """Read (board, color, last_action) from a binary data file, skipping the prob section
       returns None if the file is incomplete/corrupt
    """
    n = config['n']
    N2 = n * n
    bytes_per_step = N2 * 4 + N2 * 4 + 3 * 4
    try:
        file_size = path.getsize(file_path)
        with open(file_path, 'rb') as binfile:
            step = int.from_bytes(binfile.read(4), byteorder='little', signed=True)
            if step <= 0 or file_size < 4 + step * bytes_per_step:
                print(f"skip incomplete data file {file_path}: "
                      f"step={step}, size={file_size}")
                return None
            board = np.frombuffer(binfile.read(step * N2 * 4), dtype='<i4').reshape(step, n, n)
            binfile.seek(step * N2 * 4, 1)  # skip prob
            _ = binfile.read(step * 4)     # skip v
            color = np.frombuffer(binfile.read(step * 4), dtype='<i4')
            last_action = np.frombuffer(binfile.read(step * 4), dtype='<i4')
        return board.astype(np.int8), color, last_action
    except (ValueError, OSError) as error:
        print(f"skip corrupted data file {file_path}: {error}")
        return None


def collect_candidates(dirs):
    """Collect (mtime, batch_id, path) from a list of dirs, same rules as
       learner.select_replay_files"""
    from learner import parse_batch_id
    candidates = []
    for folder in dirs:
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
    candidates.sort(
        key=lambda item: (item[0], item[1] if item[1] is not None else -1),
        reverse=True,
    )
    return [item[2] for item in candidates]


def load_feature_data(dirs, window_files):
    """Take the newest window_files files by the (mtime, id) double key and read all raw features"""
    files = collect_candidates(dirs)[:window_files]
    boards, colors, lasts = [], [], []
    for file_path in files:
        raw = read_raw_features(file_path)
        if raw is None:
            continue
        board, color, last = raw
        boards.append(board)
        colors.append(color)
        lasts.append(last)
    if not boards:
        raise RuntimeError(f"no valid data files in {dirs}")
    return (np.concatenate(boards), np.concatenate(colors), np.concatenate(lasts)), len(files)


# ---------------------------------------------------------------- distillation

def state_of(wrapper, boards, colors, lasts, idx):
    """Build features by index and convert them to the network input tensor"""
    feats = [(boards[i], lasts[i], colors[i]) for i in idx]
    return wrapper._data_convert(*zip(*feats))


def evaluate_pair(a_net, b_net, wrapper_a, boards, colors, lasts, batch=1024):
    """Report B's mean KL(p_A||p_B), |dv| and policy top-1 agreement relative to A"""
    n_data = len(boards)
    total_kl, total_dv, total_top1, cnt = 0.0, 0.0, 0, 0
    a_net.eval()
    b_net.eval()
    for start in range(0, n_data, batch):
        idx = np.arange(start, min(start + batch, n_data))
        state = state_of(wrapper_a, boards, colors, lasts, idx)
        with torch.no_grad():
            log_pa, va = a_net(state)
            log_pb, vb = b_net(state)
            pa = torch.exp(log_pa)
            total_kl += float(torch.sum(pa * (log_pa - log_pb)))
            total_dv += float(torch.sum(torch.abs(va - vb)))
            total_top1 += int((torch.argmax(log_pa, 1) == torch.argmax(log_pb, 1)).sum())
        cnt += len(idx)
    return total_kl / cnt, total_dv / cnt, total_top1 / cnt


def distill(wrapper_a, wrapper_b, boards, colors, lasts,
            steps, batch_size, print_every=50):
    """Fit B to A's soft pi and v (sampling with replacement, AlphaZero style)"""
    a_net = wrapper_a.neural_network
    b_net = wrapper_b.neural_network
    a_net.eval()
    n_data = len(boards)
    rng = np.random.default_rng(0)

    for step in range(1, steps + 1):
        b_net.train()
        idx = rng.integers(0, n_data, size=batch_size)
        state = state_of(wrapper_a, boards, colors, lasts, idx)

        with torch.no_grad():
            log_pa, va = a_net(state)
            pa = torch.exp(log_pa)
        log_pb, vb = b_net(state)

        policy_loss = -torch.mean(torch.sum(pa * log_pb, dim=1))
        value_loss = torch.mean(torch.pow(vb - va, 2))
        loss = policy_loss + value_loss

        wrapper_b.optim.zero_grad()
        loss.backward()
        wrapper_b.optim.step()

        if step % print_every == 0 or step == steps:
            with torch.no_grad():
                top1 = float((torch.argmax(log_pa, 1) == torch.argmax(log_pb, 1))
                             .float().mean())
            print(f"distill {step}/{steps}: loss={loss.item():.4f} "
                  f"(p={policy_loss.item():.4f}, v={value_loss.item():.4f}), "
                  f"top1={top1:.4f}")


# ---------------------------------------------------------------- self-test

def self_test():
    """Self-test of structural mapping: after widening+deepening the forward output matches
       the source almost exactly; after narrowing the network still forwards normally"""
    torch.manual_seed(0)
    n, action = 15, 225

    src = NeuralNetWork(3, 64, n, action, 3)
    src.eval()

    # scenario 1: widen + deepen -> output should almost match the source
    dst = NeuralNetWork(5, 96, n, action, 3)
    print(map_model(src, dst))
    dst.eval()
    x = torch.rand(4, 3, n, n)
    with torch.no_grad():
        log_pa, va = src(x)
        log_pb, vb = dst(x)
        pa = torch.exp(log_pa)
        kl = float(torch.mean(torch.sum(pa * (log_pa - log_pb), dim=1)))
        dv = float(torch.mean(torch.abs(va - vb)))
    print(f"widen+deepen: KL={kl:.6f}, |dv|={dv:.6f}")
    assert kl < 0.05, f"KL too large: {kl}"
    assert dv < 0.1, f"|dv| too large: {dv}"

    # scenario 2: narrow -> forward runs and stays finite
    dst2 = NeuralNetWork(3, 32, n, action, 3)
    print(map_model(src, dst2))
    dst2.eval()
    with torch.no_grad():
        _, vb2 = dst2(x)
    assert torch.isfinite(vb2).all()
    print(f"narrow: output finite, v[0]={float(vb2[0][0]):.4f}")

    # scenario 3: 3 distill steps, loss descends and params get gradients
    wrapper_src = NeuralNetWorkWrapper(config['lr'], config['l2'], 3, 64, n, action)
    wrapper_dst = NeuralNetWorkWrapper(config['lr'], config['l2'], 5, 96, n, action)
    map_model(wrapper_src.neural_network, wrapper_dst.neural_network)
    boards = np.random.randint(-1, 2, size=(64, n, n))
    colors = np.random.choice([1, -1], size=64)
    lasts = np.random.randint(0, action, size=64)
    wrapper_src.neural_network.eval()
    distill(wrapper_src, wrapper_dst, boards, colors, lasts, steps=3, batch_size=16)
    print("self-test PASS")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Resize model layers/channels and distill to preserve strength")
    parser.add_argument('--src', help='source model path prefix (e.g. build/weights/1204; .pkl must exist)')
    parser.add_argument('--dst', help='target model save path prefix (e.g. build/weights/1205)')
    parser.add_argument('--layers', type=int, default=config['num_layers'],
                        help=f"target residual layer count (default {config['num_layers']})")
    parser.add_argument('--channels', type=int, default=config['num_channels'],
                        help=f"target channel count (default {config['num_channels']})")
    parser.add_argument('--steps', type=int, default=1500, help='distill mini-batch steps (0 = skip)')
    parser.add_argument('--batch', type=int, default=config['batch_size'], help='distill batch size')
    parser.add_argument('--lr', type=float, default=config['lr'], help='distill learning rate')
    parser.add_argument('--replay-files', type=int, default=320,
                        help='distill data file count (newest by mtime,id; default 320 = 20 iterations)')
    parser.add_argument('--include-archive', action='store_true',
                        help='include data_archive history files as candidates')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--self-test', action='store_true', help='run only the structural mapping self-test, no disk access')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.src or not args.dst:
        parser.error('--src and --dst are required (or use --self-test)')
    if not path.exists(args.src + '.pkl'):
        sys.exit(f"source model not found: {args.src}.pkl")

    n, action = config['n'], config['action_size']

    # source network A (structure matches config)
    wrapper_a = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                     config['num_channels'], n, action,
                                     config['input_channel_size'])
    wrapper_a.load_model(args.src)

    # target network B
    wrapper_b = NeuralNetWorkWrapper(args.lr, config['l2'], args.layers, args.channels,
                                     n, action, config['input_channel_size'])
    wrapper_b.set_learning_rate(args.lr)

    print(map_model(wrapper_a.neural_network, wrapper_b.neural_network))
    print(f"target: {args.layers} layers x {args.channels} channels")

    # distill data (replay window + optional archive)
    data_path = path.join(BUILD_DIR, 'data')
    dirs = [data_path, path.join(path.dirname(data_path), 'data_backup')]
    if args.include_archive:
        dirs.append(path.join(path.dirname(data_path), 'data_archive'))
    (boards, colors, lasts), n_files = load_feature_data(dirs, args.replay_files)
    print(f"distill data: {len(boards)} states from {n_files} files")

    print("before distill:")
    kl, dv, top1 = evaluate_pair(wrapper_a.neural_network, wrapper_b.neural_network,
                                 wrapper_a, boards, colors, lasts)
    print(f"  KL={kl:.6f}, |dv|={dv:.6f}, top1={top1:.4f}")

    if args.steps > 0:
        distill(wrapper_a, wrapper_b, boards, colors, lasts,
                args.steps, args.batch, args.print_every)
        print("after distill:")
        kl, dv, top1 = evaluate_pair(wrapper_a.neural_network, wrapper_b.neural_network,
                                     wrapper_a, boards, colors, lasts)
        print(f"  KL={kl:.6f}, |dv|={dv:.6f}, top1={top1:.4f}")

    wrapper_b.save_model(args.dst)
    print(f"saved {args.dst}.pkl and {args.dst}.onnx")
    print("next step: run train_and_eval eval_with_winner to have the candidate face the current best,")
    print("and update current_and_best_weight.txt manually once the win rate passes the threshold")


if __name__ == '__main__':
    main()
