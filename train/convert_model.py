#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型结构转换:将现有网络调整为新的层数/通道数,并尽可能保留棋力。

流程:
1. 加载源模型 A(.pkl)
2. 按目标层数/通道数创建网络 B
3. 结构映射初始化 B:
   - 通道扩宽:旧通道权重逐位复制,新增行/列/BN γ 用 1e-3 尺度小随机初始化,
     前向输出与 A 几乎完全一致(扰动 ~1e-3),同时反向传播所有新参数都有梯度
   - 通道缩窄:截取前 C_B 个通道(有误差,靠蒸馏恢复)
   - 层数:复制前 min(K_A, K_B) 个残差块;新增块近似恒等初始化(conv ~ 0 扰动)
   - 策略/价值头 FC 输入维度与主干通道数无关,原样复制
4. 蒸馏:在回放数据上最小化 CE(soft π_A || π_B) + MSE(v_A, v_B)
5. 保存 B 的 .pkl 与 .onnx(复用 NeuralNetWorkWrapper.save_model)

用法示例(4x256 -> 4x128,蒸馏 1500 步):
    python3 train/convert_model.py --src build/weights/1204 --dst build/weights/1205 \
        --layers 4 --channels 128 --steps 1500

    python3 train/convert_model.py --self-test   # 仅跑结构映射自检,不读盘
"""

import argparse
import os
import sys
from os import path

import numpy as np
import torch

from common import config
from neural_network import NeuralNetWork, NeuralNetWorkWrapper

# 与 learner.py 一致的训练工作目录推导
REPO_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
BUILD_DIR = os.environ.get('BUILD_DIR') or path.join(REPO_ROOT, 'build')

# 新增参数的扰动尺度(相对旧权重 std),前向近似恒等、反向保证梯度流动
PERTURB = 1e-3


# ---------------------------------------------------------------- 结构映射

def copy_conv(dst, src, perturb=PERTURB):
    """卷积权重复制:重叠区域逐位复制;新增输出行/输入列用小随机扰动"""
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
    """BN 复制:重叠通道复制全部统计量;新增通道 γ 取小值(输出 ~0 但有梯度)"""
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
    """全连接层原样复制(要求输入输出维度一致)"""
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)


def init_identity(block, perturb=PERTURB):
    """新增残差块近似恒等初始化:conv 小扰动,BN 默认参数;
       输出 = relu(x + 微小量) ≈ x(x 为前一 relu 输出,非负)"""
    with torch.no_grad():
        block.conv1.weight.normal_(0.0, perturb)
        block.conv2.weight.normal_(0.0, perturb)
        block.bn1.reset_parameters()
        block.bn2.reset_parameters()
        if block.downsample:
            block.downsample_conv.weight.normal_(0.0, perturb)
            block.downsample_bn.reset_parameters()


def map_model(src_net, dst_net, perturb=PERTURB):
    """把源网络 A 的结构映射到目标网络 B 作为初始权重
       要求:双方 n、action_size、input_channel 一致
    """
    src_blocks = src_net.res_layers
    dst_blocks = dst_net.res_layers
    shared = min(len(src_blocks), len(dst_blocks))

    # 共享深度:逐残差块映射
    for i in range(shared):
        s, d = src_blocks[i], dst_blocks[i]
        copy_conv(d.conv1, s.conv1, perturb)
        copy_bn(d.bn1, s.bn1)
        copy_conv(d.conv2, s.conv2, perturb)
        copy_bn(d.bn2, s.bn2)
        if d.downsample and s.downsample:
            copy_conv(d.downsample_conv, s.downsample_conv, perturb)
            copy_bn(d.downsample_bn, s.downsample_bn)

    # 新增深度:近似恒等残差块
    for i in range(shared, len(dst_blocks)):
        init_identity(dst_blocks[i], perturb)

    # 头:1x1 卷积按通道裁剪/扩宽,FC 原样复制(FC 输入与主干通道无关)
    copy_conv(dst_net.p_conv, src_net.p_conv, perturb)
    copy_bn(dst_net.p_bn, src_net.p_bn)
    copy_conv(dst_net.v_conv, src_net.v_conv, perturb)
    copy_bn(dst_net.v_bn, src_net.v_bn)
    copy_fc(dst_net.p_fc, src_net.p_fc)
    copy_fc(dst_net.v_fc1, src_net.v_fc1)
    copy_fc(dst_net.v_fc2, src_net.v_fc2)

    return (f"mapped {shared}/{len(dst_blocks)} blocks from source, "
            f"{len(dst_blocks) - shared} blocks identity-init")


# ---------------------------------------------------------------- 数据读取

def read_raw_features(file_path):
    """从二进制数据文件读取 (board, color, last_action),跳过 prob 区
       返回 None 表示文件不完整/损坏
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
    """收集目录列表中的 (mtime, batch_id, path),规则与 learner.select_replay_files 一致"""
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
    """按 (mtime, id) 双键取最新 window_files 个文件,读出全部原始特征"""
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


# ---------------------------------------------------------------- 蒸馏

def state_of(wrapper, boards, colors, lasts, idx):
    """按索引构造特征并转成网络输入 tensor"""
    feats = [(boards[i], lasts[i], colors[i]) for i in idx]
    return wrapper._data_convert(*zip(*feats))


def evaluate_pair(a_net, b_net, wrapper_a, boards, colors, lasts, batch=1024):
    """统计 B 相对 A 的平均 KL(p_A||p_B)、|Δv| 与 policy top-1 一致率"""
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
    """让 B 拟合 A 的 soft π 与 v(带放回采样,AlphaZero 风格)"""
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


# ---------------------------------------------------------------- 自检

def self_test():
    """结构映射正确性自检:扩宽+加层后前向输出与源网络几乎一致;
       缩窄后网络可正常前向"""
    torch.manual_seed(0)
    n, action = 15, 225

    src = NeuralNetWork(3, 64, n, action, 3)
    src.eval()

    # 场景 1:扩宽 + 加层 → 输出应与源几乎一致
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

    # 场景 2:缩窄 → 前向可运行且有限
    dst2 = NeuralNetWork(3, 32, n, action, 3)
    print(map_model(src, dst2))
    dst2.eval()
    with torch.no_grad():
        _, vb2 = dst2(x)
    assert torch.isfinite(vb2).all()
    print(f"narrow: output finite, v[0]={float(vb2[0][0]):.4f}")

    # 场景 3:蒸馏 3 步,损失可下降、参数有梯度
    wrapper_src = NeuralNetWorkWrapper(config['lr'], config['l2'], 3, 64, n, action)
    wrapper_dst = NeuralNetWorkWrapper(config['lr'], config['l2'], 5, 96, n, action)
    map_model(wrapper_src.neural_network, wrapper_dst.neural_network)
    boards = np.random.randint(-1, 2, size=(64, n, n))
    colors = np.random.choice([1, -1], size=64)
    lasts = np.random.randint(0, action, size=64)
    wrapper_src.neural_network.eval()
    distill(wrapper_src, wrapper_dst, boards, colors, lasts, steps=3, batch_size=16)
    print("self-test PASS")


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description="调整模型层数/通道数并蒸馏保留棋力")
    parser.add_argument('--src', help='源模型路径前缀(如 build/weights/1204,需存在 .pkl)')
    parser.add_argument('--dst', help='目标模型保存路径前缀(如 build/weights/1205)')
    parser.add_argument('--layers', type=int, default=config['num_layers'],
                        help=f"目标残差层数(默认 {config['num_layers']})")
    parser.add_argument('--channels', type=int, default=config['num_channels'],
                        help=f"目标通道数(默认 {config['num_channels']})")
    parser.add_argument('--steps', type=int, default=1500, help='蒸馏 mini-batch 步数(0=跳过)')
    parser.add_argument('--batch', type=int, default=config['batch_size'], help='蒸馏 batch size')
    parser.add_argument('--lr', type=float, default=config['lr'], help='蒸馏学习率')
    parser.add_argument('--replay-files', type=int, default=320,
                        help='蒸馏数据文件数(按 mtime,id 取最新,默认 320=20 轮)')
    parser.add_argument('--include-archive', action='store_true',
                        help='把 data_archive 历史文件也纳入候选')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--self-test', action='store_true', help='仅跑结构映射自检,不读盘')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.src or not args.dst:
        parser.error('--src 与 --dst 必填(或用 --self-test 自检)')
    if not path.exists(args.src + '.pkl'):
        sys.exit(f"source model not found: {args.src}.pkl")

    n, action = config['n'], config['action_size']

    # 源网络 A(结构与 config 一致)
    wrapper_a = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                     config['num_channels'], n, action,
                                     config['input_channel_size'])
    wrapper_a.load_model(args.src)

    # 目标网络 B
    wrapper_b = NeuralNetWorkWrapper(args.lr, config['l2'], args.layers, args.channels,
                                     n, action, config['input_channel_size'])
    wrapper_b.set_learning_rate(args.lr)

    print(map_model(wrapper_a.neural_network, wrapper_b.neural_network))
    print(f"target: {args.layers} layers x {args.channels} channels")

    # 蒸馏数据(回放窗口 + 可选归档)
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
    print("下一步建议:运行 train_and_eval eval_with_winner 让候选与当前 best 对战验收,")
    print("胜率达标后再手动更新 current_and_best_weight.txt")


if __name__ == '__main__':
    main()
