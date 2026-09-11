import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from common import config


def conv3x3(in_channels, out_channels, stride=1):
    """Create the 3x3 convolution used by every residual block."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResidualBlock(nn.Module):
    """A two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out


class NeuralNetWork(nn.Module):
    """Policy and value network used by self-play and model training."""

    def __init__(self, num_layers, num_channels, n, action_size, input_channel_size):
        super().__init__()

        residual_blocks = [ResidualBlock(input_channel_size, num_channels)]
        residual_blocks.extend(
            ResidualBlock(num_channels, num_channels)
            for _ in range(num_layers - 1)
        )
        self.res_layers = nn.Sequential(*residual_blocks)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 4, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU(inplace=True)

        self.p_fc = nn.Linear(4 * n ** 2, action_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # value head
        self.v_conv = nn.Conv2d(num_channels, 2, kernel_size=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(num_features=2)

        # KataGo-style global-feature injection into the value head: append the
        # constant absolute side-to-move color (ch3, +1 Black / -1 White) to the
        # value-head input so the value head can represent color-asymmetric rules
        # (e.g. Renju). The scale is zero-initialized, so symmetric rules are
        # untouched at init and the data decides whether absolute color matters.
        self.value_color_inject = input_channel_size >= 4
        if self.value_color_inject:
            self.color_scale = nn.Parameter(torch.zeros(1))
            self.v_fc1 = nn.Linear(2 * n ** 2 + 1, 256)
        else:
            self.v_fc1 = nn.Linear(2 * n ** 2, 256)
        self.v_fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        out = self.res_layers(inputs)

        # policy head
        p = self.p_conv(out)
        p = self.p_bn(p)
        p = self.relu(p)

        p = self.p_fc(p.view(p.size(0), -1))
        p = self.log_softmax(p)

        # value head
        v = self.v_conv(out)
        v = self.v_bn(v)
        v = self.relu(v)

        v = v.view(v.size(0), -1)
        if self.value_color_inject:
            # ch3 is constant across the board, so read a single cell and scale it.
            color = inputs[:, 3, 0, 0].unsqueeze(1)  # (B, 1)
            v = torch.cat((v, self.color_scale * color), dim=1)
        v = self.v_fc1(v)
        v = self.relu(v)
        v = self.v_fc2(v)
        v = self.tanh(v)

        return p, v


class AlphaLoss(nn.Module):
    """AlphaZero loss: value MSE plus cross-entropy against the visit policy."""

    def __init__(self):
        super().__init__()

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss, policy_loss = self.split(log_ps, vs, target_ps, target_vs)

        return value_loss + policy_loss

    @staticmethod
    def split(log_ps, vs, target_ps, target_vs):
        """Value MSE and policy cross-entropy as separate terms.

        The policy term is exactly CE(target || model), so logging it on its own
        keeps the two heads attributable while training.
        """
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, dim=1))

        return value_loss, policy_loss


class NeuralNetWorkWrapper:
    """Own the model, optimizer, device selection, and data conversion."""

    def __init__(self, lr, l2, num_layers, num_channels, n, action_size, input_channel_size=4):
        """ init
        """
        self.lr = lr
        self.l2 = l2
        self.num_channels = num_channels
        self.n = n
        self.input_channel_size = input_channel_size

        if config['train_use_gpu']:
            self.is_cuda_available = torch.cuda.is_available()
        else:
            self.is_cuda_available = False

        if self.is_cuda_available:
            print("Find and use GPU")
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("use CPU")

        self.neural_network = NeuralNetWork(
            num_layers, num_channels, n, action_size, input_channel_size
        )
        if self.is_cuda_available:
            self.neural_network.cuda()

        self.optim = Adam(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.alpha_loss = AlphaLoss()

    def train(self, example_buffer, batch_size, epochs):
        """Run mini-batch updates, sampling examples with replacement."""
        n_data = len(example_buffer)
        for epo in range(1, epochs + 1):
            self.neural_network.train()

            # sample with replacement, O(batch_size); as in the AlphaZero paper
            sample_idx = np.random.randint(0, n_data, size=batch_size)
            train_data = [example_buffer[i] for i in sample_idx]


            board_batch, last_action_batch, cur_player_batch, p_batch, v_batch = list(zip(*train_data))

            state_batch = self._data_convert(board_batch, last_action_batch, cur_player_batch)
            p_batch = np.array(p_batch)
            v_batch = np.array(v_batch)
            p_batch = torch.Tensor(p_batch).cuda() if self.is_cuda_available else torch.Tensor(p_batch)
            v_batch = torch.Tensor(v_batch).unsqueeze(
                1).cuda() if self.is_cuda_available else torch.Tensor(v_batch).unsqueeze(1)

            self.optim.zero_grad()

            log_ps, vs = self.neural_network(state_batch)
            value_loss, policy_loss = self.alpha_loss.split(log_ps, vs, p_batch, v_batch)
            loss = value_loss + policy_loss

            log_this_epoch = epo % 20 == 0 or epo == epochs
            # read the metric tensors before backward: the autograd engine releases
            # the saved buffers of non-leaf tensors while back-propagating
            loss_value = float(loss.detach())
            value_loss_value = float(value_loss.detach())
            policy_loss_value = float(policy_loss.detach())
            head_metrics = (
                self._batch_metrics(log_ps, p_batch) if log_this_epoch else (0.0, 0.0, 0.0)
            )

            loss.backward()
            self.optim.step()

            if log_this_epoch:
                entropy, target_entropy, agreement = head_metrics
                print("EPOCH: {}/{}, LOSS: {}, LOSS_V: {}, LOSS_P: {}, ENTROPY: {}, "
                      "TGT_ENTROPY: {}, ACC: {}".format(
                          epo, epochs, loss_value, value_loss_value, policy_loss_value,
                          entropy, target_entropy, agreement))

    @staticmethod
    def _batch_metrics(log_ps, target_ps):
        """Policy-head diagnostics for the current batch.

        ENTROPY is the model's own H(p) and TGT_ENTROPY is H(pi_target): the gap
        between them shows whether the search produced a sharper target than the
        prediction, and ACC is the share of positions whose argmax move agrees.
        """
        with torch.no_grad():
            probs = torch.exp(log_ps)
            entropy = -float(torch.mean(torch.sum(probs * log_ps, dim=1)))
            target_entropy = -float(
                torch.mean(
                    torch.sum(
                        torch.where(
                            target_ps > 0,
                            target_ps * torch.log(target_ps.clamp_min(1e-30)),
                            torch.zeros_like(target_ps),
                        ),
                        dim=1,
                    )
                )
            )
            agreement = float(
                torch.mean(
                    (torch.argmax(probs, dim=1) == torch.argmax(target_ps, dim=1)).float()
                )
            )

        return entropy, target_entropy, agreement

    def infer(self, feature_batch):
        """Predict policy probabilities and values for raw feature tuples."""
        board_batch, last_action_batch, cur_player_batch = list(zip(*feature_batch))
        states = self._data_convert(board_batch, last_action_batch, cur_player_batch)

        self.neural_network.eval()
        with torch.no_grad():
            log_ps, vs = self.neural_network(states)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _infer(self, state_batch):
        """Predict policy probabilities and values for an encoded state tensor."""
        self.neural_network.eval()
        with torch.no_grad():
            log_ps, vs = self.neural_network(state_batch)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _data_convert(self, board_batch, last_action_batch, cur_player_batch):
        """Convert board features to [batch, input_channel_size, board, board] tensors."""
        n = self.n

        board_batch = torch.as_tensor(
            np.asarray(board_batch), dtype=torch.float32
        ).unsqueeze(1)
        state0 = (board_batch > 0).float()
        state1 = (board_batch < 0).float()

        # when the current player is white (-1), swap the two channels to unify the perspective (vectorized)
        cur_player = np.asarray(cur_player_batch, dtype=np.int64)
        swap = torch.from_numpy(cur_player == -1)
        if swap.any():
            tmp = state0[swap].clone()
            state0[swap] = state1[swap]
            state1[swap] = tmp

        # last_action marker (vectorized)
        state2 = torch.zeros((len(board_batch), 1, n, n)).float()
        last_action = np.asarray(last_action_batch, dtype=np.int64)
        valid = np.nonzero(last_action >= 0)[0]
        if valid.size > 0:
            pos = last_action[valid]
            rows = torch.from_numpy(valid)
            xs = torch.from_numpy(pos // n)
            ys = torch.from_numpy(pos % n)
            state2[rows, 0, xs, ys] = 1

        # channel 3: constant color plane carrying the absolute side-to-move color
        # (+1 Black / -1 White). The only color-asymmetric input; required to
        # represent color-asymmetric rules (e.g. Renju) and a harmless constant
        # for symmetric rules. Must match ortopt.rs and ort_train.rs.
        if self.input_channel_size >= 4:
            color = np.where(cur_player == 1, 1.0, -1.0).astype(np.float32)
            state3 = torch.from_numpy(color).view(-1, 1, 1, 1).expand(-1, 1, n, n)
            res = torch.cat((state0, state1, state2, state3), dim=1)
        else:
            res = torch.cat((state0, state1, state2), dim=1)
        return res.cuda() if self.is_cuda_available else res

    def set_learning_rate(self, lr):
        """Update the learning rate for all optimizer parameter groups."""
        for param_group in self.optim.param_groups:
            param_group["lr"] = lr

    def load_model(self, filepath):
        """Load network and optimizer state from a path prefix."""
        if self.is_cuda_available:
            state = torch.load(filepath+'.pkl', weights_only=True)
        else:
            state = torch.load(filepath+'.pkl', map_location='cpu', weights_only=True)
        self.neural_network.load_state_dict(state['network'])
        self.optim.load_state_dict(state['optim'])
        if self.is_cuda_available:
            self.neural_network.cuda()


    def save_model(self, filepath):
        """Save network state and an inference ONNX model to a path prefix."""
        # remove old files with the same name before saving to avoid stale models
        for suffix in ('.pkl', '.onnx'):
            old_path = filepath + suffix
            if os.path.exists(old_path):
                os.remove(old_path)

        state = {'network':self.neural_network.state_dict(), 'optim':self.optim.state_dict()}
        torch.save(state, filepath+'.pkl')


        self.neural_network.eval()
        self.neural_network.cpu()
        example = torch.rand(1, self.input_channel_size, self.n, self.n).cpu()
        dynamic_axes = {
            "board": {0: "batch_size"},
            "P": {0: "batch_size"},
            "V": {0: "batch_size"},
        }
        torch.onnx.export(
            self.neural_network,
            example,
            filepath + ".onnx",
            input_names=["board"],
            output_names=["P", "V"],
            dynamic_axes=dynamic_axes,
        )

        # restore the training device so that later reuse in the same process doesn't silently fall back to CPU
        if self.is_cuda_available:
            self.neural_network.cuda()


if __name__ == '__main__':
    net=NeuralNetWorkWrapper(lr=0.1, l2=0.1, num_layers=3, num_channels=32, n=15, action_size=15*15)
    # print("save model")
    # net.save_model("/data/AlphaZero-Onnx/python/mymodel")

    print("load model")
    net.load_model("/data/AlphaZero-Onnx/python/mymodel")
    batch_all = 5
    state_batch = np.zeros((batch_all+1,4,15,15))

    state_batch[batch_all][1][0][0] = 1 # gomoku.execute_move(0);
    state_batch[batch_all][0][0][1] = 1 #   gomoku.execute_move(1);
    state_batch[batch_all][1][3][4] = 1 #   gomoku.execute_move(3*15+4=49);

    state_batch[batch_all][2][3][4] = 1 # last move
    state_batch[batch_all][3][0][0] = 1 # color plane: black to move


    if net.is_cuda_available:
        state_batch = torch.Tensor(state_batch).cuda()
    else:
        state_batch = torch.Tensor(state_batch)
    print("predict")
    P,V = net._infer(state_batch)
    print(f"P[{batch_all}:5]={P[batch_all][0:5]},V={V[batch_all][0]}")
