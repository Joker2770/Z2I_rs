#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate ONNX Runtime Training artifacts for `ort_train` (src/ort_train.rs).

`ort_train` expects, inside a single artifact directory (see
`Trainer::new_from_artifacts` in ort_train.rs):

    training_model.onnx   -- training graph: input `board`, labels `target_p`
                             / `target_v`, output is the (scalar) loss
    eval_model.onnx       -- eval graph: input `board`, outputs `P` / `V`
    optimizer_model.onnx  -- optimizer (AdamW) graph
    checkpoint/           -- initial checkpoint directory

The Rust binary feeds `board` (shape [batch, 3, 15, 15]), `target_p`
([batch, 225]) and `target_v` ([batch, 1]) and later exports the trained
model with output names ["P", "V"]. All of these names/shapes are matched
here.

The AlphaZero loss used by `train/neural_network.py::AlphaLoss` is:

    value_loss  = mean((V - target_v) ** 2)
    policy_loss = -mean(sum(target_p * P, dim=1))     # P is log_softmax
    loss        = value_loss + policy_loss

Because ONNX Runtime Training only ships MSE / CrossEntropy / BCE / L1 as
built-in losses, and CrossEntropy expects logits (it applies softmax again),
we provide a custom `onnxblock.Block` that reproduces the exact combined
loss above (P is already `log_softmax`, so no extra softmax is applied).

This script has two independent stages so each one can run in a different
Python environment (only the stage actually used is imported):

    # Stage 1 (needs torch) -- export the base ONNX model:
    python3 generate_ort_artifacts.py export --base-out base_model.onnx

    # Stage 2 (needs onnxruntime-training, torch NOT required) -- generate
    # the full artifact set:
    python3 generate_ort_artifacts.py artifacts \
        --base-model base_model.onnx \
        --out ort_training_artifacts
"""

import argparse
import os
import sys
from pathlib import Path

N = 15
ACTION_SIZE = N * N
INPUT_CHANNEL_SIZE = 3


# --------------------------------------------------------------------------
# Stage 1: base model export (requires torch)
# --------------------------------------------------------------------------

def export_base_model(
    out_path: Path,
    num_channels: int = 32,
    num_layers: int = 4,
    opset: int = 17,
) -> None:
    # Make neural_network.py importable from this file's directory.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import torch  # noqa: F401  (imported lazily so stage 2 needs no torch)

    from neural_network import NeuralNetWork

    model = NeuralNetWork(
        num_layers=num_layers,
        num_channels=num_channels,
        n=N,
        action_size=ACTION_SIZE,
        input_channel_size=INPUT_CHANNEL_SIZE,
    )
    model.eval()

    dummy = torch.randn(1, INPUT_CHANNEL_SIZE, N, N)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["board"],
        output_names=["P", "V"],
        opset_version=opset,
        dynamic_axes={"board": {0: "batch"}, "P": {0: "batch"}, "V": {0: "batch"}},
        do_constant_folding=True,
    )

    # torch.onnx may emit the weights as an external ".data" file; inline them
    # so the base model is a single self-contained file.
    import onnx

    model_onnx = onnx.load(str(out_path), load_external_data=True)
    onnx.save(model_onnx, str(out_path))
    leftover = Path(str(out_path) + ".data")
    if leftover.exists():
        leftover.unlink()
    print(f"[export] wrote {out_path}")


# --------------------------------------------------------------------------
# Stage 2: ORT training artifacts (requires onnxruntime-training)
# --------------------------------------------------------------------------

def _requires_grad_names(onnx_model) -> list:
    """Return the model initializers that are actual trainable weights.

    Only the weight/bias inputs of parameterized operators (Conv, Gemm,
    BatchNormalization scale/bias) require gradient. Everything else that
    ends up as an initializer -- BatchNorm running_mean/running_var and
    constants emitted by the exporter (e.g. reshape shape vectors) -- must
    stay out of the trainable set, otherwise ORT would try to learn them.
    """
    trainable = set()
    for node in onnx_model.graph.node:
        if node.op_type in ("Conv", "BatchNormalization"):
            # Conv: [X, W, B?] ; BatchNormalization: [X, scale, B, mean, var].
            # Only indices 1 and 2 (weights) are trainable; BatchNorm's
            # running_mean/running_var live at indices 3 and 4 and are skipped.
            for index in (1, 2):
                if len(node.input) > index and node.input[index]:
                    trainable.add(node.input[index])
        elif node.op_type == "Gemm":
            # Gemm: [A, B, C?] -- B is weight, C is bias
            if len(node.input) > 1 and node.input[1]:
                trainable.add(node.input[1])
            if len(node.input) > 2 and node.input[2]:
                trainable.add(node.input[2])

    # Preserve the original order of initializers for deterministic output.
    initializer_names = [initializer.name for initializer in onnx_model.graph.initializer]
    return [name for name in initializer_names if name in trainable]


def _make_alpha_zero_loss_block():
    """Return a custom onnxblock.Block implementing the AlphaLoss above."""
    import onnx

    from onnxruntime.training.onnxblock import Block
    from onnxruntime.training.onnxblock import blocks

    class AlphaZeroLossBlock(Block):
        def __init__(self):
            super().__init__()
            self._counter = 0

        def _unique(self, base: str) -> str:
            self._counter += 1
            return f"alpha_zero_loss_{base}_{self._counter}"

        def _add_node(self, op_type, inputs, outputs, **attrs):
            name = self._unique(op_type)
            node = onnx.helper.make_node(op_type, inputs, outputs, name=name, **attrs)
            self.base.graph.node.append(node)
            return outputs[0]

        def _binary(self, op_type, a, b):
            out = self._unique(op_type.lower() + "_output")
            return self._add_node(op_type, [a, b], [out])

        def _unary(self, op_type, x, **attrs):
            out = self._unique(op_type.lower() + "_output")
            return self._add_node(op_type, [x], [out], **attrs)

        def _pow(self, x, exponent):
            exp_name = self._unique("pow_exponent")
            self.base.graph.initializer.append(
                onnx.helper.make_tensor(
                    exp_name, onnx.TensorProto.FLOAT, [1], [exponent]
                )
            )
            out = self._unique("pow_output")
            return self._add_node("Pow", [x, exp_name], [out])

        def _reduce_mean(self, x):
            return self._unary("ReduceMean", x, keepdims=0)

        def _reduce_sum_axis1(self, x):
            # Sum over the policy class dimension (dim=1) -> [batch].
            # ReduceSum (opset 13+) takes `axes` as an optional INPUT tensor,
            # not as an attribute.
            axes_name = self._unique("reduce_sum_axes")
            self.base.graph.initializer.append(
                onnx.helper.make_tensor(axes_name, onnx.TensorProto.INT64, [1], [1])
            )
            out = self._unique("reduce_sum_output")
            self._add_node("ReduceSum", [x, axes_name], [out], keepdims=0)
            return out

        def build(self, p_output_name: str, v_output_name: str):
            # Value loss: mean((V - target_v)^2)
            target_v = blocks.InputLike(v_output_name)("target_v")
            sub_v = self._binary("Sub", v_output_name, target_v)
            pow_v = self._pow(sub_v, 2.0)
            value_loss = self._reduce_mean(pow_v)

            # Policy loss: -mean(sum(target_p * P, dim=1))
            target_p = blocks.InputLike(p_output_name)("target_p")
            mul_p = self._binary("Mul", p_output_name, target_p)
            sum_p = self._reduce_sum_axis1(mul_p)
            mean_p = self._reduce_mean(sum_p)
            policy_loss = self._unary("Neg", mean_p)

            total = self._binary("Add", value_loss, policy_loss)
            return total

    return AlphaZeroLossBlock()


def generate_training_artifacts(
    base_model_path: Path,
    out_dir: Path,
    prefix: str = "",
) -> None:
    import onnx
    from onnxruntime.training import artifacts

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(base_model_path))
    requires_grad = _requires_grad_names(model)
    if not requires_grad:
        raise RuntimeError("No trainable parameters found in the base model.")

    loss_block = _make_alpha_zero_loss_block()

    artifacts.generate_artifacts(
        model=model,
        requires_grad=requires_grad,
        frozen_params=None,
        loss=loss_block,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=str(out_dir),
        prefix=prefix,
        ort_format=False,
        custom_op_library=None,
        additional_output_names=["P", "V"],
        nominal_checkpoint=False,
        loss_input_names=None,
    )

    print(f"[artifacts] generated into {out_dir}:")
    for name in sorted(
        [p.name for p in out_dir.iterdir()], key=lambda n: (n != "checkpoint", n)
    ):
        print(f"  - {name}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ONNX Runtime Training artifacts for ort_train."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export the base ONNX model (needs torch).")
    export_parser.add_argument(
        "--base-out",
        type=Path,
        default=Path(__file__).resolve().parent / "base_model.onnx",
        help="Output path for the base ONNX model.",
    )
    export_parser.add_argument("--num-channels", type=int, default=32)
    export_parser.add_argument("--num-layers", type=int, default=4)
    export_parser.add_argument("--opset", type=int, default=17)

    artifacts_parser = sub.add_parser(
        "artifacts", help="Generate training artifacts (needs onnxruntime-training)."
    )
    artifacts_parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(__file__).resolve().parent / "base_model.onnx",
        help="Path to the base ONNX model produced by the 'export' stage.",
    )
    artifacts_parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "ort_training_artifacts",
        help="Directory to write training_model.onnx / eval_model.onnx / optimizer_model.onnx / checkpoint.",
    )
    artifacts_parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()

    if args.command == "export":
        export_base_model(
            args.base_out,
            num_channels=args.num_channels,
            num_layers=args.num_layers,
            opset=args.opset,
        )
    elif args.command == "artifacts":
        generate_training_artifacts(args.base_model, args.out, prefix=args.prefix)


if __name__ == "__main__":
    main()
