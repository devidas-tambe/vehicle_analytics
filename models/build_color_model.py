"""Generate a deterministic ONNX prototype color classifier; no trained weights.
Input: NCHW RGB [0,1], 32x32. Output: categorical scores, not calibrated probabilities.
The central crop reduces (but does not eliminate) background influence.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

LABELS = [
    "black", "white", "silver", "gray", "red",
    "blue", "green", "yellow", "orange", "brown",
]
PROTOTYPES = (
    np.array(
        [
            [25, 25, 25],
            [237, 237, 237],
            [178, 184, 191],
            [105, 108, 112],
            [180, 38, 42],
            [37, 73, 165],
            [40, 125, 65],
            [225, 197, 43],
            [224, 115, 35],
            [115, 71, 44],
        ],
        dtype=np.float32,
    )
    / 255
)


def build(out):
    # -||x-p||² = 2*x*p - ||p||² - ||x||²; last term cancels in softmax.
    temperature = 20.0
    W = (2 * temperature * PROTOTYPES.T).astype(np.float32)
    B = (-temperature * (PROTOTYPES**2).sum(axis=1)).astype(np.float32)
    inits = [
        numpy_helper.from_array(np.array(v, dtype=np.int64), n)
        for n, v in [("starts", [8, 8]), ("ends", [24, 24]), ("axes", [2, 3])]
    ]
    inits += [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")]
    nodes = [
        helper.make_node("Slice", ["image", "starts", "ends", "axes"], ["central"]),
        helper.make_node("ReduceMean", ["central"], ["rgb"], axes=[2, 3], keepdims=0),
        helper.make_node("Gemm", ["rgb", "W", "B"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "vehicle_color_prototype",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 32, 32])],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, ["N", len(LABELS)]
            )
        ],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.checker.check_model(model)
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out / "vehicle_color.onnx")
    (out / "colors.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    print("Generated", out / "vehicle_color.onnx")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)