"""Build and verify a native OpenVINO semantic classifier."""

import numpy as np
import openvino as ov
from openvino import opset13 as ops


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

rng = np.random.default_rng(42)

W1 = rng.normal(0, 0.1, (6, 12)).astype(np.float32)
b1 = np.zeros(12, dtype=np.float32)

W2 = rng.normal(0, 0.1, (12, 3)).astype(np.float32)
b2 = np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------
# OpenVINO graph
# ---------------------------------------------------------

input_node = ops.parameter(
    [ov.Dimension.dynamic(), 6],
    np.float32,
    name="float_input",
)

hidden = ops.matmul(
    input_node,
    ops.constant(W1),
    False,
    False,
)

hidden = ops.add(
    hidden,
    ops.constant(b1),
)

hidden = ops.relu(hidden)

logits = ops.matmul(
    hidden,
    ops.constant(W2),
    False,
    False,
)

logits = ops.add(
    logits,
    ops.constant(b2),
)

output = ops.softmax(
    logits,
    axis=1,
)

output.set_friendly_name("class_probabilities")

model = ov.Model(
    [output],
    [input_node],
    "SemanticClassifier",
)


# ---------------------------------------------------------
# Compile on CPU
# ---------------------------------------------------------

core = ov.Core()

compiled = core.compile_model(
    model,
    "CPU",
)


# ---------------------------------------------------------
# Test inference
# ---------------------------------------------------------

test_input = np.array(
    [
        [8, 45, 2, 0, 0, 1],
        [80, 500, 12, 0, 2, 0],
        [20, 100, 4, 3, 1, 0],
    ],
    dtype=np.float32,
)

result = compiled([test_input])

probabilities = result[compiled.output(0)]

print("OpenVINO model compiled successfully.")
print("Device: CPU")
print("Input shape:", compiled.input(0).partial_shape)
print("Output shape:", compiled.output(0).partial_shape)
print("Test output shape:", probabilities.shape)
print("Test output:")
print(probabilities)
print("Model build and inference verification completed successfully.")