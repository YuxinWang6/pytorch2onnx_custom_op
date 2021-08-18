import onnxruntime as rt
import numpy as np
import os
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


@onnx_op(op_type="neg_tensor")
def neg_vec(x):
    return -x

so1 = rt.SessionOptions()

so1.register_custom_ops_library(_get_library_path())

x_data = np.random.rand(3, 2, 1, 8).astype(np.float32)
sess = rt.InferenceSession("./pytorch_custom_op/model_custom_op.onnx", so1)


input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred = sess.run([label_name], {input_name: x_data})[0]

print('input')
print(x_data)
print('output')
print(pred)