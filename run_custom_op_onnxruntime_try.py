import onnxruntime as rt
import numpy as np
import os
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)

@onnx_op(op_type="try_ncrelu_forward")
def neg_vec(x):
    return -x

# shared_library = '.pytorch_custom_op/try_ncrelu_forward.cpython-37m-x86_64-linux-gnu.so'
shared_library = './pytorch_custom_op/build/lib.linux-x86_64-3.7/try_ncrelu_forward.cpython-37m-x86_64-linux-gnu.so'
if not os.path.exists(shared_library):
    raise FileNotFoundError("Unable to find '{0}'".format(shared_library))


so1 = rt.SessionOptions()
so1.register_custom_ops_library(shared_library)
# so1.register_custom_ops_library(_get_library_path())

x_data = np.random.rand(3, 2, 1, 8).astype(np.float32)
sess = rt.InferenceSession("./pytorch_custom_op/model_custom_op.onnx", so1)

