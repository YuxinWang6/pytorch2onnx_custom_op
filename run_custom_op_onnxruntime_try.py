import onnxruntime as rt
import numpy as np
import os
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

# shared_library = '.pytorch_custom_op/try_ncrelu_forward.cpython-37m-x86_64-linux-gnu.so'
shared_library = './pytorch_custom_op/build/lib.linux-x86_64-3.7/try_ncrelu_forward.cpython-37m-x86_64-linux-gnu.so'
if not os.path.exists(shared_library):
    raise FileNotFoundError("Unable to find '{0}'".format(shared_library))
# this = os.path.dirname(__file__)
# custom_op_model = os.path.join(this, "testdata", "custom_op_library", "custom_op_test.onnx")


so1 = rt.SessionOptions()
so1.register_custom_ops_library(shared_library)

