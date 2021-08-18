import torch
# import neg_tensor
torch.ops.load_library("./build/lib.linux-x86_64-3.7/neg_tensor.cpython-37m-x86_64-linux-gnu.so")

from torch.onnx import register_custom_op_symbolic

def my_custom_op(g, input):
    return g.op("ai.onnx.contrib::neg_tensor", input)

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x):
            return torch.ops.usercustom.neg_tensor(x)
            # return neg_tensor.usercustom.neg_tensor(x)

    X = torch.randn(3, 2, 1, 8)

    inputs = (X)

    f = 'model_custom_op.onnx'
    torch.onnx.export(CustomModel(), inputs, f, opset_version=11, input_names=["X"], output_names=["Y"])


register_custom_op_symbolic('usercustom::neg_tensor', my_custom_op, 11)
export_custom_op()
