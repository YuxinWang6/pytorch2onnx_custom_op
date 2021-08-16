import torch
# import try_ncrelu_forward
torch.ops.load_library("./build/lib.linux-x86_64-3.7/try_ncrelu_forward.cpython-37m-x86_64-linux-gnu.so")

from torch.onnx import register_custom_op_symbolic

def my_group_norm(g, input):
    return g.op("usercustom::try_ncrelu_forward", input)

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x):
            return torch.ops.usercustom.try_ncrelu_forward(x)

    X = torch.randn(3, 2, 1, 8)

    inputs = (X)

    f = 'modeleddy.onnx'
    torch.onnx.export(CustomModel(), inputs, f, opset_version=11, input_names=["X"], output_names=["Y"],
        custom_opsets={"usercustom": 11})


register_custom_op_symbolic('usercustom::try_ncrelu_forward', my_group_norm, 11)
export_custom_op()
