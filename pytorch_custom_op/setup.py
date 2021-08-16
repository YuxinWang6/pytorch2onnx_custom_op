from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='try_ncrelu_forward',
      ext_modules=[cpp_extension.CppExtension('try_ncrelu_forward', ['try_custom_op.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
