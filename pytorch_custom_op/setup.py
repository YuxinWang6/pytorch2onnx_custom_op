from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='neg_tensor',
      ext_modules=[cpp_extension.CppExtension('neg_tensor', ['implement_custom_op.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
