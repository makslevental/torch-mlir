from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='traceWithShapes',
      ext_modules=[cpp_extension.CppExtension('traceWithShapes_cpp', ['traceWithShapes.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
