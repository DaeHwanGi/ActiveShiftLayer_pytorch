from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='activeshift2d',
    ext_modules=[
        CppExtension('activeshift2d', ['activeshift2d.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
