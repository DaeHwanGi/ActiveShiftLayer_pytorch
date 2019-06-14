from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='DepthwiseAffineGrid',
    ext_modules=[
        CppExtension('DepthwiseAffineGrid', ['DepthwiseAffineGrid.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
