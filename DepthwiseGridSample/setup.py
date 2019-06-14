from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='DepthwiseGridSampler',
    ext_modules=[
        CppExtension('DepthwiseGridSampler', ['DepthwiseGridSampler.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
