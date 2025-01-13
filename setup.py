from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='pawner',
    ext_modules=[
        CUDAExtension(
            'pawner',
            ['pawner.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

