from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='pawner',
    ext_modules=[
        CUDAExtension(
            'pawner',
            ['chess.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

