from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cpawner',
    ext_modules=[
        CUDAExtension(
            'cpawner',
            ['csrc/extension.cu'],
            #extra_compile_args=['-O0', '-lineinfo', '-g'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

