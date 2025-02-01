from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='torch-chess',
    version='0.0.1',
    author='Bertolotti Francesco',
    author_email='f14.bertolotti@gmail.com',
    description='A CUDA chess engine extension for PyTorch',
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[
        CUDAExtension(
            'torch-chess',
            ['csrc/extension.cu', 'pysrc/pawner.py'],
            extra_compile_args=['-O3'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

