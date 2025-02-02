from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='torchess',
    version='0.0.4',
    author='Bertolotti Francesco',
    author_email='f14.bertolotti@gmail.com',
    description='A CUDA chess engine extension for PyTorch',
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="pysrc"),
    package_dir={'': 'pysrc'},
    ext_modules=[
        CUDAExtension(
            'cpawner',
            ['csrc/extension.cu'],
            extra_compile_args=['-O3'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
     install_requires=["numpy","torch"],
)

