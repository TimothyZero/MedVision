# -*- coding:utf-8 -*-

import os
import glob

import setuptools
from setuptools import setup
import numpy
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, CUDA_HOME


def print_compile_env():
    import subprocess

    print('torch :', torch.__version__)
    print('torch.cuda :', torch.version.cuda)
    print("CUDA_HOME :", CUDA_HOME)
    try:
        with open(os.devnull, 'w') as devnull:
            gcc = subprocess.check_output(['gcc', '--version'],
                                          stderr=devnull).decode().rstrip('\r\n').split('\n')[0]
        print('gcc :', gcc)
    except Exception as e:
        pass


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'medvision', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))
    source_cuda += glob.glob(os.path.join(extensions_dir, 'cuda_aug', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    extra_compile_args = {}
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args = {
            'cxx': ['-O0'],
            'nvcc': nvcc_flags,
        }

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir, numpy.get_include()]

    ext_modules = [
        extension(
            'medvision._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    print_compile_env()

    with open("README.md", "r", encoding='utf-8') as fh:
        long_description = fh.read()

    setup(
        name='medvision',
        version='0.0.3',
        author="timothy, qiaoyf",
        author_email="thyzyfx@qq.com, qyyyyyf@gmail.com",
        description='Medical Image Vision',
        keywords='Medical Image, Vision',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/TimothyZero/MedVision',
        license='Apache License 2.0',
        packages=setuptools.find_packages('.'),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )
