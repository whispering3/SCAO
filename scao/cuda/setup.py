"""
Optional CUDA extension build script for SCAO.
Run:  python setup.py build_ext --inplace

For production builds, prefer setting TORCH_CUDA_ARCH_LIST explicitly, e.g.:
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" python setup.py build_ext --inplace
"""

import os

from setuptools import setup

DEFAULT_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
]

if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    DEFAULT_NVCC_FLAGS.extend(
        [
            "-gencode=arch=compute_75,code=sm_75",   # T4
            "-gencode=arch=compute_80,code=sm_80",   # A100
            "-gencode=arch=compute_86,code=sm_86",   # A10/A10G/RTX 3090
            "-gencode=arch=compute_89,code=sm_89",   # RTX 4090/L4/L40S
            "-gencode=arch=compute_90,code=sm_90",   # H100
        ]
    )

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules = [
        CUDAExtension(
            name="scao.cuda._scao_cuda",
            sources=["low_rank_ops.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": DEFAULT_NVCC_FLAGS,
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    ext_modules = []
    cmdclass = {}

setup(
    name="scao_cuda",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
