import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

common_setup_kwargs = {
    "version": "0.0.3",
    "name": "auto_quant",
    "author": "qwopqwop",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "An easy-to-use LLMs quantization package",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/qwopqwop200/AutoQuant",
    "keywords": ["autoquant", "quantization", "transformers"],
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ]
}

requirements = [
    "torch>=2.0.0",
    "transformers>=4.32.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "sentencepiece",
    "lm_eval",
]

include_dirs = []

conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
if os.path.isdir(conda_cuda_include_dir):
    include_dirs.append(conda_cuda_include_dir)

def check_dependencies():
    if CUDA_HOME is None:
        raise RuntimeError(
            f"Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    compute_capabilities = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError("GPUs with compute capability less than 8.0 are not supported.")
        compute_capabilities.add(major * 10 + minor)

    # figure out compute capability
    compute_capabilities = {80, 86, 89, 90}

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags

check_dependencies()
arch_flags = get_compute_capabilities()

if os.name == "nt":
    # Relaxed args on Windows
    extra_compile_args={
        "nvcc": arch_flags
    }
else:
    extra_compile_args={
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"] + arch_flags
    }

extensions = [
    CUDAExtension(
        "exllama_kernels",
        [
            "auto_quant/kernels/exllama/exllama_ext.cpp",
            "auto_quant/kernels/exllama/cuda_buffers.cu",
            "auto_quant/kernels/exllama/cuda_func/column_remap.cu",
            "auto_quant/kernels/exllama/cuda_func/q4_matmul.cu",
            "auto_quant/kernels/exllama/cuda_func/q4_matrix.cu"
        ], extra_compile_args=extra_compile_args
    )
]

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    **common_setup_kwargs
)