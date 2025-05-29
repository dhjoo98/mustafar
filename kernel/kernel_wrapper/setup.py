#setup.py 

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import find_packages, setup
import torch.utils.cpp_extension
import os
import torch
import site
import sys

# Get the current Python's site-packages directory
site_packages = site.getsitepackages()[0]

# Find torch's lib directory relative to site-packages
torch_lib_path = os.path.join(site_packages, 'torch', 'lib')

# Verify the path exists
if not os.path.exists(torch_lib_path):
    raise RuntimeError(f"Could not find torch lib directory at {torch_lib_path}")

os.environ["TORCH_CUDA_ARCH_LIST"] = "Ada"  

setup(
    name='mustafar_batched_spmv_package', #pip package name
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='mustafar_package', #import module name
            sources=['pybind.cpp', 'mustafar_wrapper.cu'], 
            extra_objects=['../build/SpMM_API.o'],          
            include_dirs=[torch.utils.cpp_extension.include_paths(), '../build'],  
            extra_link_args=[f"-L{torch_lib_path}", "-lc10"],
        )
    ],

    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=["torch"]
)

#should now work with pip install -e . 