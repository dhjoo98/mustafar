//pybind.cpp 
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "mustafar_wrapper.h"

// Expose the function to Python using PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch extension for Mustafar batched spmv CUDA kernel";
    m.def("mustafar_key_formulation", &mustafar_key_formulation, "A CUDA implementation of a key formulation kernel called 'mustafar_key_formulation'");
    m.def("mustafar_value_formulation", &mustafar_value_formulation, "A CUDA implementation of a key formulation kernel called 'mustafar_key_formulation'");
}