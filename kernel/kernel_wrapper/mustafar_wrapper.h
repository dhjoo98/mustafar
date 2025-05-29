#pragma once 
#include <torch/extension.h>
#include <cuda_runtime.h> 



torch::Tensor mustafar_key_formulation(
    //torch::Tensor A,  // In PyTorch level, this will be NULL
    torch::Tensor bmp, // torch.int64
    torch::Tensor NZ, //torch.float16 -> Must be recast to uint4* 
    torch::Tensor idx, //torch.int32
    torch::Tensor NZ_Offset, //torch.int32
    torch::Tensor B, //torch.float16
    //torch::Tensor C, //torch.float16
    //torch::Tensor Reduction_Workspace, // torch.float16 // cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    int M_Global,
    //int N_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups
);

torch::Tensor mustafar_value_formulation(
    //torch::Tensor A,  // In PyTorch level, this will be NULL
    torch::Tensor bmp, // torch.int64
    torch::Tensor NZ, //torch.float16 -> Must be recast to uint4* 
    torch::Tensor idx, //torch.int32
    torch::Tensor NZ_Offset, //torch.int32
    torch::Tensor B, //torch.float16
    //torch::Tensor C, //torch.float16
    torch::Tensor Reduction_Workspace, // torch.float16 // cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    int M_Global,
    //int N_Global,
    int K_Global, 
    int Batch_Size, 
    int num_key_value_groups
);