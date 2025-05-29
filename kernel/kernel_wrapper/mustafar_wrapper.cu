#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

//Note that <Aten/Aten.h> and at::kHalf can also be used. 

#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#include "mustafar_wrapper.h"

#include "SpMM_API.cuh"

//Make sure that all calls to this operation use the same Reduction Workspace. 
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
) 
{
    // Check if the input tensors are on the same device
    if (B.device() != bmp.device() || B.device() != NZ.device() || B.device() != idx.device() || B.device() != NZ_Offset.device()){ //|| B.device() != Reduction_Workspace.device()) {
        throw std::runtime_error("All input tensors must be on the same device.");
    }
    

    // Check if the input tensors are of type float16
        // Distinguish between at::Half and at::kHalf. one is a type name, the other is a enumerator. 
    if (B.dtype() != at::kHalf) {
        throw std::runtime_error("Tensor B must be of type float16.");
    }
    //if (C.dtype() != at::kHalf) {
    //    throw std::runtime_error("Tensor C must be of type float16.");
    //}
    if (NZ.dtype() != at::kHalf) {
        throw std::runtime_error("Tensor NZ must be of type float16.");
    }
    //if (Reduction_Workspace.dtype() != at::kHalf) {
    //    throw std::runtime_error("Tensor Reduction_Workspace must be of type float16.");
    //}
    if (bmp.dtype() != at::kLong) {
        throw std::runtime_error("Tensor bmp must be of type int64.");
    }
    if (idx.dtype() != at::kInt) {
        throw std::runtime_error("Tensor idx must be of type int.");
    }
    if (NZ_Offset.dtype() != at::kInt) {
        throw std::runtime_error("Tensor NZ_Offset must be of type int.");
    }

    TORCH_CHECK(
        bmp.is_contiguous() && NZ.is_contiguous() && idx.is_contiguous() && B.is_contiguous() && NZ_Offset.is_contiguous(),  //&& Reduction_Workspace.is_contiguous(),
        "bmp, NZ, idx, B, C, and Reduction_Workspace tensors must be contiguous."
        );

    TORCH_CHECK(
    bmp.is_cuda() && NZ.is_cuda() && idx.is_cuda() && B.is_cuda() && NZ_Offset.is_cuda(), //&& Reduction_Workspace.is_cuda(),
    "bmp, NZ, idx, B, C, and (not)Reduction_Workspace tensors must be on CUDA device."
        );

    // Get the CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    
    //printf("Allocating output tensor C\n");
    //auto C = torch::empty({Batch_Size, M_Global, 8}, B.options());
    auto C = torch::zeros({Batch_Size, 8, M_Global}, B.options());
    //auto Reduction_Workspace = torch::empty({M_Global, N_Global, 8}, B.options());

    // casting torch as pointers 
    //const at::Half* A_aten_ptr = A.data_ptr<at::Half>();
    //const half* A_cuda_ptr = reinterpret_cast<const half*>(A_aten_ptr);
    
    //const at::Long* bmp_aten_ptr = bmp.data_ptr<at::Long>();
    //const uint64_t* bmp_cuda_ptr = reinterpret_cast<const uint64_t*>(bmp_aten_ptr);
    bmp = bmp.to(at::kUInt64);
    const uint64_t* bmp_cuda_ptr = bmp.data_ptr<uint64_t>();
    
    const at::Half* NZ_aten_ptr = NZ.data_ptr<at::Half>();
    const uint4* NZ_cuda_ptr = reinterpret_cast<const uint4*>(NZ_aten_ptr);
    
    //const at::Int* idx_aten_ptr = idx.data_ptr<at::Int>();
    //const uint32_t* idx_cuda_ptr = reinterpret_cast<const uint32_t*>(idx_aten_ptr);
    //idx = idx.to(at::kUInt);
    const int32_t* idx_aten_ptr = idx.data_ptr<int32_t>();
    const uint32_t* idx_cuda_ptr = reinterpret_cast<const uint32_t*>(idx_aten_ptr);

    const int32_t* NZ_Offset_aten_ptr = NZ_Offset.data_ptr<int32_t>();
    const uint32_t* NZ_Offset_cuda_ptr = reinterpret_cast<const uint32_t*>(NZ_Offset_aten_ptr);
    
    const at::Half* B_aten_ptr = B.data_ptr<at::Half>();
    const half* B_cuda_ptr = reinterpret_cast<const half*>(B_aten_ptr);
    at::Half* C_aten_ptr = C.data_ptr<at::Half>();
    half* C_cuda_ptr = reinterpret_cast<half*>(C_aten_ptr);
    //at::Half* Reduction_Workspace_aten_ptr = Reduction_Workspace.data_ptr<at::Half>();
    //half* Reduction_Workspace_cuda_ptr = reinterpret_cast<half*>(Reduction_Workspace_aten_ptr);

    // Call the CUDA kernel
    Key_SplitK_API(stream,
        static_cast<half*>(nullptr), //const half*  A,
        bmp_cuda_ptr, //const uint64_t* bmp, 
        NZ_cuda_ptr, //const uint4* NZ,
        //const uint32_t* NZ, 
        idx_cuda_ptr, //const uint32_t* idx,
        NZ_Offset_cuda_ptr, //const uint32_t* NZ_offset,
        //const uint4* Compressed_A,
        //const int*   TileOffsets,
        B_cuda_ptr, //const half*  B,
        C_cuda_ptr, //half*        C,
        M_Global,
        8, //N_Global,
        K_Global,
        static_cast<half*>(nullptr), //half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
        1, //int          Split_K, //given that this is always 1. 
        Batch_Size, 
        num_key_value_groups); //const int    Batch_Size)

    return C;
}




//Make sure that all calls to this operation use the same Reduction Workspace. 
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
) 
{
    // Check if the input tensors are on the same device
    if (B.device() != bmp.device() || B.device() != NZ.device() || B.device() != idx.device() || B.device() != NZ_Offset.device()){ //|| B.device() != Reduction_Workspace.device()) {
        throw std::runtime_error("All input tensors must be on the same device.");
    }
    

    // Check if the input tensors are of type float16
        // Distinguish between at::Half and at::kHalf. one is a type name, the other is a enumerator. 
    if (B.dtype() != at::kHalf) {
        throw std::runtime_error("Tensor B must be of type float16.");
    }
    //if (C.dtype() != at::kHalf) {
    //    throw std::runtime_error("Tensor C must be of type float16.");
    //}
    if (NZ.dtype() != at::kHalf) {
        throw std::runtime_error("Tensor NZ must be of type float16.");
    }
    //if (Reduction_Workspace.dtype() != at::kHalf) {
    //    throw std::runtime_error("Tensor Reduction_Workspace must be of type float16.");
    //}
    if (bmp.dtype() != at::kLong) {
        throw std::runtime_error("Tensor bmp must be of type int64.");
    }
    if (idx.dtype() != at::kInt) {
        throw std::runtime_error("Tensor idx must be of type int.");
    }
    if (NZ_Offset.dtype() != at::kInt) {
        throw std::runtime_error("Tensor NZ_Offset must be of type int.");
    }

    TORCH_CHECK(
        //bmp.is_contiguous() && NZ.is_contiguous() && idx.is_contiguous() && B.is_contiguous() && NZ_Offset.is_contiguous(),  //&& Reduction_Workspace.is_contiguous(),
        bmp.is_contiguous() && NZ.is_contiguous() && idx.is_contiguous() && NZ_Offset.is_contiguous(),  //&& Reduction_Workspace.is_contiguous(),
        "bmp, NZ, idx, B, C, and Reduction_Workspace tensors must be contiguous."
        );

    TORCH_CHECK(
    bmp.is_cuda() && NZ.is_cuda() && idx.is_cuda() && B.is_cuda() && NZ_Offset.is_cuda(), //&& Reduction_Workspace.is_cuda(),
    "bmp, NZ, idx, B, C, and (not)Reduction_Workspace tensors must be on CUDA device."
        );

    // Get the CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int Split_K = 1;
    //printf("Allocating output tensor C\n");
    //auto C = torch::empty({Batch_Size, M_Global, 8}, B.options());
    auto C = torch::zeros({Batch_Size, 8, M_Global}, B.options());
    //auto Reduction_Workspace = torch::empty({M_Global, 8, Split_K, Batch_Size}, B.options());

    // casting torch as pointers 
    //const at::Half* A_aten_ptr = A.data_ptr<at::Half>();
    //const half* A_cuda_ptr = reinterpret_cast<const half*>(A_aten_ptr);
    
    //const at::Long* bmp_aten_ptr = bmp.data_ptr<at::Long>();
    //const uint64_t* bmp_cuda_ptr = reinterpret_cast<const uint64_t*>(bmp_aten_ptr);
    bmp = bmp.to(at::kUInt64);
    const uint64_t* bmp_cuda_ptr = bmp.data_ptr<uint64_t>();
    
    const at::Half* NZ_aten_ptr = NZ.data_ptr<at::Half>();
    const uint4* NZ_cuda_ptr = reinterpret_cast<const uint4*>(NZ_aten_ptr);
    
    //const at::Int* idx_aten_ptr = idx.data_ptr<at::Int>();
    //const uint32_t* idx_cuda_ptr = reinterpret_cast<const uint32_t*>(idx_aten_ptr);
    //idx = idx.to(at::kUInt);
    const int32_t* idx_aten_ptr = idx.data_ptr<int32_t>();
    const uint32_t* idx_cuda_ptr = reinterpret_cast<const uint32_t*>(idx_aten_ptr);

    const int32_t* NZ_Offset_aten_ptr = NZ_Offset.data_ptr<int32_t>();
    const uint32_t* NZ_Offset_cuda_ptr = reinterpret_cast<const uint32_t*>(NZ_Offset_aten_ptr);
    
    const at::Half* B_aten_ptr = B.data_ptr<at::Half>();
    const half* B_cuda_ptr = reinterpret_cast<const half*>(B_aten_ptr);
    at::Half* C_aten_ptr = C.data_ptr<at::Half>();
    half* C_cuda_ptr = reinterpret_cast<half*>(C_aten_ptr);
    //at::Half* Reduction_Workspace_aten_ptr = Reduction_Workspace.data_ptr<at::Half>();
    //half* Reduction_Workspace_cuda_ptr = reinterpret_cast<half*>(Reduction_Workspace_aten_ptr);

    //half* Reduction_Workspace_cuda_ptr = NULL;
    //cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace_cuda_ptr), sizeof(half) * M_Global * 8 * Split_K * Batch_Size);
    //const at::Half* Reduction_Workspace_aten_ptr = Reduction_Workspace.data_ptr<at::Half>();
    //const half* Reduction_Workspace_cuda_ptr = reinterpret_cast<const half*>(Reduction_Workspace_aten_ptr);
    at::Half* Reduction_Workspace_aten_ptr = Reduction_Workspace.data_ptr<at::Half>();
    half* Reduction_Workspace_cuda_ptr = reinterpret_cast<half*>(Reduction_Workspace_aten_ptr);
    

    // Call the CUDA kernel
    Value_SplitK_API(stream,
        static_cast<half*>(nullptr), //const half*  A,
        bmp_cuda_ptr, //const uint64_t* bmp, 
        NZ_cuda_ptr, //const uint4* NZ,
        //const uint32_t* NZ, 
        idx_cuda_ptr, //const uint32_t* idx,
        NZ_Offset_cuda_ptr, //const uint32_t* NZ_offset,
        //const uint4* Compressed_A,
        //const int*   TileOffsets,
        B_cuda_ptr, //const half*  B,
        C_cuda_ptr, //half*        C,
        M_Global,
        8, //N_Global,
        K_Global,
        //static_cast<half*>(nullptr), //half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
        Reduction_Workspace_cuda_ptr, //half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
        Split_K, //int          Split_K, //given that this is always 1. 
        Batch_Size, 
        num_key_value_groups); //const int    Batch_Size)

    return C;
}



