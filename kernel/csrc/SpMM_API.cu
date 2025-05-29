/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void print_packed_halfs(uint32_t packed_value) {
    // Extract the first half (lower 16 bits)
    half first_half = (half)(packed_value & 0xFFFF);  // Mask to get the lower 16 bits

    // Extract the second half (upper 16 bits)
    half second_half = (half)((packed_value >> 16) & 0xFFFF);  // Shift right and mask to get the upper 16 bits

    // Print the two half values
    printf("First half: %f\n", __half2float(first_half));  // Convert half to float for readable output
    printf("Second half: %f\n", __half2float(second_half));
}

template<typename TilingConfig, typename SparseKernelConfig>
static void Key_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint64_t* bmp, 
                                  const uint4* NZ,
                                  //const uint32_t* NZ, 
                                  const uint32_t* idx,
                                  //const uint32_t* bmp_idx_offset, 
                                  const uint32_t* NZ_offset,
                                  //const uint4* Compressed_A,
                                  //const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K, 
                                  const int    Batch_Size, 
                                  const int    num_key_value_groups)
{
    Split_K = 1;
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        Key_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
        //fatter N size might benefit from dimN larger than 1. (1 is the preset for Coruscant)
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    //dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
        //each M tiled row handled by SplitK TBs.
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    

    //std::cout << "----SpMM_SplitK_Kernel_Ex(): Shared Memory Size: " << SHMEM_SZ << " Bytes" << std::endl;
    //std::cout << "----SpMM_SplitK_Kernel_Ex(): GridDim: " << dimN << "x" << dimM << " BlockDim: " << WARP_SIZE * TilingConfig::BLOCK_WARPS << "x1x1" << std::endl;
        // GridDim: 1x196: (7168/256) * 7(Split_K)
    // stream is just the GPU job_ID.
    Key_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ, idx, /*bmp_idx_offset,*/ NZ_offset, //Compressed_A, TileOffsets, 
        B, Reduction_Workspace, M_Global, N_Global, K_Global, 1, Batch_Size, num_key_value_groups); //explicitly set Split_K to 1. 
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t Key_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ,
                            //const uint32_t* NZ, 
                            const uint32_t* idx,
                            const uint32_t* NZ_offset,
                            //const uint4* Compressed_A,
                            //const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K, //given that this is always 1. 
                            const int    Batch_Size, 
                            const int    num_key_value_groups)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API.cu/SpMM_SplitK_API(): Entering SpMM_SplitK_API----\n");
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    //printf("Beginning of SpMM_SplitK_Kernel_Ex, N_Global is %d\n", N_Global); donghyeon: it's just the input.
    switch (N_Global) {

        case 8:
            Key_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx, NZ_offset,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K, Batch_Size, num_key_value_groups);
            break;

    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    //dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    //dim3 BlockDim(WARP_SIZE, 1, 1);
    //SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}


template<typename TilingConfig, typename SparseKernelConfig>
static void Value_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint64_t* bmp, 
                                  const uint4* NZ,
                                  //const uint32_t* NZ, 
                                  const uint32_t* idx,
                                  //const uint32_t* bmp_idx_offset, 
                                  const uint32_t* NZ_offset,
                                  //const uint4* Compressed_A,
                                  //const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K, 
                                  const int    Batch_Size, 
                                  const int    num_key_value_groups)
{
    //Split_K = 1;
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        Value_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    //printf("DEBUG: testing if this is reflected to pip\n");
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
        //fatter N size might benefit from dimN larger than 1. (1 is the preset for Coruscant)
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    //dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
        //each M tiled row handled by SplitK TBs.
    dim3 GridDim(dimN, dimM, Batch_Size);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);

    //std::cout << "----SpMM_SplitK_Kernel_Ex(): Shared Memory Size: " << SHMEM_SZ << " Bytes" << std::endl;
    //if DEBUG: std::cout << "----SpMM_SplitK_Kernel_Ex(): GridDim: " << dimN << "x" << dimM << "x" << Batch_Size << " BlockDim: " << WARP_SIZE * TilingConfig::BLOCK_WARPS << "x1x1" << std::endl;
        // GridDim: 1x196: (7168/256) * 7(Split_K)
    // stream is just the GPU job_ID.
    Value_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, bmp, NZ, idx, /*bmp_idx_offset,*/ NZ_offset, //Compressed_A, TileOffsets, 
        B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, Batch_Size, num_key_value_groups); //explicitly set Split_K to 1. AHH [05/22]
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t Value_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ,
                            //const uint32_t* NZ, 
                            const uint32_t* idx,
                            const uint32_t* NZ_offset,
                            //const uint4* Compressed_A,
                            //const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K, //given that this is always 1. 
                            const int    Batch_Size, 
                            const int    num_key_value_groups)
{
#ifdef DEBUG_MODE
    printf("--- SpMM_API.cu/SpMM_SplitK_API(): Entering SpMM_SplitK_API----\n");
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    //assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1){
        //printf("Split_K is 1, so no reduction is needed\n");
        SpMM_SplitK_OutputPTR = C;
    }
    else{
        //printf("Reduction Workspace is selected as output location\n");
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    }
   
    switch (N_Global) {
       
        case 8:
            //SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(
            Value_SplitK_Kernel_Ex<TilingConfig<2, 1, 1, 1>, SparseKernelConfig<64>>(
                stream, A, bmp, NZ, idx, NZ_offset,//Compressed_A, TileOffsets, 
                B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K, Batch_Size, num_key_value_groups);
            break;
        
    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;
    
    if (Split_K == 1)
        return Error;
   
    //cudaStreamSynchronize(stream);
    //if DEBUG: printf("Starting Reduction with Split_K: %d, Warp_Size: %d\n", Split_K, WARP_SIZE);
    //dim3 GridDim((M_Global * N_Global) / 256, 1, Batch_Size);
    dim3 GridDim((M_Global * N_Global) / 256, 1, Batch_Size);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K, Batch_Size);
    return cudaGetLastError();
}
