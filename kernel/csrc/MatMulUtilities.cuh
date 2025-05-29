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
#ifndef MatMulUtilities_H
#define MatMulUtilities_H
// C = A*B
// C: col major
// A: row major
// B: col major

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "AsyncCopy_PTX.cuh"
#include "MMA_PTX.cuh"
#include "TilingConfig.h"

int cuda_CheckError()
{
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
}

// New features: Copy size is X * 64, X can be any multiple to 8
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS //NumOfRowsToCopy = 8
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride, //K_Global
                                                                bool        Pred = true) //to stop copying at the end of pipeline
{
    //
    int lane_id       = threadIdx.x % 32; 
    int col           = lane_id % 8;
    int row1          = lane_id / 8;
    int row2          = lane_id / 8 + 4; //offset by 4

    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS; //#define COPY_UNIT_FP16_ROWS 8 // == 1
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); //distinguishes the different warps
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
            //this predicates off the warps other than warp0 
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K, //copy 16Bs, which is 8 halfs.
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);

    }
}

template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputations(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{

//WARP_ROW_TENSORS: number of row tiles the warp processes
//WARP_COL_TENSORS: number of col tiles the warp processes
//N8: whether output marix has 8 cols (16 cols if false)
//TILE_M: Tile size
//TILE_K: Tile size. 


    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    // First Register Loading
    FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR, warp_start_row, 0); //defined in MMA_PTX.cuh
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, 0);

#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) { //BLOCK_K_TENSOR also defined in TilingConfig. 
        //ping-pong buffering
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[4] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        //ping-pong buffers for alternating loading and computing.
        a_read += ((k) % 2) * WARP_ROW_TENSORS;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR, warp_start_row, (k + 1) * MMA_K);
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, (k + 1) * MMA_K);
        }

        //matrix multiplication
        //for each tile in M and N dimension, compute the matrix multiplication.
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) //4
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) { //1 for N8, 2 for larger batch size. 
                // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j]); //Warp-level, innately synchronizing.
                    //but in/output of MMA distributed across the warp threads. 
                if (!TilingConfig::N8) //additional computation for N16
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
            }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

#endif