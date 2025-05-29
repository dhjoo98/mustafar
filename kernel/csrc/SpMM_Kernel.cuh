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


#include "MatMulUtilities.cuh"
#include <vector>

#define DEBUG 0
#define DEBUG2 0
#define DEBUG1 0



template<typename TilingConfig, typename SparseKernelConfig>

__device__ __forceinline__ void SpMM_CopyFromGlobalToReg(//uint32_t* Registers_nz,
                                                         uint32_t    Registers_nz[64],
                                                         uint64_t*    Registers_bmp,
                                                         uint32_t*    Registers_nnz,
                                                         //const uint32_t* GlobalPTR_nz,
                                                         const uint4* GlobalPTR_nz,
                                                         const uint64_t* GlobalPTR_bmp,
                                                         const uint32_t* GlobalPTR_nnz, 
                                                         uint32_t* nnz_tile0, 
                                                         uint32_t* nnz_tile1,
                                                         int startTileIdx) 
{

constexpr int MAX_NZ_PER_BMP_div_2_4 = 8; //first divide by 2 for half, then divide by 4 for uint4. : 64 / 8 = 8
   
    // Each thread handles 2 bitmaps (each of a column)

    #if DEBUG2
        if (blockIdx.x == 0 && blockIdx.y == 383 && threadIdx.x == 127) { //[7168, 7168, 8]  //383
            printf("------Check inside Reg load...\n");
            printf("StartTileIdx: %d\n", startTileIdx);
            printf("bmp0: %u\n", GlobalPTR_bmp[startTileIdx]);
            printf("nnz0: %u\n", GlobalPTR_nnz[startTileIdx]);
        }
    #endif
#pragma unroll     
    for (int i = 0; i < 2; i++) {
        int globalTileIdx = startTileIdx + i;
        // Load bitmap
        Registers_bmp[i] = GlobalPTR_bmp[globalTileIdx];
        Registers_nnz[i] = GlobalPTR_nnz[globalTileIdx]; 

        // Load non-zero values into the register
        uint32_t num_nz_per_bitmap = __popcll(Registers_bmp[i]);
        if (i){
            *nnz_tile1 = num_nz_per_bitmap; //This is the number of halfs
        }
        else{
            *nnz_tile0 = num_nz_per_bitmap;
        }

        // Load non-zero elements (half precision) into the register
#pragma unroll 
        for (int j = 0; j < MAX_NZ_PER_BMP_div_2_4 ; j++) { //8 iterations to copy the 4 x packed two fp16s.
            //loading Vectors 
            if (j <= num_nz_per_bitmap / 8 ) {
            //if (j < num_nz_per_bitmap / 8 ) {
                //**Registers_nnz is in 'uint32' units. 
                Registers_nz[i * 32 + j * 4 + 0] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].x; // load nz
                Registers_nz[i * 32 + j * 4 + 1] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].y; // load nz
                Registers_nz[i * 32 + j * 4 + 2] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].z; // load nz
                Registers_nz[i * 32 + j * 4 + 3] = GlobalPTR_nz[Registers_nnz[i] / 4 + j].w; // load nz
            }
        }
    }
}

// Init Shared Memory to 0
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    //
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;
    //
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;
    //
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}


template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared(half* __restrict__ SharedPTR,
                                                                    uint32_t Registers_nz[64],
                                                                    uint64_t* Registers_bmp,
                                                                    uint32_t* nnz_tile0, 
                                                                    uint32_t* nnz_tile1,
                                                                    int TB_ROW, 
                                                                    int TB_COL)
                                                                    //int tileIdx)
{
    //tildIdx = 2*tid = nth 64x1 tile to start with. 
//entire smem space is 256x64. 
int tile_element_start = TB_ROW * 64 * 64 + TB_COL * 2;
#pragma unroll
    for (int i = 0; i < 2; i++) {
         // Reinterpret Registers_nz as half*
        half* nz_values = reinterpret_cast<half*>(Registers_nz+i*32);

        uint64_t bmp = Registers_bmp[i];
        int pos1 = 0;  // Initialize pos1 before processing rows

        // Precompute tile positions
        int fuk = tile_element_start + i;
        //int tileCol = 64 * (tileIdx + i);

        uint32_t nnz_tile = i? *nnz_tile1 : *nnz_tile0;


    #pragma unroll
        for (int j = 0; j < 64; j++){
            if (j == nnz_tile){
                break; //becomes inactive thread, waits for other threads to finish. 
            }
            pos1 = __clzll(bmp); 
            bmp &= ~(0x8000000000000000 >> pos1);

            int output_idx = fuk + (pos1 << 6);
            SharedPTR[output_idx] = nz_values[j];

            pos1++;
        }
    }
}




template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
//__maxnreg__(255)
Key_Kernel(const half*  A,
                            const uint64_t* bmp, 
                            const uint4* NZ, 
                            const uint32_t* idx,
                            const uint32_t* NZ_offset,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K,
                            const int    Batch_Size, 
                            const int    num_key_value_groups)
{    

    const int mustafar_batch_id = blockIdx.z;
    const int mustafar_group_id = blockIdx.z / num_key_value_groups;
    // Access batched data using offsets
    const uint4* NZ_batch = NZ + NZ_offset[mustafar_group_id]; 
    //const uint32_t* idx_batch = idx + bmp_idx_offset[mustafar_batch_id];
    const uint32_t* idx_batch = idx + mustafar_group_id * (1 + M_Global * K_Global / 64); //because idx has 1 extra element per batch. 
    const uint64_t* bmp_batch = bmp + mustafar_group_id * (M_Global * K_Global / 64);

    // Access B and C with strides
    const half* B_batch = B + mustafar_batch_id * K_Global * N_Global;
    //const half* B_batch = B + mustafar_batch_id * K_Global; //note that spmm_debug has not been updated yet. 
    half* C_batch = Reduction_Workspace + mustafar_batch_id * M_Global * N_Global;

    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M); //M_Global / TILE_M: tiling the M dimension of Matrix A.
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x; //block DimX is 1 for skinny matrices (see SpMM_API/line 42)
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);  //blockIdx.y % (num M Tile rows): wrap around num_tile_rows
        //i.e., TB0, TB(num M tile rows), TB(2*num M tile rows) .. handle the first M tile row
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    #if DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) { //[7168, 7168, 8]
            printf("------K dimension related Debugging info...\n");
            printf("NumKBlock: %d\n", NumKBlock); //112: how many iterations it takes to finish computing that output tile
            printf("AverageNumKBlock: %d\n", AverageNumKBlock); //16: 
            printf("RoundedKBlock: %d\n", RoundedKBlock); //112: related to the padding
            printf("PaddingKBlock: %d\n", PaddingKBlock); //0: re  lated to the padding
            printf("NumIter: %d\n", NumIter); //16: thus the final conclusion
        }
    #endif

    //the following will reside in SMSP regfile
    uint64_t Registers_bmp[2];  //4 regs
    uint32_t Registers_nnz[2];  //2 regs
    uint32_t Registers_nz[64];  //64 regs // Enough to hold non-zero values for 2 tiles 
    uint32_t nnz_tile0;
    uint32_t nnz_tile1;

    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned 

    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //    printf("#1 TilingConfig::WARP_COL_TENSORS: %d\n", TilingConfig::WARP_COL_TENSORS); //1 for sub-16, 2 for 32. 
    //}
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];//[8][4] = 32 uint32 
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; //[8][4] = 32 uint32
    // copying B tile from GlobalMemory to SharedMemory
    //const half* BTileGlobalPTR = //B was supposed to be col-major. 
    //    B + Tile_Start_N * K_Global
    //    + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //
    const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;

    //my definition ~ see whiteboard and paper
    //int BaseTileIdx = y * (32 * K_Global / 8) + BatchID * K_Global / (8*Split_K); //For original 8x8 tile
    //int BaseTileIdx = y * (4 * K_Global) + BatchID * K_Global / Split_K; //For 1-64 col tiles. 
    int BaseTileIdx = y * (4 * K_Global) + BatchID * K_Global / Split_K; //For 1-64 col tiles. new ver (2/7) -> hm looks correct? 
    //below changed to allow the column-wise bitmap format. 
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;
    //int StartTileIdx = BaseTileIdx + TB_Row * K_Global / 8 + TB_Col * 2; 
    int StartTileIdx = BaseTileIdx + TB_Row * K_Global + TB_Col * 2;
    //int StartTileIdx = BaseTileIdx + tid_times_2 -2;
    //int tileIdx = 2 * tid; // for 64x1 local index for DecompressFromRegisterToShared (64x64)


    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp, 
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0, 
                                                                &nnz_tile1,
                                                                StartTileIdx); 

    SpMM_InitSharedMemory<TilingConfig>(smem); //rst_smem
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global); //ld_dense: this is async, defined in MatMulUtilies.cuh
    cp_async_group_commit();
    
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16]; // [4*4][8 in TilingConfig] = 64 floats
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();

    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    //SharedPTR,
                                                                    smem,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0, 
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2 
    //
    cp_async_wait_group<0>();
    __syncthreads();
     #if DEBUG
        //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
        if ( (threadIdx.x == 31 | threadIdx.x == 127) && blockIdx.x == 0 && blockIdx.y == 0){ //Debugging 256x64 for rectangular sanity
        //if (blockIdx.x == 0 && blockIdx.y == 0){
        //if (blockIdx.x == 127){
                printf("---Exit SpMM Decompression...\n \
                For thread %d, blockIdx.x: %d, blockIdx.y: %d, mustafar_batch_id: %d\n \
                StartTileIdx, the access index for bmp and nnz: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, mustafar_batch_id, StartTileIdx);
            }
        __syncthreads(); // only for debugging
    #endif
    //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
    StartTileIdx +=64;


//
// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1) //unroll exactly once.
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) { //remove the last iteration and move computation to epilogue
        
        const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);

        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); //place for 256x64 A and 64x16 B (or TileN=32)
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR); //rst_smem
        cp_async_group_commit();
        
        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp,
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0,
                                                                &nnz_tile1, 
                                                                StartTileIdx); 

        // Copying B Tile
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);  //ld_dense
        cp_async_group_commit();


        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    smem_write_PTR,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0,
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2
            
            //smem_write_PTR,
            //Registers_GlobalToShared,
            //NNZ_ThreadLocal1,
            //smem_write_PTR + TilingConfig::TILE_M * TILE_K / 2,
            //Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            //NNZ_ThreadLocal2); //extract 
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
        //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
        StartTileIdx += 64;

    }
    
    
    //add epliogue
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
    __syncthreads();
    //end of epliogue

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    //half* BlockGlobalPTR =
    //    Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
    //half* BlockGlobalPTR = C_batch + Tile_Start_M + Tile_Start_N * M_Global;
    half* BlockGlobalPTR =
        C_batch + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;

    #if DEBUG
        //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
        if ( (threadIdx.x == 31 | threadIdx.x == 127) && blockIdx.x == 0 && blockIdx.y == 0){ //Debugging 256x64 for rectangular sanity
        //if (blockIdx.x == 0 && blockIdx.y == 0){
        //if (blockIdx.x == 127){
                printf("---Exit StoreToSharedMemoryFromRegister(), Entering write to global memory...\n \
                For thread %d, blockIdx.x: %d, blockIdx.y: %d, mustafar_batch_id: %d\n \
                StartTileIdx, the access index for bmp and nnz: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, mustafar_batch_id, StartTileIdx);
            }
        __syncthreads(); // only for debugging
    #endif
    
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}

template<typename TilingConfig, typename SparseKernelConfig>
__global__ void 
//__maxnreg__(255)
Value_Kernel(const half*  A,
                            const uint64_t* bmp, 
                            //const uint32_t* NZ,
                            const uint4* NZ, 
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


    //batched SpMV ID from Z dimension
    //const int mustafar_batch_id = blockIdx.z;
    //Logic for supporting GQA. 
        //Multiple Batches work on same compressed KV, but use different vector and output location.
    const int mustafar_batch_id = blockIdx.z; //Batch num
    const int mustafar_group_id = blockIdx.z / num_key_value_groups; //GQA number
    // Access batched data using offsets
    const uint4* NZ_batch = NZ + NZ_offset[mustafar_group_id]; 
    //const uint32_t* idx_batch = idx + bmp_idx_offset[mustafar_batch_id];
    const uint32_t* idx_batch = idx + mustafar_group_id * (1 + M_Global * K_Global / 64); //because idx has 1 extra element per batch. 
    const uint64_t* bmp_batch = bmp + mustafar_group_id * (M_Global * K_Global / 64);

    const half* B_batch = B + mustafar_batch_id * K_Global * N_Global;

    half* C_batch = Reduction_Workspace + mustafar_batch_id * M_Global * N_Global * Split_K;


    
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M); //M_Global / TILE_M: tiling the M dimension of Matrix A.
        //(M_Global / TilingConfig::TILE_M) is 1 for Value formulation. 
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x; //block DimX is 1 for skinny matrices (see SpMM_API/line 42)
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);  //blockIdx.y % (num M Tile rows): wrap around num_tile_rows
        //i.e., TB0, TB(num M tile rows), TB(2*num M tile rows) .. handle the first M tile row
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;


    //the following will reside in SMSP regfile
    uint64_t Registers_bmp[2];  //4 regs
    uint32_t Registers_nnz[2];  //2 regs
    uint32_t Registers_nz[64];  //64 regs // Enough to hold non-zero values for 2 tiles 
    uint32_t nnz_tile0;
    uint32_t nnz_tile1;

    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned 

    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //    printf("#1 TilingConfig::WARP_COL_TENSORS: %d\n", TilingConfig::WARP_COL_TENSORS); //1 for sub-16, 2 for 32. 
    //}
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];//[8][4] = 32 uint32 
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; //[8][4] = 32 uint32
    // copying B tile from GlobalMemory to SharedMemory
    //const half* BTileGlobalPTR = //B was supposed to be col-major. 
    //    B + Tile_Start_N * K_Global
    //    + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //

    const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K;


 
    int BaseTileIdx = BatchID * (M_Global/64) * (K_Global / Split_K); // y ==0 when M_GLOBAL = TILE_M = 128.
        //This works because: K_Global/SplitK already accounts for each 'tile' 
    
    //below changed to allow the column-wise bitmap format. 
    int tid = threadIdx.x;
    int TB_Row = tid / 32;
    int TB_Col = tid % 32;


    //code for value SpMV
    int StartTileIdx = BaseTileIdx + TB_Row * 64 + TB_Col * 2;



    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp, 
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0, 
                                                                &nnz_tile1,
                                                                StartTileIdx); 


    SpMM_InitSharedMemory<TilingConfig>(smem); //rst_smem
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global); //, true); //ld_dense: this is async, defined in MatMulUtilies.cuh
    
    //CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>( 
    //    smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, true, K_Global); //ld_dense: this is async, defined in MatMulUtilies.cuh
    cp_async_group_commit();
    
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16]; // [4*4][8 in TilingConfig] = 64 floats
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();

    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    //SharedPTR,
                                                                    smem,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0, 
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2 
    //
    cp_async_wait_group<0>();
    __syncthreads();

    StartTileIdx += M_Global;




#pragma unroll(1) //unroll exactly once.
    for (int tile_id_k = 0; tile_id_k < NumIter-1; tile_id_k++) { //remove the last iteration and move computation to epilogue
        

        const half* BTileGlobalPTR = B_batch + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);

        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N); //place for 256x64 A and 64x16 B (or TileN=32)
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR); //rst_smem
        cp_async_group_commit();

        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_nz,
                                                                Registers_bmp,
                                                                Registers_nnz,
                                                                //NZ, 
                                                                //bmp,
                                                                //idx,
                                                                NZ_batch,
                                                                bmp_batch,
                                                                idx_batch,
                                                                &nnz_tile0,
                                                                &nnz_tile1, 
                                                                StartTileIdx); 

        // Copying B Tile
        //CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        ////    smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy, K_Global);  //ld_dense
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);  //ld_dense
        cp_async_group_commit();


        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
                                                                    smem_write_PTR,
                                                                    Registers_nz,
                                                                    Registers_bmp,
                                                                    &nnz_tile0,
                                                                    &nnz_tile1,
                                                                    TB_Row, 
                                                                    TB_Col);
                                                                    //tileIdx); //make sure to keep this tid * 2
            

        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
        //StartTileIdx += 8; //for the next 246x64 tile (8 8x8 block apart row-wise)
         // code for key SpMV
        //StartTileIdx +=64;
        // code for value SpMV
        StartTileIdx += M_Global;

    }
    

    
    //add epliogue
    half* __restrict__ smem_read_PTR  = smem;
    smem_read_PTR  = smem + ((NumIter-1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
    PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col); //compute
    __syncthreads();
    //end of epliogue

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();

    
    half* BlockGlobalPTR =
        C_batch + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
        //BatchID is effectively the SplitK number. 
        


//int RDWS_write_offset = BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global + mustafar_batch_id * M_Global * N_Global * Split_K;

#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS){  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE) { // j-th row
           BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
        }
    }
}

