import torch 
import triton
import triton.language as tl
import numpy as np
import gc


@triton.jit
def calculate_bitmap_key_batched(
    input_ptr,        # [B * num_tiles_per_batch * 64]
    bitmaps_ptr,      # [B * num_tiles_per_batch]
    counts_ptr,       # [B * num_tiles_per_batch]
    total_elems: tl.constexpr,
    shifts_ptr,       # [64]
    stride_batch: tl.constexpr,  # = num_tiles_per_batch * 64
    M: tl.constexpr,
    N: tl.constexpr
):
    #batch_id = tl.program_id(0)
    #tile_id = tl.program_id(1)
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)


    #naive version
    #tile_offset = batch_id * stride_batch + tile_id * 64
    #offsets = tl.arange(0, 64)
    #indices = tile_offset + offsets

    #for key version: 
    # Calculate which 64x64 block we're in, accounting for batch
    block_row = tile_id % N
    block_col = tile_id // N
    base_idx = batch_id * stride_batch + block_row * M + block_col * 64
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    #valid = indices < total_elems
    #vals = tl.load(input_ptr + indices, mask=valid, other=0.0)
    vals = tl.load(input_ptr + indices)
    #bit_mask = (vals != 0.0).to(tl.int32)
    bit_mask = tl.where(vals != 0.0, 1, 0)
    shifts = tl.load(shifts_ptr + offsets)  # shifts_ptr[0:64]
    bitmap = tl.sum(bit_mask * shifts, axis=0)

    cnt = tl.sum(bit_mask, axis=0)
    #padding happens here.
    cnt = ((cnt + 7) & ~7) >> 1  # padded to nearest multiple of 8, then halved
    #cnt = (((cnt + 7) // 8) * 8) / 2

    #to address per-tile bitmap and counts. 
    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    tl.store(bitmaps_ptr + flat_tile_index, bitmap)
    tl.store(counts_ptr + flat_tile_index, cnt)

@triton.jit
def calculate_bitmap_value_batched(
    input_ptr,        # [B * num_tiles_per_batch * 64]
    bitmaps_ptr,      # [B * num_tiles_per_batch]
    counts_ptr,       # [B * num_tiles_per_batch]
    total_elems: tl.constexpr,
    shifts_ptr,       # [64]
    stride_batch: tl.constexpr,  # = num_tiles_per_batch * 64
    M: tl.constexpr,
    N: tl.constexpr
):
    #batch_id = tl.program_id(0)
    #tile_id = tl.program_id(1)
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)


    #naive version
    #tile_offset = batch_id * stride_batch + tile_id * 64
    #offsets = tl.arange(0, 64)
    #indices = tile_offset + offsets

    #for key version: 
    # Calculate which 64x64 block we're in, accounting for batch
    #block_row = tile_id % N
    #block_col = tile_id // N
    #base_idx = batch_id * stride_batch + block_row * M + block_col * 64
    #offsets = tl.arange(0, 64)
    #indices = base_idx + offsets

    #for value version: 
    tiles_per_row = N // 64 #number if 64-column segments in a row. 
    tiles_per_block = tiles_per_row * 64 
    block_idx = tile_id // tiles_per_block
    rem = tile_id % tiles_per_block
    col_tile = rem // 64
    r_in_block = rem % 64 
    row = block_idx*64 + r_in_block
    col_start = col_tile * 64 
    base_idx = batch_id * stride_batch + row * N + col_start
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    #valid = indices < total_elems
    #vals = tl.load(input_ptr + indices, mask=valid, other=0.0)
    vals = tl.load(input_ptr + indices)
    #bit_mask = (vals != 0.0).to(tl.int32)
    bit_mask = tl.where(vals != 0.0, 1, 0)
    shifts = tl.load(shifts_ptr + offsets)  # shifts_ptr[0:64]
    bitmap = tl.sum(bit_mask * shifts, axis=0)

    cnt = tl.sum(bit_mask, axis=0)
    #padding happens here.
    cnt = ((cnt + 7) & ~7) >> 1  # padded to nearest multiple of 8, then halved
    #cnt = (((cnt + 7) // 8) * 8) / 2

    #to address per-tile bitmap and counts. 
    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    tl.store(bitmaps_ptr + flat_tile_index, bitmap)
    tl.store(counts_ptr + flat_tile_index, cnt)

@triton.jit
def compress_key_batched(
    input_ptr,          # flattened [B * num_tiles_per_batch * 64]
    bitmaps_ptr,        # flattened [B * num_tiles_per_batch]
    counts_ptr,         # flattened [B * (num_tiles_per_batch + 1)]
    packed_not_ptr,     # flattened output buffer
    batch_offsets_ptr,  # flattened [B]
    total_elems: tl.constexpr,
    stride_batch: tl.constexpr,  # = num_tiles_per_batch * 64
    M: tl.constexpr,
    N: tl.constexpr
):

    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    
    #naive version
    #tile_offset = batch_id * stride_batch + tile_id * 64
    #offsets = tl.arange(0, 64)
    #indices = tile_offset + offsets

    #for key version: 
    # Calculate which 64x64 block we're in, accounting for batch
    block_row = tile_id % N
    block_col = tile_id // N
    base_idx = batch_id * stride_batch + block_row * M + block_col * 64
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    #to address per-tile bitmap and counts. 
    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    
    bitmap = tl.load(bitmaps_ptr + flat_tile_index)
    # Get the base offset for this batch (because accum_count has num_tiles_per_batch + 1 per batch. )
    #idx = tl.load(counts_ptr + flat_tile_index)
    batch_base_idx = batch_id * ((stride_batch // 64) + 1)  # +1 for the extra count per batch
    idx = tl.load(counts_ptr + batch_base_idx + tile_id)
    
    #idx_plus_one = tl.load(counts_ptr + flat_tile_index + 1)
    #cnt = (idx_plus_one - idx) * 2  # number of float16s to store for this tile

    #valid = indices < total_elems
    #vals = tl.load(input_ptr + indices, mask=valid, other=0.0)
    vals = tl.load(input_ptr + indices)
    
    #extract the non-zero lanes. 
    shifted = bitmap >> (63 - offsets)
    bit_mask = (shifted & 1) #.to(tl.int32)

    #slot index. 
    prefix = tl.cumsum(bit_mask, axis=0) - 1
    valid_pos = bit_mask.to(tl.int1)

    batch_base_offset = tl.load(batch_offsets_ptr + batch_id)
    #store_idx = idx * 2 + prefix
    store_idx = batch_base_offset + idx * 2 + prefix
    tl.store(packed_not_ptr + store_idx, vals, mask=valid_pos)


@triton.jit
def compress_value_batched(
    input_ptr,          # flattened [B * num_tiles_per_batch * 64]
    bitmaps_ptr,        # flattened [B * num_tiles_per_batch]
    counts_ptr,         # flattened [B * (num_tiles_per_batch + 1)]
    packed_not_ptr,     # flattened output buffer
    batch_offsets_ptr,  # flattened [B]
    total_elems: tl.constexpr,
    stride_batch: tl.constexpr,  # = num_tiles_per_batch * 64
    M: tl.constexpr,
    N: tl.constexpr
):

    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    
    #naive version
    #tile_offset = batch_id * stride_batch + tile_id * 64
    #offsets = tl.arange(0, 64)
    #indices = tile_offset + offsets

    #for key version: 
    # Calculate which 64x64 block we're in, accounting for batch
    #block_row = tile_id % N
    #block_col = tile_id // N
    #base_idx = batch_id * stride_batch + block_row * M + block_col * 64
    #offsets = tl.arange(0, 64)
    #indices = base_idx + offsets

    #for value version: 
    tiles_per_row = N // 64 #number if 64-column segments in a row. 
    tiles_per_block = tiles_per_row * 64 
    block_idx = tile_id // tiles_per_block
    rem = tile_id % tiles_per_block
    col_tile = rem // 64
    r_in_block = rem % 64 
    row = block_idx*64 + r_in_block
    col_start = col_tile * 64 
    base_idx = batch_id * stride_batch + row * N + col_start
    offsets = tl.arange(0, 64)
    indices = base_idx + offsets

    #to address per-tile bitmap and counts. 
    flat_tile_index = batch_id * (stride_batch // 64) + tile_id
    
    bitmap = tl.load(bitmaps_ptr + flat_tile_index)
    # Get the base offset for this batch (because accum_count has num_tiles_per_batch + 1 per batch. )
    #idx = tl.load(counts_ptr + flat_tile_index)
    batch_base_idx = batch_id * ((stride_batch // 64) + 1)  # +1 for the extra count per batch
    idx = tl.load(counts_ptr + batch_base_idx + tile_id)
    
    #idx_plus_one = tl.load(counts_ptr + flat_tile_index + 1)
    #cnt = (idx_plus_one - idx) * 2  # number of float16s to store for this tile

    #valid = indices < total_elems
    #vals = tl.load(input_ptr + indices, mask=valid, other=0.0)
    vals = tl.load(input_ptr + indices)
    
    #extract the non-zero lanes. 
    shifted = bitmap >> (63 - offsets)
    bit_mask = (shifted & 1) #.to(tl.int32)

    #slot index. 
    prefix = tl.cumsum(bit_mask, axis=0) - 1
    valid_pos = bit_mask.to(tl.int1)

    batch_base_offset = tl.load(batch_offsets_ptr + batch_id)
    #store_idx = idx * 2 + prefix
    store_idx = batch_base_offset + idx * 2 + prefix
    tl.store(packed_not_ptr + store_idx, vals, mask=valid_pos)

def convert_key_batched(inputs: torch.Tensor):
    B, M, N = inputs.shape
    assert inputs.is_cuda
    assert inputs.dim() == 3
    assert M % 64 == 0

    inputs_t = inputs.transpose(1, 2).contiguous()  # [B, N, M]
    #input_flat = inputs[:, :, :].transpose(1, 2).contiguous().view(-1)
    
    num_tiles_per_batch = (M * N) // 64
    #total_tiles = B * num_tiles_per_batch

    bitmaps = torch.empty((B, num_tiles_per_batch), dtype=torch.int64, device=inputs.device)
    counts  = torch.empty((B, num_tiles_per_batch), dtype=torch.int32, device=inputs.device)

    # Precomputed shifts
    shift_amounts = np.arange(63, -1, -1, dtype=np.int64)
    shifts_np = np.left_shift(np.int64(1), shift_amounts)
    const_shifts = torch.tensor(shifts_np, device='cuda')

    #grid = (B, num_tiles_per_batch)
    grid = (num_tiles_per_batch, B) # flip grid to escape tigher limit on y dim. 
    stride_batch = num_tiles_per_batch * 64

    
    # Debug prints
    #print(f"Input shape: {inputs.shape}")
    #print(f"Inputs_t shape: {inputs_t.shape}")
    #print(f"Num tiles per batch: {num_tiles_per_batch}")
    #print(f"Stride batch: {stride_batch}")
    
    #print("GRID: ", grid)

    calculate_bitmap_key_batched[grid](
        #input_flat,
        inputs_t.view(-1)   ,
        bitmaps.view(-1),
        counts.view(-1),
        total_elems=B * M * N,
        shifts_ptr=const_shifts,
        stride_batch=stride_batch,
        M=M,
        N=N
    )

    accum_counts = torch.cumsum(counts, dim=1).to(torch.int32)  # [B, T]
    accum_counts = torch.cat([
        torch.zeros((B, 1), dtype=counts.dtype, device=counts.device),
        accum_counts
    ], dim=1).contiguous()  # [B, T+1]

    #print("accum_counts debugged access: ", accum_counts.view(-1)[66])

    total = 2 * accum_counts[:, -1]  # [B], float16 count per batch
    offsets = torch.cumsum(total, dim=0) #per batch access offset. 
    batch_offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=inputs.device), offsets[:-1]])
    #print("accum_counts shape: ", accum_counts.shape) #[B, 65]
    #print('total: ', total)
    #print("offsets: ", offsets)
    total_packed_size = offsets[-1].item()
    packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.float16, device=inputs.device) #non-zero padding directly applied here.

    #pass in tensor points as view(-1) for linear access, 
    compress_key_batched[grid](
        #input_flat,
        inputs_t.view(-1),
        bitmaps.view(-1),
        accum_counts.view(-1),
        packed_not_flat.view(-1),
        batch_offsets.view(-1),
        total_elems=B * M * N,
        stride_batch=stride_batch,
        M=M,
        N=N
    )
    #print("packed_not_flat: ", packed_not_flat)

    # Step 2: Slice `packed_not_flat` into per-batch tensors
    start_offsets = torch.zeros_like(offsets)
    start_offsets[1:] = offsets[:-1]
    #print("start_offsets: ", start_offsets)
    #return packed_not as a list of tensors, one per batch. 
    packed_not_batched = []
    for b in range(B):
        start = start_offsets[b].item()
        end = offsets[b].item()
        packed_not_batched.append(packed_not_flat[start:end].clone())

    #bitmaps and accoum_counts size is deterministic [B, num_tiles_per_batch]
    #packed_not_batched determined right above. 
    return bitmaps, accum_counts, packed_not_batched 
    
def convert_value_batched(inputs: torch.Tensor):
    B, M, N = inputs.shape
    assert inputs.is_cuda
    assert inputs.dim() == 3
    assert M % 64 == 0

    #inputs_t = inputs.transpose(1, 2).contiguous()  # [B, N, M]
    inputs_t = inputs.contiguous()
    #input_flat = inputs[:, :, :].transpose(1, 2).contiguous().view(-1)
    
    num_tiles_per_batch = (M * N) // 64
    #total_tiles = B * num_tiles_per_batch

    bitmaps = torch.empty((B, num_tiles_per_batch), dtype=torch.int64, device=inputs.device)
    counts  = torch.empty((B, num_tiles_per_batch), dtype=torch.int32, device=inputs.device)

    # Precomputed shifts
    shift_amounts = np.arange(63, -1, -1, dtype=np.int64)
    shifts_np = np.left_shift(np.int64(1), shift_amounts)
    const_shifts = torch.tensor(shifts_np, device='cuda')

    #grid = (B, num_tiles_per_batch)
    grid = (num_tiles_per_batch, B) # flip grid to escape tigher limit on y dim. 
    stride_batch = num_tiles_per_batch * 64

    
    # Debug prints
    #print(f"Input shape: {inputs.shape}")
    #print(f"Inputs_t shape: {inputs_t.shape}")
    #print(f"Num tiles per batch: {num_tiles_per_batch}")
    #print(f"Stride batch: {stride_batch}")
    
    #print("GRID: ", grid)

    calculate_bitmap_value_batched[grid](
        #input_flat,
        inputs_t.view(-1)   ,
        bitmaps.view(-1),
        counts.view(-1),
        total_elems=B * M * N,
        shifts_ptr=const_shifts,
        stride_batch=stride_batch,
        M=M,
        N=N
    )

    accum_counts = torch.cumsum(counts, dim=1).to(torch.int32)  # [B, T]
    accum_counts = torch.cat([
        torch.zeros((B, 1), dtype=counts.dtype, device=counts.device),
        accum_counts
    ], dim=1).contiguous()  # [B, T+1]

    #print("accum_counts debugged access: ", accum_counts.view(-1)[66])

    total = 2 * accum_counts[:, -1]  # [B], float16 count per batch
    offsets = torch.cumsum(total, dim=0) #per batch access offset. 
    batch_offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=inputs.device), offsets[:-1]])
    #print("accum_counts shape: ", accum_counts.shape) #[B, 65]
    #print('total: ', total)
    #print("offsets: ", offsets)
    total_packed_size = offsets[-1].item()
    packed_not_flat = torch.zeros((total_packed_size,), dtype=torch.float16, device=inputs.device) #non-zero padding directly applied here.

    #pass in tensor points as view(-1) for linear access, 
    compress_value_batched[grid](
        #input_flat,
        inputs_t.view(-1),
        bitmaps.view(-1),
        accum_counts.view(-1),
        packed_not_flat.view(-1),
        batch_offsets.view(-1),
        total_elems=B * M * N,
        stride_batch=stride_batch,
        M=M,
        N=N
    )
    #print("packed_not_flat: ", packed_not_flat)

    # Step 2: Slice `packed_not_flat` into per-batch tensors
    start_offsets = torch.zeros_like(offsets)
    start_offsets[1:] = offsets[:-1]
    #print("start_offsets: ", start_offsets)
    #return packed_not as a list of tensors, one per batch. 
    packed_not_batched = []
    for b in range(B):
        start = start_offsets[b].item()
        end = offsets[b].item()
        packed_not_batched.append(packed_not_flat[start:end].clone())

    #bitmaps and accoum_counts size is deterministic [B, num_tiles_per_batch]
    #packed_not_batched determined right above. 
    return bitmaps, accum_counts, packed_not_batched 
    