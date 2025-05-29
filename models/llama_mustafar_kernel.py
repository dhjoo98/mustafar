import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import mustafar_package

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernel.compression as compression

import torch.nn.functional as F

import gc

import torch.cuda.nvtx as nvtx



#Note that only flashattention is supported for now. 


_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaAttention_MUSTAFAR(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, shared_reduction_workspace=None):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size  #4096
        self.num_heads = config.num_attention_heads #32
        self.head_dim = self.hidden_size // self.num_heads  #128
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_sparsity = config.k_sparsity
        self.v_sparsity = config.v_sparsity
        self.group_size = config.group_size
        self.residual_length = config.residual_length
        #assert getattr(config, "use_flash", False), "currently KIVI is only available for flash-attn. Please add ```config.use_flash = True```"
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)  #(4096 by 4096) 
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.Reduction_Workspace = shared_reduction_workspace
        #self.Reduction_Workspace = torch.zeros([self.num_heads, 8192, 8, 4], dtype=torch.float16, device='cuda') #[B, H, self.group_size, D]
        
        #self.key_score_accumulator = None
        #self.value_score_accumulator = None
        #self.generation_count = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    
    def dh_prune_key(self, key_states: torch.Tensor, target_sparsity=None):
        if target_sparsity is None:
            target_sparsity = self.k_sparsity
        """
        Performs magnitude-based pruning along the hidden dimension.
        
        Args:
            key_states (torch.Tensor): Tensor of shape [batch_size, num_heads, tokens, hidden_dim].
            target_sparsity (float): Fraction of elements to prune per vector (between 0 and 1).
            
        Returns:
            torch.Tensor: Pruned tensor with the same shape, with values pruned to zero.
        """
        assert 0 <= target_sparsity < 1, "Target sparsity must be between 0 and 1"

        # Get the shape of key_states
        B, H, T, D = key_states.shape  # Batch size, number of heads, tokens, hidden dimension

        # Compute the number of elements to keep per vector (hidden dimension)
        #num_to_keep = max(1, int((1 - target_sparsity) * D))
        num_to_keep = max(1, int((target_sparsity) * D))

        # Flatten along batch, head, and tokens, keeping only the hidden_dim axis separate
        key_states_flat = key_states.reshape(-1, D)  # Shape: [(B * H * T), D]

        # Compute pruning threshold per vector
        threshold_values, _ = torch.kthvalue(torch.abs(key_states_flat), num_to_keep, dim=-1, keepdim=True)
        

        # Create a mask: Keep only values larger than or equal to the threshold
        mask = torch.abs(key_states_flat) >= threshold_values

        # Apply the mask (zero out pruned elements)
        pruned_key_states = key_states_flat * mask

        # Reshape back to original dimensions
        return pruned_key_states.view(B, H, T, D)
        
        #return key[:, :, -self.residual_length:, :].contiguous()
        
    def dh_prune_value(self, key_states: torch.Tensor, target_sparsity=None):
        if target_sparsity is None:
            target_sparsity = self.v_sparsity
        """
        Performs magnitude-based pruning along the hidden dimension.
        
        Args:
            key_states (torch.Tensor): Tensor of shape [batch_size, num_heads, tokens, hidden_dim].
            target_sparsity (float): Fraction of elements to prune per vector (between 0 and 1).
            
        Returns:
            torch.Tensor: Pruned tensor with the same shape, with values pruned to zero.
        """
        assert 0 <= target_sparsity < 1, "Target sparsity must be between 0 and 1"

        # Get the shape of key_states
        B, H, T, D = key_states.shape  # Batch size, number of heads, tokens, hidden dimension

        # Compute the number of elements to keep per vector (hidden dimension)
        #num_to_keep = max(1, int((1 - target_sparsity) * D))
        num_to_keep = max(1, int((target_sparsity) * D))

        # Flatten along batch, head, and tokens, keeping only the hidden_dim axis separate
        key_states_flat = key_states.reshape(-1, D)  # Shape: [(B * H * T), D]

        # Compute pruning threshold per vector
        threshold_values, _ = torch.kthvalue(torch.abs(key_states_flat), num_to_keep, dim=-1, keepdim=True)
        
        # Create a mask: Keep only values larger than or equal to the threshold
        mask = torch.abs(key_states_flat) >= threshold_values

        # Apply the mask (zero out pruned elements)
        pruned_key_states = key_states_flat * mask

        
        # Reshape back to original dimensions
        return pruned_key_states.view(B, H, T, D)


    def calculate_sparsity(self, tensor: torch.Tensor) -> float:
        """
        Calculates the sparsity of a 4D PyTorch tensor.
        
        Sparsity is defined as the fraction of elements that are zero.
        
        Args:
            tensor (torch.Tensor): A 4D tensor (batch, channels, height, width)
        
        Returns:
            float: The sparsity ratio (between 0 and 1)
        """
        if tensor.dim() != 4:
            raise ValueError("Input tensor must be 4D (batch, channels, height, width)")
        
        total_elements = tensor.numel()  # Total number of elements
        zero_elements = torch.sum(tensor == 0).item()  # Count of zero elements
        
        sparsity = zero_elements / total_elements  # Compute sparsity ratio
        return sparsity
        
    #note than this forward method is not used for MUSTAFAR. (support flashattention only for now.)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        #deprecated. Currently only supports flash attention. 

        return torch.zeros_like(hidden_states)

class LlamaFlashAttention_MUSTAFAR(LlamaAttention_MUSTAFAR):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()


        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            #a, b, c = hidden_states.shape
            #if b < 10:
           #     hidden_states = hidden_states[:, -1, :]
            #hidden_states = hidden_states[:, -1, :]
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        model_dim = key_states.shape[-1] #head_dime.
        batch_size = key_states.shape[0] #support batched input
        total_batch_size = batch_size * self.num_heads
        total_batch_kv = batch_size * self.num_key_value_heads
        kv_seq_len = query_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len = past_key_value[-1] + 1 #for decode, increment seq_len by 1. 
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            self.generation_count += 1


            k_compressed = past_key_value[0]
            k_local_window = past_key_value[1]
            v_compressed = past_key_value[2]
            v_local_window = past_key_value[3]
            compressed_length = past_key_value[4]

        

            #[1]compute attention weight 
            #key_states = repeat_kv(key_states, self.num_key_value_groups)
            k_local_window = torch.cat([k_local_window, key_states], dim=2)

            if compressed_length != 0:
                padded_query = F.pad(query_states.view(total_batch_size, -1, self.head_dim), (0, 0, 0, 7), mode='constant', value=0)
                att_compressed = mustafar_package.mustafar_key_formulation(k_compressed[0], torch.cat(k_compressed[2]), k_compressed[1], k_compressed[3], padded_query, compressed_length, model_dim, total_batch_size, self.num_key_value_groups)
                att_compressed = att_compressed[:, 0:1, :].view(batch_size, self.num_heads, 1, compressed_length)
                
          
                att_local = torch.matmul(query_states, repeat_kv(k_local_window, self.num_key_value_groups).transpose(2, 3))
                att_qkfull = torch.cat([att_compressed, att_local], dim=-1) #concat along the seq_len dimension
            else:
                #att_qkfull = torch.matmul(query_states, k_local_window.transpose(2, 3))
                att_qkfull = torch.matmul(query_states, repeat_kv(k_local_window, self.num_key_value_groups).transpose(2, 3))

            attn_weights = att_qkfull / math.sqrt(self.head_dim)

    
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            

            
            #value_states = repeat_kv(value_states, self.num_key_value_groups)
            v_local_window = torch.cat([v_local_window, value_states], dim=2)
            

            if compressed_length != 0:
                padded_score = F.pad(attn_weights[:,:,:,:compressed_length].view(total_batch_size, -1, compressed_length), (0, 0, 0, 7)).contiguous()
                attn_output_compressed = mustafar_package.mustafar_value_formulation(v_compressed[0], torch.cat(v_compressed[2]), v_compressed[1], v_compressed[3], padded_score, self.Reduction_Workspace, model_dim, compressed_length, total_batch_size, self.num_key_value_groups)
                attn_output_compressed = attn_output_compressed[:, 0:1, :].view(batch_size, self.num_heads, 1, model_dim)
                attn_output_local = torch.matmul(attn_weights[:,:,:,compressed_length:], repeat_kv(v_local_window, self.num_key_value_groups))
                attn_output = attn_output_compressed + attn_output_local
            else:
                #attn_output = torch.matmul(attn_weights, v_local_window)
                attn_output = torch.matmul(attn_weights, repeat_kv(v_local_window, self.num_key_value_groups))
            
            
            
            if (kv_seq_len - self.residual_length - compressed_length) % 256 == 0:
                k_local_window[:, :, :256, :] = self.dh_prune_key(k_local_window[:, :, :256, :])
                v_local_window[:, :, :256, :] = self.dh_prune_value(v_local_window[:, :, :256, :])
                if compressed_length == 0:
                    k_bmps, k_idxs, k_nzs = compression.convert_key_batched(k_local_window[:, :, :256, :].reshape(total_batch_kv, -1, self.head_dim))
                    k_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=k_local_window.device)
                    for i in range(1, total_batch_kv):
                        k_nz_offset[i] = k_nz_offset[i-1] + k_idxs[i-1][-1] // 4
                    k_compressed = [k_bmps, k_idxs, k_nzs, k_nz_offset]
                    v_bmps, v_idxs, v_nzs = compression.convert_value_batched(v_local_window[:, :, :256, :].reshape(total_batch_kv, -1, self.head_dim))
                    v_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=v_local_window.device)
                    for i in range(1, total_batch_kv):
                        v_nz_offset[i] = v_nz_offset[i-1] + v_idxs[i-1][-1] // 4
                    v_compressed = [v_bmps, v_idxs, v_nzs, v_nz_offset]
                
                else:
                    k_new_bmps, k_new_idxs, k_new_nzs = compression.convert_key_batched(k_local_window[:, :, :256, :].reshape(total_batch_kv, -1, self.head_dim))
                    last_elements = torch.tensor([k[-1] for k in k_new_idxs[:-1]], device=k_compressed[3].device)
                    increments = last_elements // 4# Compute cumulative sum
                    cumsum = torch.cumsum(increments, dim=0)
                    k_compressed[3][1:] += cumsum
                    tiles_per_token = model_dim // 64
                    update_slice_idx = compressed_length * tiles_per_token #with the slice index of the new one being 256 * model_dim // 64
                    #update k_idxs
                        #parse into each batch's idx, 
                        #increment the new k_idxs by last element of old k_idxs. 
                        # concatenate new ones. 
                    # Precompute base offset (last index of old_idxs) for all heads
                    base = k_compressed[1].view(total_batch_kv, -1)[:, -1]  # shape [num_heads]
                    base = base.unsqueeze(1)  # shape [num_heads, 1] for broadcasting

                    # Slice and shift
                    old_part = k_compressed[1].view(total_batch_kv, -1)[:, :-1]
                    new_part = k_new_idxs.view(total_batch_kv, -1) + base

                    # Fuse and flatten
                    k_compressed[1] = torch.cat([old_part, new_part], dim=1).flatten()
                    
                    #update k_bmps
                        #parse into each batch's bmp, 
                    k_compressed[0] = torch.cat([k_compressed[0].view(total_batch_kv, update_slice_idx), k_new_bmps.view(total_batch_kv, 256*tiles_per_token)], dim=1).flatten()
                        #concatenate new ones. 
                    #update k_nzs [lists]
                        #concatenate new ones. 
                    k_compressed[2] = [torch.cat([k_compressed[2][b], k_new_nzs[b]], dim=0) for b in range(total_batch_kv)]


                    # Value cache compression update
                    v_new_bmps, v_new_idxs, v_new_nzs = compression.convert_value_batched(v_local_window[:, :, :256, :].reshape(total_batch_kv, -1, self.head_dim))
                    last_elements = torch.tensor([v[-1] for v in v_new_idxs[:-1]], device=v_compressed[3].device)
                    increments = last_elements // 4
                    cumsum = torch.cumsum(increments, dim=0)
                    # Add to the tail of v_compressed[3]
                    v_compressed[3][1:] += cumsum

                    base = v_compressed[1].view(total_batch_kv, -1)[:, -1]  # shape [num_heads]
                    base = base.unsqueeze(1)  # shape [num_heads, 1] for broadcasting

                    # Slice and shift
                    old_part = v_compressed[1].view(total_batch_kv, -1)[:, :-1]
                    new_part = v_new_idxs.view(total_batch_kv, -1) + base
                    # Fuse and flatten
                    v_compressed[1] = torch.cat([old_part, new_part], dim=1).flatten()
                    #update v_bmps
                        #parse into each batch's bmp, 
                    v_compressed[0] = torch.cat([v_compressed[0].view(total_batch_kv, update_slice_idx), v_new_bmps.view(total_batch_kv, 256*tiles_per_token)], dim=1).flatten()
                    v_compressed[2] = [torch.cat([v_compressed[2][b], v_new_nzs[b]], dim=0) for b in range(total_batch_kv)]

                k_local_window_new = k_local_window[:, :, 256:, :].clone().contiguous()
                v_local_window_new = v_local_window[:, :, 256:, :].clone().contiguous()
                del k_local_window, v_local_window
                torch.cuda.empty_cache()
                k_local_window = k_local_window_new
                v_local_window = v_local_window_new
                compressed_length = compressed_length + 256
            else:
                pass
            
            
           
            
        else:            
            
            input_dtype = query_states.dtype
            self.generation_count = 0

            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2), 
                value_states.transpose(1, 2), None, q_len, dropout=0.0
            )


            compressed_length = ((kv_seq_len - self.residual_length) // 256) * 256
            if compressed_length != 0:
                
                key_states[:, :, :compressed_length, :] = self.dh_prune_key(key_states[:, :, :compressed_length, :])
                value_states[:, :, :compressed_length, :] = self.dh_prune_value(value_states[:, :, :compressed_length, :])
                
                k_bmps, k_idxs, k_nzs = compression.convert_key_batched(key_states[:, :, :(compressed_length), :].reshape(total_batch_kv, -1, self.head_dim)) # support batching.
                k_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=key_states.device)
                for i in range(1, total_batch_kv):
                    k_nz_offset[i] = k_nz_offset[i-1] + k_idxs[i-1][-1] // 4
                k_compressed = [k_bmps, k_idxs, k_nzs, k_nz_offset]
                k_local_window = key_states[:, :, (compressed_length):, :].clone().contiguous()
                del key_states
                torch.cuda.empty_cache()
                v_bmps, v_idxs, v_nzs = compression.convert_value_batched(value_states[:, :, :(compressed_length), :].reshape(total_batch_kv, -1, self.head_dim)) # support batching.
                v_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=value_states.device)
                for i in range(1, total_batch_kv):
                    v_nz_offset[i] = v_nz_offset[i-1] + v_idxs[i-1][-1] // 4
                v_compressed = [v_bmps, v_idxs, v_nzs, v_nz_offset]
                v_local_window = value_states[:, :, (compressed_length):, :].clone().contiguous()
                del value_states
                torch.cuda.empty_cache()
            else:
                k_compressed = None
                k_local_window = key_states 
                v_compressed = None
                v_local_window = value_states

            
        past_key_value = (k_compressed, k_local_window, v_compressed, v_local_window, compressed_length, kv_seq_len)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None

        return attn_output, attn_weights, past_key_value

    #for calling flash_attention
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output


    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    

class LlamaDecoderLayer_MUSTAFAR(nn.Module):
    def __init__(self, config: LlamaConfig, shared_reduction_workspace=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            #LlamaAttention_MUSTAFAR(config=config)
            #if not getattr(config, "use_flash", False)
            #else LlamaFlashAttention_MUSTAFAR(config=config)
            LlamaFlashAttention_MUSTAFAR(config=config, shared_reduction_workspace=shared_reduction_workspace)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_flash == False:
            raise ValueError("Only Flash attention is supported for Llama3 for now.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_MUSTAFAR(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        '''
        self.shared_reduction_workspace = torch.zeros(
            [32, 8192, 8, 4], 
            dtype=torch.float16, 
            device='cuda'
        )
        '''
        '''
        self.shared_reduction_workspace = torch.zeros(
            [8192, 8, 4], 
            dtype=torch.float16, 
            device='cuda'
        )
        '''
        self.shared_reduction_workspace = torch.zeros(1, dtype=torch.float16, device='cuda')


        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_MUSTAFAR(config, shared_reduction_workspace=self.shared_reduction_workspace) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
 

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_MUSTAFAR(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_MUSTAFAR(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None
        if past_key_values is not None:
            #print("Debug: past_key_values: ", past_key_values)
            past_length = past_key_values[0][-1]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
