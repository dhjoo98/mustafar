import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

#from quant.new_pack import triton_quantize_and_pack_along_last_dim
#from quant.matmul import cuda_bmm_fA_qB_outer

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

_CONFIG_FOR_DOC = "LlamaConfig"

DEBUG = 0

class LlamaAttention_MUSTAFAR(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
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
        #self.k_bits = config.k_bits
        #self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length
        #assert getattr(config, "use_flash", False), "currently KIVI is only available for flash-attn. Please add ```config.use_flash = True```"
        if DEBUG: print("----LlamaAttention_MUSTAFAR initialized with residual length: ", self.residual_length)
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
        self.key_score_accumulator = None
        self.value_score_accumulator = None
        self.generation_count = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    #def dh_prune_key(self, key_states: torch.Tensor, target_sparsity=0.7):
    #def dh_prune_key(self, key_states: torch.Tensor, target_sparsity=0.5):
    #def dh_prune_key(self, key_states: torch.Tensor, target_sparsity=None):
    
    
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
        if DEBUG: print("NUM TO KEEP for Key: ", num_to_keep)

        # Flatten along batch, head, and tokens, keeping only the hidden_dim axis separate
        key_states_flat = key_states.reshape(-1, D)  # Shape: [(B * H * T), D]

        # Compute pruning threshold per vector
        threshold_values, _ = torch.kthvalue(torch.abs(key_states_flat), num_to_keep, dim=-1, keepdim=True)

        # Create a mask: Keep only values larger than or equal to the threshold
        mask = torch.abs(key_states_flat) >= threshold_values

        # Apply the mask (zero out pruned elements)
        pruned_key_states = key_states_flat * mask

        if DEBUG: 
                print("Debug: -- sparsity of just pruned key: ", self.calculate_sparsity(pruned_key_states.view(B, H, T, D)))

        # Reshape back to original dimensions
        return pruned_key_states.view(B, H, T, D)
        
        #return key[:, :, -self.residual_length:, :].contiguous()
        
    
    #def dh_prune_value(self, value_states: torch.Tensor, target_sparsity = 0.9, group_size = 128):
    #def dh_prune_value(self, value_states: torch.Tensor, target_sparsity = 0.3, group_size = 128):
    #pass in: iteration number, window of attention score computed, value_state, score accumulator, 
        #note attention_score is [B, H, 1, seq_len]
    #output: pruned_flag, pruned_value_states
    def dh_prune_value(self, iteration_num, attention_score:torch.Tensor, value_states: torch.Tensor, score_accumulator:torch.Tensor, target_sparsity = None, group_size = None):
        if target_sparsity is None:
            target_sparsity = self.v_sparsity
        if group_size is None:
            group_size = self.group_size
        """
        Performs magnitude-based pruning along the tokens dimension in groups of 32 tokens.

        Args:
            value_states (torch.Tensor): Tensor of shape [batch_size, num_heads, tokens, hidden_dim].
            target_sparsity (float): Fraction of elements to prune per group of tokens (between 0 and 1).
            group_size (int): The number of tokens in each pruning group (default: 32).

        Returns:
            torch.Tensor: Pruned tensor with the same shape, with values pruned to zero.
        """
        assert 0 <= target_sparsity <= 1, "Target sparsity must be between 0 and 1"

        # Get the shape of value_states
        B, H, T, D = value_states.shape  # Batch size, number of heads, tokens, hidden dimension
        B_, H_, T_, S = attention_score.shape
        if DEBUG: print("Debug: GQA: attention score shape: ", attention_score.shape) #[B, H, 1, T] H = 32
        if DEBUG: print("Debug: GQA: value states shape: ", value_states.shape) #[B, G, T, D] G = 8
        if DEBUG: print("Debug: GQA: num_heads: ", self.num_heads)
        if DEBUG: print("Debug: num_key_value_groups: ", self.num_key_value_groups) #so THIS IS GROUP SIZE. = 4
       


        #for GQA.
        #[05/01]for GQA.
        #later add if statement based on self.num_key_value_groups. 
        if iteration_num == 0:
            #assert T_ == 1, "Attention score must be 1 token"
            assert T_ == self.group_size, "Attention score must be group_size tokens"
            assert H == self.num_heads // self.num_key_value_groups, "GQA: number of key value groups must match"
            assert T == S, "GQA: number of tokens must match"
            #print("Debug: num_key_value_groups: ", self.num_key_value_groups)
            #Formulation-wise, same query attends to groups of KVs.
            if DEBUG: print("Debug: GQA: query dimension(attention weight dim):", attention_score.shape)
            if DEBUG: print("Debug: GQA: value dimension:", value_states.shape)

            #summation of attention score eltwise per KV head. 
            # attn_scores: [B, H, 1, S]
            # We want to sum over every group of `group_size` query heads
            #group_size = self.num_key_value_groups 
            #print("Debug: GQA: attention score shape: ", attention_score.shape) #[B, H, 1, T]
            #print("Debug: GQA: attention score: ", attention_score)
            attn_scores = torch.abs(attention_score).sum(dim=-2) # [B, H, 1, S] -> [B, H, S]
            #attn_scores = torch.abs(attention_score.squeeze(2))  # [B, H, S]
            attn_scores_grouped = attn_scores.view(B, H, self.num_key_value_groups, S)  # [B, G, group_size, S]

            # Sum over query heads in the group → [B, G, S]
            last_attention = attn_scores_grouped.sum(dim=-2, keepdim=True)  # [B, G, 1 , T]
            
            #[05_09] normalization inserted
            '''
            norm_factors = torch.arange(group_size, 0, -1, device=last_attention.device).float()  # [group_size]
            norm_factors = norm_factors.view(1, 1, group_size, 1)  # reshape for broadcasting

            # 6. Normalize the relevance score
            normalized_score = last_attention / norm_factors  # [B, G, 1, group_size]
            
            #compute score for the latest group_size tokens.
            last_attention = normalized_score.transpose(2, 3)  # [B, G, group_size, 1]
            #end of normalization
            '''
            last_attention.transpose_(2, 3) # [B, H, T, 1] 
            
            relevant_score_broadcast = last_attention.expand(-1, -1, -1, D)  # [B, H, token_length, D]
            score = torch.abs(relevant_score_broadcast * value_states)  # [B, H, token_length, D]
            pruned = True
            #find the pruning mask, 
            sort_res = torch.sort(score, dim=-1, descending=True)  # [B, H, token_length, D]
            # Create mask of same shape
            mask = torch.zeros_like(value_states, dtype=torch.bool)  # [B, H, token_length, D]
            
            # Get indices of top (1-sparsity) tokens for each feature
            #indices = sort_res[1][:, :, :int(group_size * (1 - target_sparsity)), :]  # Keep top (1-sparsity) tokens
            #indices = sort_res[1][:, :, :int(T * (1 - target_sparsity)), :]  # Keep top (1-sparsity) tokens
            # Scatter into mask along group_size dimension
            #mask.scatter_(-2, indices, True)
            indices = sort_res[1][:, :, :, :int(D * (1 - target_sparsity))]  # Keep top (1-sparsity) tokens
            # Scatter into mask along group_size dimension
            mask.scatter_(-1, indices, True)

            # Apply mask to value states
            pruned_value_states = value_states * mask  # [B, H, group_size, D]
            
            #[0509_sliding window with prefill]
            pruned_value_states[:, :, -self.group_size:, :] = value_states[:, :, -self.group_size:, :]
            #score_accumulator
            score_accumulator[:, :, -self.group_size:, :] = score[:, :, -self.group_size:, :] 
            if DEBUG: print("Debug: -- sparsity of just pruned value: ", self.calculate_sparsity(pruned_value_states), "with T: ", T)
            return pruned, pruned_value_states
            
        
        else: #decode
            #[04/24] Wanda: 
                #token window is 2 x group_size: score is computed for the latest group_size tokens, but pruning is triggered for the oldest group_size tokens. 
                #Pruned triggered only when iteration number % group_size == 0
            pruned = False
            if DEBUG: print("Debug:---- DH_prune_value during Decode, generation count: ", iteration_num)
            '''
            # Ensure token dimension is a multiple of group_size (for simplicity)
            if T % group_size != 0:
                raise ValueError("Token dimension must be a multiple of group_size")
                #pad_size = group_size - (T % group_size)
                #value_states = torch.nn.functional.pad(value_states, (0, 0, 0, pad_size))  # Pad tokens dimension
                #T = value_states.shape[2]  # Update token count after padding
            else:
                pad_size = 0
            '''
            #num_groups = T // group_size  # Number of token groups

            # Reshape so that each group of 32 tokens is processed together
            #value_states_grouped = value_states.view(B, H, num_groups, group_size, D)  # Shape: (B, H, num_groups, group_size, D)
            #value_states_window = value_states[:, :, -2 * group_size:, :]
            #value_states_score_compute = value_states[:, :, -group_size:, :] 
            value_states_window = value_states[:, :, -(group_size+1):-group_size, :]
            value_states_score_compute = value_states[:, :, -group_size:, :] 
            #L2
            #relevant_score = attention_score[:, :, :, -group_size:] ** 2
            #L1
            relevant_score = torch.abs(attention_score[:, :, :, -group_size:]) # [B, H, 1, group_size]
            if DEBUG: print("Debug: GQA: relevant_score shape: ", relevant_score.shape)

            relevant_score = torch.abs(relevant_score.squeeze(2))  # [B, H, S]
            relevant_score = relevant_score.view(B, H, self.num_key_value_groups, group_size)  # [B, G, group_size, S]
            
            relevant_score = relevant_score.sum(dim=-2, keepdim=True)  # [B, G, 1 , T]

            relevant_score = relevant_score.transpose(2, 3)  # [B, H, group_size, 1]
            relevant_score_broadcast = relevant_score.expand(-1, -1, -1, D)  # [B, H, group_size, D]
            score_accumulator[:, :, -group_size:, :] += torch.abs(relevant_score_broadcast * value_states_score_compute)  # [B, H, group_size, D]


            '''
            #group_size = self.num_key_value_groups 
            relevant_score = torch.abs(relevant_score.squeeze(2))  # [B, H, S]
            relevant_score = relevant_score.view(B, H, self.num_key_value_groups, group_size)  # [B, G, group_size, S]
            
            # Sum over query heads in the group → [B, G, S]
            relevant_score = relevant_score.sum(dim=-2, keepdim=True)  # [B, G, 1 , T]

            #compute score for the latest group_size tokens.
            relevant_score = relevant_score.transpose(2, 3)  # [B, H, group_size, 1]
            relevant_score_broadcast = relevant_score.expand(-1, -1, -1, D)  # [B, H, group_size, D]
            score_accumulator[:, :, -group_size:, :] += torch.abs(relevant_score_broadcast * value_states_score_compute)  # [B, H, group_size, D]
            '''

            #[0509_sliding window with prefill]
            '''
            #if iteration_num % (2 * group_size ) == 0:
            if iteration_num == group_size:
                pruned_value_states = value_states_window
            #elif iteration_num % (group_size) == 0:
            else:
            '''
            #unindented
            pruned = True
            #find the pruning mask, 
            #sort_res = torch.sort((score_accumulator[:, :, -2*group_size:-group_size, :].squeeze(2) / group_size), dim=-2, descending=True)  # [B, H, group_size, D]
            # Create mask of same shape
            #mask = torch.zeros_like(score_accumulator[:, :, -2*group_size:-group_size, :], dtype=torch.bool)  # [B, H, group_size, D]
            sort_res = torch.sort((score_accumulator[:, :, 0:1, :] / group_size), dim=-1, descending=True)  # [B, H, 1 token, D]
            # Create mask of same shape
            #mask = torch.zeros_like(score_accumulator[:, :, -2*group_size:-group_size, :], dtype=torch.bool)  # [B, H, group_size, D]
            mask = torch.zeros_like(score_accumulator[:, :, 0:1, :], dtype=torch.bool)  # [B, H, group_size, D]

            # Get indices of top (1-sparsity) tokens for each feature
            #indices = sort_res[1][:, :, :int(group_size * (1 - target_sparsity)), :]  # Keep top (1-sparsity) tokens
            indices = sort_res[1][:, :, :, :int(D * (1 - target_sparsity))]  # Keep top (1-sparsity) tokens

            # Scatter into mask along group_size dimension
            mask.scatter_(-1, indices, True)
            
            # Apply mask to value states
            #pruned_value_states = value_states_window * mask  # [B, H, group_size, D]
            #pruned_value_states = torch.cat((value_states_window[:, :, :-group_size, :] * mask, value_states_score_compute), dim=2)
            #pruned_value_states = value_states_window[:, :, :-group_size, :] * mask  # [B, H, group_size, D]
            pruned_value_states = value_states_window * mask  # [B, H, group_size, D]
            #value_states_full = torch.cat((self.dh_prune_value(value_states_prune).contiguous(), value_states_full), dim=2)
            if DEBUG: print("Debug: -- sparsity of just pruned value: ", self.calculate_sparsity(pruned_value_states))
            #else:
            #    pruned_value_states = value_states_window
            
            #shift score (only in decode.)
            score_accumulator[:, :, :-1, :] = score_accumulator[:, :, 1:, :]
            score_accumulator[:, :, -1, :] = 0

            return pruned, pruned_value_states

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
        
        if DEBUG: print(f"Matrix size: {tensor.shape}, Zero elements: {zero_elements}, Total elements: {total_elements}")
        sparsity = zero_elements / total_elements  # Compute sparsity ratio
        return sparsity
        
    #note than this forward method is not used for MUSTAFAR.
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

        #[04/24] Wanda: 
        if self.generation_count is None:
            self.generation_count = 0
        

            #initialize score_accumulator
        key_score_accumulator = self.key_score_accumulator
        value_score_accumulator = self.value_score_accumulator
        if key_score_accumulator is None:
            key_score_accumulator = torch.zeros([bsz, self.num_heads, self.group_size, self.head_dim], dtype=torch.float32, device='cuda') #[B, H, self.group_size, D]
        if value_score_accumulator is None:
            value_score_accumulator = torch.zeros([bsz, self.num_heads, 2 * self.group_size, self.head_dim], dtype=torch.float32, device='cuda') #[B, H, self.group_size, D]

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
            print("Debug: Hidden states shape: ", hidden_states.shape)
            query_states = self.q_proj(hidden_states) #(bsz, seq_length-> set to 1000, hidden_size = 4096)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            if DEBUG: print("---------key shape:   ", key_states.shape) # so this is always initialized as MAX supported seqlength KV cache size. 
                # [bsz, nh, t, hd]
                #which is exactly why Flash Attention is used. 
            if DEBUG: print("---------value shape: ", value_states.shape)
            #print("-----------Current Position ID: ", position_ids)


        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        #Control Branch 1: During Decode, with past_key_value.
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        if DEBUG: print(f"kv_seq_len before update: {key_states.shape[-2]}")
        if DEBUG: print(f"kv_seq_len after update: {kv_seq_len}")

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        #print("Debug: self.num_key_value_groups: ", self.num_key_value_groups)
        assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            self.generation_count += 1
            #key_states_quant_trans = past_key_value[0]
            #key_states_full = past_key_value[1]
            #key_scale_trans = past_key_value[2]
            #key_mn_trans = past_key_value[3]
            #value_states_quant = past_key_value[4]
            #value_states_full = past_key_value[5]
            #value_scale = past_key_value[6]
            #value_mn = past_key_value[7]

            #past_key_value: (key_states, value_states)
            key_states_full = past_key_value[0]
            assert key_states_full is not None
            value_states_full = past_key_value[1]
            #print("Debug: the full past_key_value: ", past_key_value)
            assert value_states_full is not None
            if DEBUG: print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOODebug: Decode current position id: ", position_ids.item())  #for [bsz, nh, t, hd]
            if DEBUG: print("************************* Previous Keys shape: ", key_states_full.shape)
            if DEBUG: print("************************* Previous Values shape: ", value_states_full.shape)
            #MUSTAFAR control flow:
                    #concatenating no longer needed. 
                #[1]compute attention weight 
                #[2]prune key with each token 
                #[3]prune value once residual length is reached
                    #[4]update the last 'residual_length' number of value_states.  

            #[1]compute attention weight 
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            if DEBUG: print("***************************** Query State shape: ", query_states.shape)
            if DEBUG: print("***************************** Key State shape: ", key_states_full.shape)

            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            attn_weights = att_qkfull / math.sqrt(self.head_dim)


            '''
            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                                key_scale_trans, key_mn_trans, self.k_bits)
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)
            '''

            # [2]prune key with each token 
                # Code inherited from KIVI value quantization

            #[04/25] Wanda: 
                # Key states: Compressed + window [Group size] + score [same dim, t_dim groupsize]

            if DEBUG: print("---------key_states shape before entering dh_prune_key: ", key_states.shape)
            #key_states_new = self.dh_prune_key(key_states).contiguous()
            #print("Debug:---- key_states_before concat: ", key_states_full.shape)
            #key_states_full = torch.cat([key_states_full[:, :, :-1, :], key_states_new], dim=2)
            #print("Debug:---- key_states_after concat: ", key_states_full.shape) 

            #remember, this is during decode. 
                #retain the most recent 'residual_length tokens' as dense. 
            #if the full length is smaller than residual_length, nothing happens.
            if (key_states_full.shape[-2] / self.residual_length) == 0:
                pass
            else:
                #take the first token of 'residual group' and prune it. 
                #so compressed_key + newly_pruned + 127 residual group ones + new cache. 
                
                # Get all states except the one we want to modify
                prefix = key_states_full[:, :, :-(self.residual_length+1), :]
                suffix = key_states_full[:, :, -(self.residual_length):, :]
                
                # Reshape pruned state to match original dimensions
                #key_states_new = self.dh_prune_key(key_states_full[:, :, -(self.residual_length+1), :]).contiguous()
                key_states_new = self.dh_prune_key(key_states_full[:, :, -(self.residual_length+1):(-(self.residual_length+1)+1), :]).contiguous()
                
                
                # Concatenate back together with pruned state in the middle
                key_states_full = torch.cat([prefix, key_states_new, suffix], dim=2)
                
                if DEBUG: print("Debug:---- key_states_after modification: ", key_states_full.shape)

            '''
            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new
            '''
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
            

            #[3]prune value once residual length is reached
                    #[4]update the last 'residual_length' number of value_states.  
            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            #value_full_length = value_states_full.shape[-2]
                #computation is done before prune
            if DEBUG: print("Debug:---- shape of attn_weight and val_stat_full: ", attn_weights.shape ,value_states_full.shape)  
            #Debug:---- shape of attn_weight and val_stat_full:  torch.Size([1, 32, 1, 67]) torch.Size([1, 32, 67, 128])
            attn_output = torch.matmul(attn_weights, value_states_full) #[B, H, 1, seq_len]
            '''
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, value_states_full)
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full)
            '''
            #the actual pruning, inherited from KIVI_key_quantization 
            ''' #This is the previous Decode Value Pruning code, with respect to the residual_length. 
            if (value_states_full.shape[-2] % self.residual_length) == 0:
                assert self.residual_length % self.group_size == 0
                value_states_pruned = self.dh_prune_value(value_states_full[:,:,-32:,:]).contiguous() #pass in the last residual_length number of vectors. 
                value_states_full = torch.cat((value_states_full[:,:,:-32,:], value_states_pruned), dim=2)
            '''

            #if (value_states_full.shape[-2] % self.group_size) == 0:
            #this should be rather, if (generated tokens % group_size) == 0
                #no, calculate score every iteration: Pruning is decided inside the function. 
                    #calculate score for the latest 32 tokens, shift score, prune when condition met. 
                #assert self.residual_length % self.group_size == 0
                #pass in: 
            #value_prune_iteration = generated_tokens
            pruned_flag, value_states_pruned = self.dh_prune_value(self.generation_count, attn_weights, value_states_full, value_score_accumulator)#.contiguous() #pass in the last residual_length number of vectors. 
            if pruned_flag:
                if DEBUG: print("Debug:---- During Decode, Value Cache pruned!: ")
                value_states_full = torch.cat((value_states_full[:,:,:-2*self.group_size,:], value_states_pruned, value_states_full[:,:,-self.group_size:,:]), dim=2)
            else: 
                if DEBUG: print("Debug:---- During Decode, Value Cache not pruned!: ")
                pass #do nothing. 

            '''
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn
            '''
        else: #MUSTAFAR control flow for when there is no past KVs:(Thus prefill)
                    #thus there won't be any concatenation.
                #[1]compute attention weight 
                #[2]prune key with each token 
                #[3]prune value once residual length is reached
                    #[4]update the last 'residual_length' number of value_states.  
            if DEBUG: print("---------------------Entering Prefill------------------------")
            self.generation_count = 0
            # [1] compute attention weight
            attn_weights = torch.matmul(query_states, 
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # [2] prune key with each token (inherit from KIVI value quantization)
            #print("Debug:---- shape of key states: ", key_states.shape)  #[bsz, nh, t, hd]
            #key_states_full = self.dh_prune_key(key_states)
            #[04/24] Wanda: 
            #if DEBUG: print("************************* Debug: Entering dh_prune_key for prefill. Genereation Count: ", self.generation_count)
            #key_states_full = self.dh_prune_key(self.generation_count, key_states, query_states, key_score_accumulator) #generatation count also used as a prefill trigger.
            #key_states_full = self.dh_prune_key(key_states) #generatation count also used as a prefill trigger.
            if (key_states.shape[-2] / self.residual_length) == 0:
                key_states_full = self.dh_prune_key(key_states)
            else:
                prefix = key_states[:, :, :-(self.residual_length), :]
                suffix = key_states[:, :, -(self.residual_length):, :]
                prefix_pruned = self.dh_prune_key(prefix).contiguous()
                key_states_full = torch.cat([prefix_pruned, suffix], dim=2) 

            if DEBUG: print("Debug:---- Key Prune complete")

            # [3] prune value for the repetitions of 'residual_length' num of vectors. [inherit from KIVI key quantization]
            
            #####################NOPE########This isn't even necessary at this point, for now, just prune the (seq_len x hd) sized matix by the seq_len dimension. 
            if DEBUG: print("Debug:---- shape of FULL value states: ", value_states.shape)  #[bsz, nh, t, hd]
            
            '''#Old code for Prefill Value Pruning
            #value_states_full = self.dh_prune_value(value_states)
            if value_states.shape[-2] % self.residual_length != 0:
                if value_states.shape[-2] < self.residual_length:
                    value_states_prune = None
                    value_states_full = value_states
                else:
                    value_states_prune = value_states[:, :, :-(value_states.shape[-2] % self.residual_length), :].contiguous()
                    value_states_full = value_states[:, :, -(value_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                value_states_prune = value_states
                value_states_full = None
            
            if value_states_prune is not None:
                if DEBUG: print("Debug:---- shape of value states TO PRUNE: ", value_states_prune.shape)  #[bsz, nh, t, hd]
                if value_states_full is not None:
                    if DEBUG: cat_1 = self.dh_prune_value(value_states_prune).contiguous()
                    if DEBUG: cat_2 = value_states_full
                    if DEBUG: print("XXXXXXXXXXXXXXX cat 1 shape: ", cat_1.shape)
                    if DEBUG: print("XXXXXXXXXXXXXXX cat 2 shape: ", cat_2.shape)
                    value_states_full = torch.cat((self.dh_prune_value(value_states_prune).contiguous(), value_states_full), dim=2)
                    if DEBUG: print("Debug:---- Value Prune complete")
                else:
                    value_states_full = self.dh_prune_value(value_states_prune)
                     #key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            #else:
            #    key_states_quant_trans = None
            #    key_scale_trans = None
            #    key_mn_trans = None
            '''
            #[04/24] Wanda: 
            '''
            #value_states_full = self.dh_prune_value(value_states)
            #if value_states.shape[-2] % self.residual_length != 0:
            if value_states.shape[-2] % self.group_size != 0:
                #if value_states.shape[-2] < self.residual_length:
                if value_states.shape[-2] < self.group_size:
                    value_states_prune = None
                    value_states_full = value_states
                else:
                    #value_states_prune = value_states[:, :, :-(value_states.shape[-2] % self.residual_length), :].contiguous()
                    #value_states_full = value_states[:, :, -(value_states.shape[-2] % self.residual_length):, :].contiguous()
                    value_states_prune = value_states[:, :, :-(value_states.shape[-2] % self.group_size), :].contiguous()
                    value_states_full = value_states[:, :, -(value_states.shape[-2] % self.group_size):, :].contiguous()
            else:
                value_states_prune = value_states
                value_states_full = None
            '''
            #the problem here is the dh_prune_value must operate on group_size sized tensors. -> Rather, let's change the dh_prune_value to operate on group_size sized tensors. 
            #oh, that has already been implemented on the dh_prune_value function.  
            
            #[04/24] Wanda: Wait, in prefill, why is value pruned before attention computation? 
            #now, moved post-attention computation. 
            '''
            if value_states_prune is not None:
                if DEBUG: print("Debug:---- shape of value states TO PRUNE: ", value_states_prune.shape)  #[bsz, nh, t, hd]
                if value_states_full is not None:
                    if DEBUG: cat_1 = self.dh_prune_value(value_states_prune).contiguous()
                    if DEBUG: cat_2 = value_states_full
                    if DEBUG: print("XXXXXXXXXXXXXXX cat 1 shape: ", cat_1.shape)
                    if DEBUG: print("XXXXXXXXXXXXXXX cat 2 shape: ", cat_2.shape)
                    value_states_full = torch.cat((self.dh_prune_value(value_states_prune).contiguous(), value_states_full), dim=2)
                    if DEBUG: print("Debug:---- Value Prune complete")
                else:
                    value_states_full = self.dh_prune_value(value_states_prune)
            '''
            '''
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
            '''

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
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states) 
            #[04/24] Wanda: 

            _, value_states_full = self.dh_prune_value(self.generation_count, attn_weights, value_states, value_score_accumulator)


            #This is the end of the prefill. 

        #[Final] update past key value states
        #past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
        past_key_value = (key_states_full, value_states_full, kv_seq_len) 
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        if DEBUG: print("_______Exported K, V shape: ",  key_states_full.shape,  value_states_full.shape,  "   ________________________________")
        if DEBUG: print("_______KV sparsity: ",  self.calculate_sparsity(key_states_full),  self.calculate_sparsity(value_states_full),  "   ________________________________")

        return attn_output, attn_weights, past_key_value

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


        key_score_accumulator = self.key_score_accumulator
        value_score_accumulator = self.value_score_accumulator
        #if key_score_accumulator is None:
            #key_score_accumulator = torch.zeros([bsz, self.num_heads, self.group_size, self.head_dim], dtype=torch.float32, device='cuda') #[B, H, self.group_size, D]
        #    key_score_accumulator = torch.zeros([bsz, self.num_heads // self.num_key_value_groups, self.group_size, self.head_dim], dtype=torch.float32, device='cuda') #[B, G, self.group_size, D]
        if value_score_accumulator is None:
            #value_score_accumulator = torch.zeros([bsz, self.num_heads, 2 * self.group_size, self.head_dim], dtype=torch.float32, device='cuda') #[B, H, self.group_size, D]
            #value_score_accumulator = torch.zeros([bsz, self.num_heads // self.num_key_value_groups, 2 * self.group_size, self.head_dim], dtype=torch.float32, device='cuda') #[B, G, self.group_size, D]
            value_score_accumulator = torch.zeros([bsz, self.num_heads // self.num_key_value_groups, self.group_size + 1, self.head_dim], dtype=torch.float16, device='cuda') #[B, H, self.group_size, D]
    

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
            if DEBUG: print("-------------------------------------------------------------Begin of a Prefill/Decode Stage")
            if DEBUG: print("Debug: Hidden states shape: ", hidden_states.shape)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            self.generation_count += 1
            #key_states_quant_trans = past_key_value[0]
            #key_states_full = past_key_value[1]
            #key_scale_trans = past_key_value[2]
            #key_mn_trans = past_key_value[3]
            #value_states_quant = past_key_value[4]
            #value_states_full = past_key_value[5]
            #value_scale = past_key_value[6]
            #value_mn = past_key_value[7]

            #past_key_value: (key_states, value_states)
            key_states_full = past_key_value[0]
            assert key_states_full is not None
            value_states_full = past_key_value[1]
            #print("Debug: the full past_key_value: ", past_key_value)
            assert value_states_full is not None
            if DEBUG: print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOODebug: Decode current position id: ", position_ids)  #for [bsz, nh, t, hd]
            if DEBUG: print("************************* Previous Keys shape: ", key_states_full.shape)
            if DEBUG: print("************************* Previous Values shape: ", value_states_full.shape)
            #MUSTAFAR control flow:
                    #concatenating no longer needed. 
                #[1]compute attention weight 
                #[2]prune key with each token 
                #[3]prune value once residual length is reached
                    #[4]update the last 'residual_length' number of value_states.  

            #[1]compute attention weight 
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            if DEBUG: print("***************************** Query State shape: ", query_states.shape)
            if DEBUG: print("***************************** Key State shape: ", key_states_full.shape)

            #att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
            attn_weights = att_qkfull / math.sqrt(self.head_dim)


            '''
            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                                key_scale_trans, key_mn_trans, self.k_bits)
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)
            '''

            # [2]prune key with each token 
                # Code inherited from KIVI value quantization

            #[04/25] Wanda: 
                # Key states: Compressed + window [Group size] + score [same dim, t_dim groupsize]

            if DEBUG: print("---------key_states shape before entering dh_prune_key: ", key_states.shape)
            #key_states_new = self.dh_prune_key(key_states).contiguous()
            #print("Debug:---- key_states_before concat: ", key_states_full.shape)
            #key_states_full = torch.cat([key_states_full[:, :, :-1, :], key_states_new], dim=2)
            #print("Debug:---- key_states_after concat: ", key_states_full.shape) 

            #remember, this is during decode. 
                #retain the most recent 'residual_length tokens' as dense. 
            #if the full length is smaller than residual_length, nothing happens.
            if (key_states_full.shape[-2] / self.residual_length) == 0:
                pass
            else:
                #take the first token of 'residual group' and prune it. 
                #so compressed_key + newly_pruned + 127 residual group ones + new cache. 
                
                # Get all states except the one we want to modify
                prefix = key_states_full[:, :, :-(self.residual_length+1), :]
                suffix = key_states_full[:, :, -(self.residual_length):, :]
                
                # Reshape pruned state to match original dimensions
                #key_states_new = self.dh_prune_key(key_states_full[:, :, -(self.residual_length+1), :]).contiguous()
                key_states_new = self.dh_prune_key(key_states_full[:, :, -(self.residual_length+1):(-(self.residual_length+1)+1), :]).contiguous()
                
                
                # Concatenate back together with pruned state in the middle
                key_states_full = torch.cat([prefix, key_states_new, suffix], dim=2)
                
                if DEBUG: print("Debug:---- key_states_after modification: ", key_states_full.shape)

            '''
            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new
            '''
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
            

            #[3]prune value once residual length is reached
                    #[4]update the last 'residual_length' number of value_states.  
            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            #value_full_length = value_states_full.shape[-2]
                #computation is done before prune
            if DEBUG: print("Debug:---- shape of attn_weight and val_stat_full: ", attn_weights.shape ,value_states_full.shape)  
            #Debug:---- shape of attn_weight and val_stat_full:  torch.Size([1, 32, 1, 67]) torch.Size([1, 32, 67, 128])
            #attn_output = torch.matmul(attn_weights, value_states_full) #[B, H, 1, seq_len]
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups)) #[B, H, 1, seq_len]
            '''
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, value_states_full)
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full)
            '''
            #the actual pruning, inherited from KIVI_key_quantization 
            ''' #This is the previous Decode Value Pruning code, with respect to the residual_length. 
            if (value_states_full.shape[-2] % self.residual_length) == 0:
                assert self.residual_length % self.group_size == 0
                value_states_pruned = self.dh_prune_value(value_states_full[:,:,-32:,:]).contiguous() #pass in the last residual_length number of vectors. 
                value_states_full = torch.cat((value_states_full[:,:,:-32,:], value_states_pruned), dim=2)
            '''

            #if (value_states_full.shape[-2] % self.group_size) == 0:
            #this should be rather, if (generated tokens % group_size) == 0
                #no, calculate score every iteration: Pruning is decided inside the function. 
                    #calculate score for the latest 32 tokens, shift score, prune when condition met. 
                #assert self.residual_length % self.group_size == 0
                #pass in: 
            #value_prune_iteration = generated_tokens
            pruned_flag, value_states_pruned = self.dh_prune_value(self.generation_count, attn_weights, value_states_full, value_score_accumulator)#.contiguous() #pass in the last residual_length number of vectors. 
            if pruned_flag:
                if DEBUG: print("Debug:---- During Decode, Value Cache pruned!: ")
                #value_states_full = torch.cat((value_states_full[:,:,:-2*self.group_size,:], value_states_pruned, value_states_full[:,:,-self.group_size:,:]), dim=2)
                value_states_full = torch.cat((value_states_full[:,:,:-(self.group_size+1),:], value_states_pruned, value_states_full[:,:,-self.group_size:,:]), dim=2)

            else: 
                if DEBUG: print("Debug:---- During Decode, Value Cache not pruned!: ")
                pass #do nothing. 

        else:
            if DEBUG: print(f"Prefill, Using Flash Attention!")
            input_dtype = query_states.dtype
            self.generation_count = 0

            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2), 
                value_states.transpose(1, 2), None, q_len, dropout=0.0
            )

            if (key_states.shape[-2] / self.residual_length) == 0:
                key_states_full = self.dh_prune_key(key_states)
            else:
                prefix = key_states[:, :, :-(self.residual_length), :]
                suffix = key_states[:, :, -(self.residual_length):, :]
                prefix_pruned = self.dh_prune_key(prefix).contiguous()
                key_states_full = torch.cat([prefix_pruned, suffix], dim=2) 

            #[05/01] last 32 attention score for value cache pruning during decode.
            
            
            #query_states: [B, n_heads, q_len, d]
            #key_states_full: [B, n_kv_heads, k_len, d]
            #num_key_value_groups = n_heads // n_kv_heads
            #print("Debug: num_key_value_groups: ", self.num_key_value_groups)

            #for Now, let's just use the last attention score.
            #attn_weights = torch.matmul(query_states[:, :, -1:, :], repeat_kv(key_states_full, self.num_key_value_groups).transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, 32, kv_len]
            
            #well, if you think about it, even key_states here should not have been pruned.

            attn_weights = torch.matmul(query_states[:, :, -self.group_size:, :], repeat_kv(key_states_full, self.num_key_value_groups).transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, 32, kv_len]
            if DEBUG: print("Debug: attention_mask shape: ", attention_mask.shape) #[1, 1, T, T]
            if DEBUG: print("attention_mask: ", attention_mask)
            assert attention_mask is not None
            attn_mask = attention_mask[:, :, -self.group_size:, :]
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
            if DEBUG: print("Debug: attn_mask: ", attn_mask.shape)
            if DEBUG: print("Debug: attn_mask: ", attn_mask[0, 0, :, :])
            attn_weights = attn_weights + attn_mask
            attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            if DEBUG: print("Debug: attn_weights: ", attn_weights.shape)
            if DEBUG: print("Debug: attn_weights: ", attn_weights[0, 0, :, :])
            #the above code contains room for memory footprint optimization. 

            #softmax is needed here. 
            attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
            if DEBUG: print("Debug: attn_weights: ", attn_weights.shape)
            if DEBUG: print("Debug: attn_weights: ", attn_weights[0, 0, :, :])
            
            _, value_states_full = self.dh_prune_value(self.generation_count, attn_weights, value_states, value_score_accumulator)

        
        past_key_value = (key_states_full, value_states_full, kv_seq_len) 
        if DEBUG: print("Debug: past_key_value: ", past_key_value[0].shape, past_key_value[1].shape, past_key_value[2])
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        if DEBUG: print("Debug: End of a stage: attn_output shape: ", attn_output.shape)
        if DEBUG: print(f"Iteration: {self.generation_count}","KV sparsity: ",  self.calculate_sparsity(key_states_full),  self.calculate_sparsity(value_states_full),  "   ________________________________")

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
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention_MUSTAFAR(config=config)
            if not getattr(config, "use_flash", False)
            else LlamaFlashAttention_MUSTAFAR(config=config)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_MUSTAFAR(config) for _ in range(config.num_hidden_layers)])
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
            if DEBUG: print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Entering Layer: ', idx  )
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
 
            if DEBUG: print("Debug: End of an iteration: hidden_states shape: ", hidden_states.shape)

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
            if DEBUG: print("Debug: past_length: ", past_length)
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
                if DEBUG: print("Debug: remove_prefix_length: ", remove_prefix_length)
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if DEBUG: print("Debug: input_ids: ", input_ids)

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
