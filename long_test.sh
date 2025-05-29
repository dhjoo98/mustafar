# model e.g.: meta-llama/Llama-2-7b-hf

#gpuid=$1
k_sparsity=$1
v_sparsity=$2
group_size=32
model=$3
mode=$4
e=0

CUDA_VISIBLE_DEVICES=0 python ./pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_sparsity $k_sparsity \
    --v_sparsity $v_sparsity \
    --group_size $group_size \
    --residual_length $group_size \
    --mode $mode \
    --e ${e} 