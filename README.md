# Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

Include High-level explanation 

Add images. 

<!-- 
![MUSTAFAR Architecture](images/mustafar_diagram.png)
-->

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Part I: Install Dependencies](#part-i-install-dependencies)  
4. [Part II: LongBench Evaluation](#part-ii-longbench-evaluation)  
5. [Part III: Kernel Evaluation](#part-iii-kernel-evaluation)  

---

## Overview

This repository provides:

- **Dependency setup** scripts to reproducibly install/build all required Python packages and CUDA kernels.  
- **LongBench Evaluation** to reproduce the accuracy evaluations of the paper.  
- **Kernel Evaluation** to measure latency of KV cache pruning, Mustafar batched SpMV kernel, and Triton Compression kernel to compare with dense batched MV.  

---

## Prerequisites
- Linux (Ubuntu 20.04+ recommended)  
- Python 3.10+  
- NVIDIA GPU with CUDA 12.x or higher  
- `pip` installed  
- [Optional] Predownloaded huggingface `transformers` weight cache of models to test: we currently support Llama-2, Llama-3, and Mistral-7B-Instruct-v0.2

---

## Part I: Install Dependencies


1. We recommend first initializing a venv or a conda environment. 


2. Install the requirements. 

   ```bash
   pip install -r requirements.txt 
   ```

3. Build the CUDA kernel

   ```bash
   cd /kernel
   source Init_Mustafar.sh  
   cd ../build  
   make 
   ```
   Optionally, speedup build with 
   ```bash 
   make -jN
   ```
   where N is the number of build process 

4. Register the CUDA kernel as a PyTorch extension.

   ```bash
   cd ../kernel_wrapper
   pip install -e . 
   ```

---

## Part II: LongBench Evaluation

This component runs end-to-end model benchmarks on the LongBench suite.

1. **/accuracy_eval** directory contains the necessary scripts and pruning method speicifications. 

    Under `/model`, contains several pruning methods for Llama, and **Per-token, Magnitude-based Pruning** for Mistral-7B-Instruct-v0.2.  
    Following explains the naming convention: 
    
    **K/V[t/c]_Mag/Opt**: denotes the combination of pruning strategy explored in the paper. 


    - **K/V**  
    &nbsp;&nbsp;Whether this is a **Key** or **Value** cache pruning.  
    - **[t / c]**  
    &nbsp;&nbsp;Pruning **direction**:  
        - `t` = **token-wise**  
        - `c` = **channel-wise**  
    - **Mag/Opt**  
    &nbsp;&nbsp;Pruning **method**:  
        - `Mag` = **Magnitude-based**  
        - `Opt` = **Output-aware**  


    For example, 

    | Folder name  | Cache | Direction   | Method            |
    |--------------|-------|-------------|-------------------|
    | `Kt_Mag`    | Key   | token-wise  | magnitude-based   |
    | `Vc_Opt`    | Value | channel-wise| output-aware      |

    Additionally, **llama_think.py** and **llama_thinv.py** refers to applying the structured pruning method of Xu et. al **ThinK: Thinner Key Cache by Query-Driven Pruning** to Key and Value Cache, respectively. 


2. **Run** the evaluation script:

    Before running, go to `/pred_long_bench.py` Line 139. to select the pruning method to test on. 

   ```bash
   cd /accuracy/eval
   bash long_test.sh ${k_sparsity} ${v_sparsity} ${model} ${mode}
   ```
    **k_sparsity** / **v_sparsity** refers to the target sparsity for KV cache. I.e., 50% sparsity is 0.5, 70% sparsity is 0.7

    The paper tested with the following **model** params:  
    - **Llama-2-7B**: meta-llama/Llama-2-7b-hf
    - **Llama-3-8B-Instruct**: meta-llama/Meta-Llama-3-8B-Instruct 
    - **Mistral-7B-Instruct-v0.2**: mistralai/Mistral-7B-Instruct-v0.2

    for **mode**, use 'mustafar' for llama and 'mustafar-mistral' for mistral mode family. 


3. Generate LongBench Score from the evaluation run
    
    the previous script generates generation outputs on `/pred` directory. 
    
    Generate the LongBench score by running the following:

   ```bash
   python eval_long_bench.py --model ${subdir_name}
   ```
    **subdir_name** refers to the generated subdirectory under `/pred` for each run. i.e. **Llama-2-7b-hf_4096_K_0.7_V_0.0_group32** 
---

## Part III: Kernel Evaluation

`/kernel` directory contains source code for compression Triton kernel and batched SpMV CUDA kernel

**Make sure that the CUDA kernel is built and ported to python with steps from** [Installation](#part-i-install-dependencies)  

1. **Run** the kernel evaluation script:

```bash
cd /kernel 
python kernel_evaluation.py
```

The evaluation script first tests the correctenss of bitmap-based compression + batched SpMV with dense cublas batched MV. 

```bash
#Example output
---------------------------------- Correctness Check --------------------------------
Are results equal? True
```

Then we compare the execution latency of dense attention versus Mustafar Sparse Attention formulation to show that Mustafar does not incur additional latency compared to dense inference. 

```bash
#Example output
---------------------------------- Execution Time Check --------------------------------
Dense Batched MM computation time: 0.1595020294189453 ms
50% sparsity: Prune and Compression time per token: 0.00512157566845417 ms
Batched SpMV computation time: 0.1380443572998047 ms
70% sparsity: Prune and Compression time per token: 0.0023152679204940796 ms
Batched SpMV computation time: 0.08440017700195312 ms

```

Finally, the evaluation script saves the compressed KV cache as a file. Check the memory size and compare with a dense KV cache. 

```bash
#sample memory footprint in system. 
Dense KV cache: 33.6MB 
50% sparsity compressed: 22.0MB
70% sparsity compressed: 15.5MB
```

---

