#Code initially built from KIVI at https://github.com/jy-yuan/KIVI


import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"

from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer, AutoConfig



# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    dtype = torch.float16
    
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        print("---------------------model name: ", model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True, 
                                            #tokenizer_type='llama',
                                            #cache_dir=cache_dir)
                                            )
                                            # model_max_length=training_args.model_max_length)
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)
    
    
    else:
        raise NotImplementedError
    
    operation_mode = model_args.mode
    if operation_mode == 'mustafar':
    
        if operation_mode == 'mustafar':
            print("@@@@@@@@@@@Using Mustafar")
            
            #select the pruning method to use. 
            #from models.llama_mustafar_Kt_Mag_Vc_Mag import LlamaForCausalLM_MUSTAFAR
            #from models.llama_mustafar_Kt_Mag_Vc_Opa import LlamaForCausalLM_MUSTAFAR
            from models.llama_mustafar_Kt_Mag_Vt_Mag import LlamaForCausalLM_MUSTAFAR
            #from models.llama_mustafar_Kt_Mag_Vt_Opa import LlamaForCausalLM_MUSTAFAR
            #from models.llama_mustafar_Kt_Opa_Vt_Mag import LlamaForCausalLM_MUSTAFAR
            #from models.llama_think import LlamaForCausalLM_MUSTAFAR
            #from models.llama_thinv import LlamaForCausalLM_MUSTAFAR


            #print("Using the V-per-token pruning model.")
            config.k_sparsity = model_args.k_sparsity
            config.v_sparsity = model_args.v_sparsity
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            config.use_flash = True
            model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                #cache_dir=training_args.cache_dir,
                #cache_dir=cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )

        else:
            raise NotImplementedError
    
    
    elif operation_mode == 'mustafar-mistral':
        from models.mistral_mustafar_Kt_Mag_Vt_Mag import MistralForCausalLM_MUSTAFAR
        print("@@@@@@@@@@@Using Mustafar mistral")
        config.use_flash = True
        model = MistralForCausalLM_MUSTAFAR.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )


    else:
        raise NotImplementedError


    model.eval()
    max_length = model2maxlen[model_name]
    if data_args.e:
        print("Evaluating on Extended Benchmark Set!")
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        print("Evaluating on All Benchmark Set")
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    #datasets = ["qasper"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    
    
    if operation_mode in ['mustafar', 'mustafar-llama3', 'mustafar-mistral']:

        for dataset in datasets:
            if data_args.e:
                data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
                if not os.path.exists(f"pred_e/{model_name}_{max_length}_K_{model_args.k_sparsity}_V_{model_args.v_sparsity}_group{model_args.group_size}"):
                    os.makedirs(f"pred_e/{model_name}_{max_length}_K_{model_args.k_sparsity}_V_{model_args.v_sparsity}_group{model_args.group_size}")
                out_path = f"pred_e/{model_name}_{max_length}_K_{model_args.k_sparsity}_V_{model_args.v_sparsity}_group{model_args.group_size}/{dataset}.jsonl"
            else:
                data = load_dataset('THUDM/LongBench', dataset, split='test')
                if not os.path.exists(f"pred/{model_name}_{max_length}_K_{model_args.k_sparsity}_V_{model_args.v_sparsity}_group{model_args.group_size}"):
                    os.makedirs(f"pred/{model_name}_{max_length}_K_{model_args.k_sparsity}_V_{model_args.v_sparsity}_group{model_args.group_size}")
                out_path = f"pred/{model_name}_{max_length}_K_{model_args.k_sparsity}_V_{model_args.v_sparsity}_group{model_args.group_size}/{dataset}.jsonl"
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
    
