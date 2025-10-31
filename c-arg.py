import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

import re
import copy
import json
import time
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from collections import defaultdict
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run arg experiments.")

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+'
    )

    parser.add_argument(
        '--backend',
        type=str,
        default='vllm',
        choices=['hf','vllm']
    )

    args = parser.parse_args()
    return args

def prepare_arg_input_data(papers, postprocessed, adversarial=False):
    output = defaultdict(dict)
    for paper_id, item in postprocessed.items():
        
        paper = papers[paper_id]
        if adversarial:
            paper = manipulate(paper)
        
        for review_id in item['Reviews']:
            review = {}
            for field in item['Reviews'][review_id]:
                review[field] = {k: v['text'] for k, v in item['Reviews'][review_id][field].items()}
            output[paper_id][review_id] = {'paper': paper, 'review': review}
    
    return output

def arg_inference_local(paper, model, tokenizer, prompt_type, max_tokens=2048, temperature=1, backend='vllm'):

    message = [
        {'role': 'system', 'content': prompts_arg[prompt_type]},
        {'role': 'user', 'content': f"## Paper:\n{paper}"}
        ]
    message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    if backend == 'hf':
        tokenized = tokenizer(message, max_length=16384, truncation=True, return_tensors='pt').to(model.device)
        input_length = tokenized['input_ids'].shape[1]
    
        with torch.inference_mode():
            response = model.generate(
                **tokenized,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature
                )

        output = tokenizer.decode(response[0][input_length:], skip_special_tokens=True)

    if backend == 'vllm':
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None)
        result = model.generate([message], sp)
        output = result[0].outputs[0].text
    
    try:
        if 'assistantfinal' in output:
            output = output.split('assistantfinal')[1]
        output = output.strip()
        output = re.sub(r'^[^\[{]*', '', output)
        output = re.sub(r'[^}\]]*$', '', output)
        output = json.loads(output)
        return output
    except Exception as e:
        return 'FORMAT ERROR'

def main():
    venue = 'emnlp24'
    prompt_type = 'emnlp24-general'
    adversarial = True
    temperature = 0.8
    max_tokens = 2048
    seed = 2266

    args = parse_args()
    
    device = args.device
    arg_model_names = args.model_names
    backend = args.backend

    for arg_model_name in arg_model_names:

        experiment_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
        set_seed(seed)
    
        with open(f"papers-{venue}.json") as file:
            papers = json.loads(file.read())
        
        with open(f"postprocessed-{venue}.json") as file:
            postprocessed = json.loads(file.read())
        
        arg_input_data = prepare_arg_input_data(papers, postprocessed, adversarial)

        if backend == 'hf':
            arg_tokenizer = AutoTokenizer.from_pretrained(arg_model_name, trust_remote_code=True)
            arg_tokenizer.pad_token = arg_tokenizer.eos_token if arg_tokenizer.pad_token is None else arg_tokenizer.pad_token
            arg_model = AutoModelForCausalLM.from_pretrained(arg_model_name, dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')

        if backend == 'vllm':
            arg_tokenizer = AutoTokenizer.from_pretrained(arg_model_name, trust_remote_code=True)
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")
            tp = len([x for x in gpus if x.strip()!='']) or 1
            arg_model = LLM(model=arg_model_name, tensor_parallel_size=tp, dtype='bfloat16', trust_remote_code=True, download_dir='/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache/hub')
        
        device_name = torch.cuda.get_device_name(0)
    
        output = defaultdict(dict)
        with tqdm(total=len(list(arg_input_data.keys())[:100])) as t:
            for paper_id in list(arg_input_data.keys())[:100]: ########### check here
                for review_id, item in arg_input_data[paper_id].items():
        
                    human_review = item['review']
                    with torch.no_grad():
                        llm_review = arg_inference_local(item['paper'], arg_model, arg_tokenizer, prompt_type, max_tokens, temperature, backend)
        
                    if llm_review != 'FORMAT ERROR':
                        output_index = len(output)
                        output[output_index]['paper_id'] = paper_id
                        output[output_index]['review_id'] = review_id
                        output[output_index]['human_review'] = human_review
                        output[output_index]['llm_review'] = llm_review
        
                    with open(f"arg-{experiment_id}.json", 'w') as file:
                        json.dump(output, file, indent=4, ensure_ascii=False)

                t.update(1)
    
        with open('config.txt', 'a') as file:
            file.write(f'{experiment_id}\t{venue}\t{arg_model_name}\t{prompt_type}\t-\t{adversarial}\t-\t{temperature}\t{max_tokens}\t{seed}\t{device_name}\n')

        del arg_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
