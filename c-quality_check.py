import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["HF_TOKEN"] = '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'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

import re
import json
import time
import torch
import pandas as pd

from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from collections import defaultdict
from utils import *

def evidence_or_reasoning_check(llm_review, human_review, key_points, quality_check_type, model, tokenizer, max_tokens, temperature):

    message = [
        {'role': 'system', 'content': prompts_quality_check[quality_check_type]},
        {'role': 'user', 'content': f"## Key point and judgment: {key_points}\n\n## LLM-generated review: {llm_review}\n\n## Human-written review: {human_review}"}
        ]
    message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenized = tokenizer(message, max_length=16384, truncation=True, return_tensors='pt').to(model.device)
    input_length = tokenized['input_ids'].shape[1]
    
    response = model.generate(
        **tokenized,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_tokens,
        temperature=temperature
        )
    
    output = tokenizer.decode(response[0][input_length:], skip_special_tokens=True)
    
    try:
        output = output.strip()
        output = re.sub(r'^[^\[{]*', '', output)
        output = re.sub(r'[^}\]]*$', '', output)
        output = json.loads(output)
        return f"{output['label']}: {output['reason']}"
    except Exception as e:
        return 'FORMAT ERROR'

def main():
    config = pd.read_csv('config.txt', sep='\t')

    venue = 'emnlp24'
    prompt_type = 'emnlp24-aspect'
    type_of_labels = 'coarse'
    temperature = 0.8
    max_tokens = 512
    seed = 2266

    for quality_check_model_name in ['Qwen/Qwen3-32B']:
        
        for arg_model_name in ['Qwen/Qwen3-32B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B']:
    
            run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())

            set_seed(seed)
    
            match = config['experiment_id'][(config['venue'] == venue) & (config['arg_model'] == arg_model_name) & (config['prompt_type'] == prompt_type) & (config['type_of_labels'] == type_of_labels)].to_list()
            if len(match) != 1:
                raise ValueError(f"multiple matches")
            else:
                experiment_id = match[0]
                with open(f"results/judge-{experiment_id}.json") as file:
                    judge = json.loads(file.read())
                
            accelerator = Accelerator()
            quality_check_tokenizer = AutoTokenizer.from_pretrained(quality_check_model_name, trust_remote_code=True)
            quality_check_tokenizer.pad_token = quality_check_tokenizer.eos_token if quality_check_tokenizer.pad_token is None else quality_check_tokenizer.pad_token
            quality_check_model = AutoModelForCausalLM.from_pretrained(quality_check_model_name, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            quality_check_model = accelerator.prepare(quality_check_model)
            device = torch.cuda.get_device_name(0)
    
            for _, item in judge.items():
                for turn in item['llm_reviews']:
                    if item['llm_reviews'][turn]['checks']['specification'] == True:
                        pass_evidence = evidence_or_reasoning_check(item['llm_reviews'][turn]['llm_review'], item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'evidence', quality_check_model, quality_check_tokenizer, max_tokens, temperature)
                        item['llm_reviews'][turn]['checks']['evidence'] = pass_evidence
    
                        if item['llm_reviews'][turn]['checks']['evidence'].split(': ')[0] in ['PASS', 'MATCH', 'SUFFICIENT']:
                            pass_reasoning = evidence_or_reasoning_check(item['llm_reviews'][turn]['llm_review'], item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'reasoning', quality_check_model, quality_check_tokenizer, max_tokens, temperature)
                            item['llm_reviews'][turn]['checks']['reasoning'] = pass_reasoning
    
                        with open(f"quality_check-{run_id}.json", 'w') as file:
                            json.dump(judge, file, indent=4, ensure_ascii=False)
    
            with open('config_quality_check.txt', 'a') as file:
                file.write(f'{run_id}\t{experiment_id}\t{quality_check_model_name}\t{temperature}\t{max_tokens}\t{seed}\t{device}\n')

            del quality_check_model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()