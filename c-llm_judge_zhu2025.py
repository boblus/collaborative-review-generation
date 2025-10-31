import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import re
import json
import time
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from collections import defaultdict
from utils import *

def inference(paper, review_A, review_B, model, tokenizer, max_tokens=8192, temperature=1):

    message = [
        {'role': 'system', 'content': prompt_llm_judge_prometheus},
        {'role': 'user', 'content': f"## Paper: {paper}\n\n## Review A: {review_A}\n\n## Review B: {review_B}"}
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
        return output
    except Exception as e:
        return 'FORMAT ERROR'

def main():
    config = pd.read_csv('config.txt', sep='\t')

    venue = 'emnlp24'

    for arg_model_name_a in ['DeepReviewer-14B']:
    
        prompt_type_a = 'emnlp24-general'
        type_of_labels_a = '-'
        
        arg_model_name_b = arg_model_name_a
        prompt_type_b = 'emnlp24-aspect'
        type_of_labels_b = 'coarse'
    
        round_n = 'last_round'
        adversarial = False
    
        judge_model_name = 'meta-llama/Llama-3.3-70B-Instruct'
        temperature = 0.8
        max_tokens = 8192
    
        seed = 2266
        set_seed(seed)
    
        device = torch.cuda.get_device_name(0)
    
        run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
        with open(f"papers-{venue}.json") as file:
            papers = json.loads(file.read())
    
        match = config['experiment_id'][(config['venue'] == venue) & (config['arg_model'] == arg_model_name_a) & (config['prompt_type'] == prompt_type_a) & (config['type_of_labels'] == type_of_labels_a) & (config['adversarial'] == adversarial)].to_list()
        if len(match) != 1:
            raise ValueError(f"multiple matches")
        else:
            experiment_id_a = match[0]
            arg_a = defaultdict(dict)
            with open(f"results/arg-{experiment_id_a}.json") as file:
                for _, item in json.loads(file.read()).items():
                    if type(item['llm_review']) == str:
                        llm_review = json.loads(item['llm_review'])
                    elif type(item['llm_review']) == dict:
                        llm_review = item['llm_review']
                    arg_a[item['paper_id']][item['review_id']] = {'strengths': llm_review['strengths'], 'weaknesses': llm_review['weaknesses']}
    
        match = config['experiment_id'][(config['venue'] == venue) & (config['arg_model'] == arg_model_name_b) & (config['prompt_type'] == prompt_type_b) & (config['type_of_labels'] == type_of_labels_b) & (config['adversarial'] == adversarial)].to_list()
        if len(match) != 1:
            raise ValueError(f"multiple matches")
        else:
            experiment_id_b = match[0]
            arg_b = defaultdict(dict)
            with open(f"results/arg-{experiment_id_b}.json") as file:
                for _, item in json.loads(file.read()).items():
                    if type(item['llm_review']) == str:
                        llm_review = json.loads(item['llm_review'])
                    elif type(item['llm_review']) == dict:
                        llm_review = item['llm_review']
                    arg_b[item['paper_id']][item['review_id']] = {'strengths': llm_review['strengths'], 'weaknesses': llm_review['weaknesses']}
    
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
        judge_tokenizer.pad_token = judge_tokenizer.eos_token if judge_tokenizer.pad_token is None else judge_tokenizer.pad_token
        judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name, dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
        set_seed(seed)
    
        output = defaultdict(dict)
        with tqdm(total=len(arg_a)) as t:
            for paper_id in arg_a:
                for review_id in arg_a[paper_id]:
    
                    if paper_id in arg_b and review_id in arg_b[paper_id]:

                        with torch.no_grad():
                            judge_result = inference(papers[paper_id], arg_a[paper_id][review_id], arg_b[paper_id][review_id], judge_model, judge_tokenizer, max_tokens, temperature)
    
                        if judge_result != 'FORMAT ERROR':
                            output_index = len(output)
                            output[output_index]['paper_id'] = paper_id
                            output[output_index]['review_id'] = review_id
                            output[output_index]['judge'] = judge_result
                
                        with open(f'llm_judge-{run_id}.json', 'w') as file:
                            json.dump(output, file, ensure_ascii=False, indent=4)
            
                    t.update(1)
    
        with open('config_llm_judge.txt', 'a') as file:
            file.write(f'{run_id}\t{judge_model_name}\t{experiment_id_a}\t{experiment_id_b}\t{round_n}\t{temperature}\t{max_tokens}\t{seed}\t{device}\n')

        del judge_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
