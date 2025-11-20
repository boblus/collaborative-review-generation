import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

import re
import json
import time
import torch
import pandas as pd

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from collections import defaultdict
from utils import *

def inference(review, venue, model, tokenizer, max_tokens=2048, temperature=1, backend='vllm'):

    message = [
        {'role': 'system', 'content': prompt_score_prediction[venue]},
        {'role': 'user', 'content': f"## Review: {review}"}
        ]
    message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    if backend == 'hf':
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
    config = pd.read_csv('config.txt', sep='\t')

    venue = 'iclr25'
    prompt_type = 'iclr25-aspect'
    type_of_labels = 'coarse'
    adversarial = False

    round_n = 'first_round'

    backend = 'vllm'
    device = torch.cuda.get_device_name(0)

    for arg_model_name in ['openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'Qwen/Qwen3-14B', 'Qwen/Qwen3-32B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B']:

        run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        match = config['experiment_id'][(config['venue'] == venue) & (config['arg_model'] == arg_model_name) & (config['prompt_type'] == prompt_type) & (config['type_of_labels'] == type_of_labels) & (config['adversarial'] == adversarial)].to_list()
        if len(match) != 1:
            raise ValueError(f"multiple matches")
        else:
            experiment_id = match[0]
            temperature = float(config['temperature'][config['experiment_id'] == experiment_id].to_list()[0])
            max_tokens = int(config['max_tokens'][config['experiment_id'] == experiment_id].to_list()[0])
            seed = int(config['seed'][config['experiment_id'] == experiment_id].to_list()[0])
            if prompt_type in ['emnlp24-general', 'iclr25-general']:
                arg = defaultdict(dict)
                with open(f"results/arg-{experiment_id}.json") as file:
                    for _, item in json.loads(file.read()).items():
                        if type(item['llm_review']) == str:
                            llm_review = json.loads(item['llm_review'])
                        elif type(item['llm_review']) == dict:
                            llm_review = item['llm_review']

                        if 'strengths' in llm_review and 'weaknesses' in llm_review:
                            arg[item['paper_id']][item['review_id']] = {'strengths': llm_review['strengths'], 'weaknesses': llm_review['weaknesses']}
    
            if prompt_type in ['emnlp24-aspect', 'iclr25-aspect']:
                arg = {}
                with open(f"results/judge-{experiment_id}.json") as file:
                    for _, item in json.loads(file.read()).items():
    
                        if item['paper_id'] not in arg:
                            arg[item['paper_id']] = defaultdict(list)

                        if round_n == 'last_round':
                            turn = list(item['llm_reviews'].keys())[-1]
                            if item['llm_reviews'][turn]['checks']['reasoning'].split(': ')[0] == 'YES':
                                arg[item['paper_id']][item['review_id']].append((item['field'], item['llm_reviews'][turn]['llm_review']))
                            else:
                                arg[item['paper_id']][item['review_id']].append((item['field'], item['llm_reviews']['0']['llm_review']))

                        if round_n == 'first_round':
                            arg[item['paper_id']][item['review_id']].append((item['field'], item['llm_reviews']['0']['llm_review']))
                    
                    for paper_id in arg:
                        for review_id in arg[paper_id]:
                            strengths, weaknesses = {}, {}
                            for item in arg[paper_id][review_id]:
                                if item[0] == 'strengths':
                                    strengths[len(strengths)] = item[1]
                                if item[0] == 'weaknesses':
                                    weaknesses[len(weaknesses)] = item[1]
                            arg[paper_id][review_id] = {'strengths': strengths, 'weaknesses': weaknesses}

        if backend == 'hf':
            arg_tokenizer = AutoTokenizer.from_pretrained(arg_model_name, trust_remote_code=True)
            arg_tokenizer.pad_token = arg_tokenizer.eos_token if arg_tokenizer.pad_token is None else arg_tokenizer.pad_token
            arg_model = AutoModelForCausalLM.from_pretrained(arg_model_name, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

        if backend == 'vllm':
            arg_tokenizer = AutoTokenizer.from_pretrained(arg_model_name, trust_remote_code=True)
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")
            tp = len([x for x in gpus if x.strip()!='']) or 1
            arg_model = LLM(model=arg_model_name, tensor_parallel_size=tp, dtype='bfloat16', trust_remote_code=True, download_dir='/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache/hub')
        
        set_seed(seed)
    
        output = defaultdict(dict)
        with tqdm(total=len(arg)) as t:
            for paper_id in arg:
                for review_id in arg[paper_id]:
                    
                    score_prediction = inference(arg[paper_id][review_id], venue, arg_model, arg_tokenizer, max_tokens, temperature, backend)
    
                    if score_prediction != 'FORMAT ERROR':
                        output_index = len(output)
                        output[output_index]['paper_id'] = paper_id
                        output[output_index]['review_id'] = review_id
                        output[output_index]['judge'] = score_prediction
            
                    with open(f'score_prediction-{run_id}.json', 'w') as file:
                        json.dump(output, file, ensure_ascii=False, indent=4)
            
                    t.update(1)
    
        with open('config_score_prediction.txt', 'a') as file:
            file.write(f'{run_id}\t{experiment_id}\t{arg_model_name}\t{prompt_type}\t{round_n}\t{device}\n')
    
        del arg_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()