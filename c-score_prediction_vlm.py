import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

import re
import json
import time
import torch
import pandas as pd

from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from collections import defaultdict
from utils import *

def inference(review, model, processor, max_tokens=2048, temperature=1):

    message = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': prompt_score_prediction}]
        },
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': f"## Review: {review}"}]
        }
        ]
    processed = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    
    response = model.generate(
        **processed,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_new_tokens=max_tokens,
        temperature=temperature
        )

    output = processor.batch_decode(response[:, processed['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    
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
    prompt_type = 'emnlp24-aspect'
    type_of_labels = 'coarse'
    adversarial = False

    round_n = 'last_round'
    
    device = torch.cuda.get_device_name(0)

    for arg_model_name in ['Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-32B-Instruct']:

        run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        match = config['experiment_id'][(config['venue'] == venue) & (config['arg_model'] == arg_model_name) & (config['prompt_type'] == prompt_type) & (config['type_of_labels'] == type_of_labels) & (config['adversarial'] == adversarial)].to_list()
        if len(match) != 1:
            raise ValueError(f"multiple matches")
        else:
            experiment_id = match[0]
            temperature = float(config['temperature'][config['experiment_id'] == experiment_id].to_list()[0])
            max_tokens = int(config['max_tokens'][config['experiment_id'] == experiment_id].to_list()[0])
            seed = int(config['seed'][config['experiment_id'] == experiment_id].to_list()[0])
            if prompt_type in ['emnlp24-general']:
                arg = defaultdict(dict)
                with open(f"results/arg-{experiment_id}.json") as file:
                    for _, item in json.loads(file.read()).items():
                        if type(item['llm_review']) == str:
                            llm_review = json.loads(item['llm_review'])
                        elif type(item['llm_review']) == dict:
                            llm_review = item['llm_review']
                        arg[item['paper_id']][item['review_id']] = {'strengths': ' '.join(llm_review['strengths']), 'weaknesses': ' '.join(llm_review['weaknesses'])}
    
            if prompt_type in ['emnlp24-aspect']:
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
    
        accelerator = Accelerator()
        processor = AutoProcessor.from_pretrained(arg_model_name)
        arg_model = Qwen3VLForConditionalGeneration.from_pretrained(arg_model_name, dtype='auto', trust_remote_code=True, device_map='auto')
        arg_model = accelerator.prepare(arg_model)

        set_seed(seed)
            
        output = defaultdict(dict)
        with tqdm(total=len(arg)) as t:
            for paper_id in arg:
                for review_id in arg[paper_id]:

                    with torch.no_grad():
                        score_prediction = inference(arg[paper_id][review_id], arg_model, processor, max_tokens, temperature)
    
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
    
        accelerator.free_memory()
        del accelerator
        del arg_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
