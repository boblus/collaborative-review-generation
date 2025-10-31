import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import re
import copy
import json
import time
import torch
import argparse

from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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

def arg_inference_local(image_path, model, processor, prompt_type, max_tokens=2048, temperature=1):

    image = Image.open(image_path).convert('RGB')

    message = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': prompts_arg[prompt_type]}]
        },
        {
            'role': 'user',
            'content': [{'type': 'image', 'image': image}]
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
    venue = 'emnlp24'
    prompt_type = 'emnlp24-general'
    adversarial = False
    temperature = 0.8
    max_tokens = 2048
    seed = 2266

    args = parse_args()
    
    device = args.device
    arg_model_names = args.model_names

    for arg_model_name in arg_model_names:

        experiment_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
        set_seed(seed)
    
        with open(f"papers-{venue}.json") as file:
            papers = json.loads(file.read())
        
        with open(f"postprocessed-{venue}.json") as file:
            postprocessed = json.loads(file.read())
        
        arg_input_data = prepare_arg_input_data(papers, postprocessed, adversarial)

        accelerator = Accelerator()
        processor = AutoProcessor.from_pretrained(arg_model_name)
        arg_model = Qwen3VLForConditionalGeneration.from_pretrained(arg_model_name, dtype='auto', trust_remote_code=True, device_map='auto')
        arg_model = accelerator.prepare(arg_model)
        
        device_name = torch.cuda.get_device_name(0)
    
        output = defaultdict(dict)
        with tqdm(total=len(list(arg_input_data.keys())[:100])) as t:
            for paper_id in list(arg_input_data.keys())[:100]: ########### check here
                for review_id, item in arg_input_data[paper_id].items():

                    human_review = item['review']
                    with torch.no_grad():
                        llm_review = arg_inference_local(f"papers_img/{paper_id}.png", arg_model, processor, prompt_type, max_tokens, temperature)
        
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

        accelerator.free_memory()
        del accelerator
        del arg_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
