import os
os.environ["HF_HOME"] = '/pfss/mlde/workspaces/mlde_wsp_DocQuery/bob/.cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

import json
import time
import torch

from tqdm import tqdm
from collections import defaultdict
from ai_researcher import DeepReviewer
from utils import *

def prepare_arg_input_data(papers, postprocessed, type_of_labels, adversarial=False):
    output = defaultdict(dict)
    for paper_id, item in postprocessed.items():
        
        paper = papers[paper_id]
        if adversarial:
            paper = manipulate(paper)

        if type_of_labels in ['-']:
            for review_id in item['Reviews']:
                review = {}
                for field in item['Reviews'][review_id]:
                    review[field] = {k: v['text'] for k, v in item['Reviews'][review_id][field].items()}
                output[paper_id][review_id] = {'paper': paper, 'review': review}

        if type_of_labels in ['coarse']:
            for review_id in item['Reviews']:
                review, aspects = {}, {}
                for field in item['Reviews'][review_id]:
                    review[field] = {k: v['text'] for k, v in item['Reviews'][review_id][field].items()}
                    aspects[field] = {k: v[f"aspect_{type_of_labels}"] for k, v in item['Reviews'][review_id][field].items()}
                output[paper_id][review_id] = {'paper': paper, 'review': review, 'aspects': aspects}
    
    return output

def main():
    venue = 'emnlp24'
    prompt_type = 'emnlp24-general'
    type_of_labels = '-'
    adversarial = False
    temperature = '-'
    max_tokens = '-'
    seed = 2266

    arg_model_name = 'DeepReviewer-14B'
    
    experiment_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    set_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    with open(f"papers-{venue}.json") as file:
        papers = json.loads(file.read())
    
    with open(f"postprocessed-{venue}.json") as file:
        postprocessed = json.loads(file.read())
    
    arg_input_data = prepare_arg_input_data(papers, postprocessed, type_of_labels, adversarial)
    deep_reviewer = DeepReviewer(model_size=arg_model_name.split('-')[-1])

    output = defaultdict(dict)
    with tqdm(total=len(list(arg_input_data.keys())[:100])) as t:
        with torch.no_grad():
            for paper_id in list(arg_input_data.keys())[:100]:
                for review_id, item in arg_input_data[paper_id].items():
                
                    human_review = item['review']

                    if prompt_type in ['emnlp24-general']:
                        paper = item['paper']

                    if prompt_type in ['emnlp24-aspect']:
                        aspects = item['aspects']
                        paper = f"## System prompt:\n{prompts_arg[prompt_type]}\n\n\n\n## Paper:\n{item['paper']}\n\n\n\n## Bullet point dictionary: {aspects}"
    
                    try:
                        llm_review = deep_reviewer.evaluate(
                            paper,
                            mode='Standard Mode', # Options: "Fast Mode", "Standard Mode", "Best Mode"
                            reviewer_num=1
                        )
                        output_index = len(output)
                        output[output_index]['paper_id'] = paper_id
                        output[output_index]['review_id'] = review_id
                        if prompt_type in ['emnlp24-aspect']:
                            output[output_index]['aspects'] = aspects
                        output[output_index]['human_review'] = human_review
                        output[output_index]['llm_review'] = llm_review[0]['raw_text']
                    except Exception as e:
                        None
        
                    with open(f'arg-{experiment_id}.json', 'w') as file:
                        json.dump(output, file, ensure_ascii=False, indent=4)

                t.update(1)

    with open('config.txt', 'a') as file:
            file.write(f'{experiment_id}\t{venue}\t{arg_model_name}\t{prompt_type}\t{type_of_labels}\t{adversarial}\t-\t{temperature}\t{max_tokens}\t{seed}\t{device_name}\n')

if __name__ == '__main__':
    main()