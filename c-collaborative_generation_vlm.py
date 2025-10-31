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

from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Qwen3VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from collections import defaultdict
from utils import *

OPENAI_CLIENT = OpenAI(api_key='')

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

def load_rap_model_and_tokenizer(model_path, config, device):
    state_dict = torch.load(model_path)
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(state_dict)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    return model, tokenizer

def load_aspect_categories(type_of_labels, aspect_file_path):
    category_to_aspect = pd.read_csv(aspect_file_path)
    aspect_to_category = defaultdict(set)
    for i in range(len(category_to_aspect)):
        aspect_to_category[category_to_aspect['LLM annotation'].to_list()[i]].add(category_to_aspect[type_of_labels.upper()].to_list()[i])
    
    categories = ['-']
    categories.extend(category_to_aspect[type_of_labels.upper()].to_list())
    categories = sorted(list(set(categories)))

    return categories

def load_fine_to_coarse_aspect(coarse_aspect_file_path, fine_aspect_file_path):
    aspects_coarse = pd.read_csv(coarse_aspect_file_path)
    aspects_fine = pd.read_csv(fine_aspect_file_path)
    
    llm_annotation_to_aspect = defaultdict(dict)
    for type_of_labels, aspects in {'COARSE': aspects_coarse, 'FINE': aspects_fine}.items():
        for i in range(len(aspects)):
            llm_annotation_to_aspect[aspects['LLM annotation'].to_list()[i]][type_of_labels] = aspects[type_of_labels].to_list()[i]

    fine_to_coarse_aspect = dict()
    for _, pair in llm_annotation_to_aspect.items():
        fine_to_coarse_aspect[pair['FINE']] = pair['COARSE']

    return fine_to_coarse_aspect

def rap_inference(input, model, tokenizer, categories):
    input = tokenizer(input, return_tensors='pt', max_length=512, truncation=True)
    input = {k: v.to(model.device) for k, v in input.items()}

    with torch.no_grad():
        model_output = model(**input)
    
    logits = model_output.logits
    prediction = torch.sigmoid(logits) > 0.5
    prediction = prediction.cpu().numpy().squeeze()
    
    output = []
    for i in range(len(prediction)):
        if prediction[i] == 1:
            output.append(categories[i])
    
    return output

def prepare_arg_input_data(papers, postprocessed, type_of_labels, adversarial=False):
    output = defaultdict(dict)
    for paper_id, item in postprocessed.items():
        
        paper = papers[paper_id]
        if adversarial:
            paper = manipulate(paper)
        
        for review_id in item['Reviews']:
            review, aspects = {}, {}
            for field in item['Reviews'][review_id]:
                review[field] = {k: v['text'] for k, v in item['Reviews'][review_id][field].items()}
                aspects[field] = {k: v[f"aspect_{type_of_labels}"] for k, v in item['Reviews'][review_id][field].items()}
            output[paper_id][review_id] = {'paper': paper, 'review': review, 'aspects': aspects}
    
    return output

def arg_inference_openai(paper, aspects, model, tokenizer, prompt_type, max_tokens='low', temperature=1):

    response = OPENAI_CLIENT.responses.create(
        model=model,
        input=[
            {'role': 'developer', 'content': [{'type': 'input_text', 'text': prompts_arg[prompt_type]}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': f"## Paper:\n{paper}\n\n\n\n## Bullet point dictionary: {aspects}"}]}
            ],
        text={'format': {'type': 'json_object'}, 'verbosity': max_tokens},
        temperature=temperature
        )

    try:
        output = json.loads(response.output[1].content[0].text)
        return output
    except Exception as e:
        return 'FORMAT ERROR'

def arg_inference_next_round_openai(paper, aspects, llm_review, problematic_review, model, tokenizer, prompt_type, quality_check_type, new_aspects='', max_tokens='low', temperature=1):

    response = OPENAI_CLIENT.responses.create(
        model=model,
        input=[
            {'role': 'developer', 'content': [{'type': 'input_text', 'text': prompts_arg[prompt_type]}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': f"## Paper:\n{paper}\n\n\n\n## Bullet point dictionary: {aspects}"}]},
            {'role': 'assistant', 'content': [{'type': 'output_text', 'text': f"{llm_review}"}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': prompts_arg_next_round[quality_check_type].replace('$[GENERATED_REVIEW]$', problematic_review).replace('$[KEY_POINTS]$', new_aspects)}]}
            ],
        text={'format': {'type': 'json_object'}, 'verbosity': max_tokens},
        temperature=temperature
        )

    try:
        output = json.loads(response.output[1].content[0].text)['review']
        return output
    except Exception as e:
        return 'FORMAT ERROR'

def arg_inference_local(image_path, aspects, model, processor, prompt_type, max_tokens=2048, temperature=1):

    image = Image.open(image_path).convert('RGB')
    
    message = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': prompts_arg[prompt_type]}]
        },
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': f"## Bullet point dictionary: {aspects}"}
            ]
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

def arg_inference_next_round_local(image_path, aspects, llm_review, problematic_review, model, processor, prompt_type, quality_check_type, new_aspects='', max_tokens=2048, temperature=1):

    image = Image.open(image_path).convert('RGB')
    
    message = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': prompts_arg[prompt_type]}]
        },
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': f"## Bullet point dictionary: {aspects}"}
            ]
        },
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': f"{llm_review}"}]
        },
        {
            'role': 'user', 'content': [{'type': 'text', 'text': prompts_arg_next_round[quality_check_type].replace('$[GENERATED_REVIEW]$', problematic_review).replace('$[KEY_POINTS]$', new_aspects)}]
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
        output = json.loads(output)['review']
        return output
    except Exception as e:
        return 'FORMAT ERROR'

def relevance_check(candidate_aspect_set, reference_aspect_set, fine_to_coarse_aspect):
    passed = True
    coarse_labels = set([fine_to_coarse_aspect[_] for _ in candidate_aspect_set if _ != '-'])
    for aspect in reference_aspect_set:
        if aspect not in coarse_labels:
            passed = False
            break
    return passed

def specification_check(candidate_aspect_set, reference_aspect_set):
    passed = True
    for aspect in reference_aspect_set:
        if aspect not in candidate_aspect_set:
            passed = False
            break
    return passed

def evidence_or_reasoning_check(llm_review, human_review, key_points, quality_check_type):
    response = OPENAI_CLIENT.responses.create(
        model='o3-mini',
        input=[
            {'role': 'developer', 'content': [{'type': 'input_text', 'text': prompts_quality_check[quality_check_type]}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': f"## Key point and judgment: {key_points}\n\n## LLM-generated review: {llm_review}\n\n## Human-written review: {human_review}"}]}
            ],
        text={'format': {'type': 'json_object'}, 'verbosity': 'medium'},
        temperature=1
        )
    
    try:
        output = json.loads(response.output[1].content[0].text)
        return f"{output['label']}: {output['reason']}"
    except Exception as e:
        return 'FORMAT ERROR'

def find_target_llm_review(data, paper_id, review_id):
    for _, item in data.items():
        if item['paper_id'] == paper_id and item['review_id'] == review_id:
            return item['llm_review']

def locate_review(human_review, item):
    for field in item:
        for index in item[field]:
            if item[field][index]['text'] == human_review:
                return field, index

def main():
    venue = 'emnlp24'
    prompt_type = 'emnlp24-aspect'
    type_of_labels = 'coarse'
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

        config = AutoConfig.from_pretrained('FacebookAI/roberta-base')
        config.num_labels = 72
        if not hasattr(config, "_output_attentions"):
            config._output_attentions = False
        
        rap_model_id = {'coarse': '20250102_125421', 'fine': '20250102_135556'}['fine']
        rap_model_path = f"{rap_model_id}_state_dict.pth"
        rap_model, rap_tokenizer = load_rap_model_and_tokenizer(rap_model_path, config, device)
    
        aspect_file_path = 'aspects - {}.csv'
        aspect_categories = load_aspect_categories('fine', aspect_file_path.format('fine'))
        fine_to_coarse_aspect = load_fine_to_coarse_aspect(aspect_file_path.format('coarse'), aspect_file_path.format('fine'))
        
        arg_input_data = prepare_arg_input_data(papers, postprocessed, type_of_labels, adversarial)
    
        if arg_model_name in ['gpt-5']:
            arg_model = arg_model_name
            arg_inference = arg_inference_openai
            arg_inference_next_round = arg_inference_next_round_openai
            arg_tokenizer = None
            device_name = '-'
    
        else:
            arg_inference = arg_inference_local
            arg_inference_next_round = arg_inference_next_round_local

            accelerator = Accelerator()
            arg_processor = AutoProcessor.from_pretrained(arg_model_name)
            arg_model = Qwen3VLForConditionalGeneration.from_pretrained(arg_model_name, dtype='auto', trust_remote_code=True, device_map='auto')
            arg_model = accelerator.prepare(arg_model)
            device_name = torch.cuda.get_device_name(0)
    
        output = defaultdict(dict)
        with tqdm(total=len(list(arg_input_data.keys())[:100])) as t:
            for paper_id in list(arg_input_data.keys())[:100]: ########### check here
                for review_id, item in arg_input_data[paper_id].items():
        
                    aspects = item['aspects']
                    human_review = item['review']
                    with torch.no_grad():
                        llm_review = arg_inference(f"papers_img/{paper_id}.png", aspects, arg_model, arg_processor, prompt_type, max_tokens, temperature)
        
                    if llm_review != 'FORMAT ERROR':
                        if len(llm_review['strengths']) == len(human_review['strengths']) and len(llm_review['weaknesses']) == len(human_review['weaknesses']):
                            output_index = len(output)
                            output[output_index]['paper_id'] = paper_id
                            output[output_index]['review_id'] = review_id
                            output[output_index]['aspects'] = aspects
                            output[output_index]['human_review'] = human_review
                            output[output_index]['llm_review'] = llm_review
        
                    with open(f"arg-{experiment_id}.json", 'w') as file:
                        json.dump(output, file, indent=4, ensure_ascii=False)

                t.update(1)
    
        judge = {}
        for _, item in output.items():
            for field in ['strengths', 'weaknesses']:
                for i, human_review in item['human_review'][field].items():
    
                    llm_review = str(item['llm_review'][field][i])
                    llm_aspects = rap_inference(llm_review, rap_model, rap_tokenizer, aspect_categories)
                    pass_relevance = relevance_check(llm_aspects, postprocessed[item['paper_id']]['Reviews'][item['review_id']][field][i]['aspect_coarse'], fine_to_coarse_aspect)
                    
                    pass_specification, pass_evidence, pass_reasoning = '-', '-', '-'
                    if pass_relevance:
                        pass_specification = specification_check(llm_aspects, postprocessed[item['paper_id']]['Reviews'][item['review_id']][field][i]['aspect_fine'])
                    if pass_specification == True:
                        pass_evidence = evidence_or_reasoning_check(llm_review, human_review, f"{field.upper()}: {item['aspects'][field][i]}", 'evidence')
                    if pass_evidence.split(': ')[0] in ['PASS', 'MATCH', 'SUFFICIENT']:
                        pass_reasoning = evidence_or_reasoning_check(llm_review, human_review, f"{field.upper()}: {item['aspects'][field][i]}", 'reasoning')
                    
                    entry = {'0': {'llm_review': llm_review, 'checks': {'relevance': pass_relevance, 'specification': pass_specification, 'evidence': pass_evidence, 'reasoning': pass_reasoning}}}
                    judge[len(judge)] = {'paper_id': item['paper_id'], 'review_id': item['review_id'], 'human_review': human_review, 'field': field, 'aspects': item['aspects'][field][i], 'llm_reviews': entry}
    
            with open(f"judge-{experiment_id}.json", 'w') as file:
                json.dump(judge, file, indent=4, ensure_ascii=False)
    
        # next round: relevance
        for _, item in judge.items():
            turn = list(item['llm_reviews'].keys())[-1]
            if item['llm_reviews'][turn]['checks']['relevance'] == False:
    
                _, index = locate_review(item['human_review'], postprocessed[item['paper_id']]['Reviews'][item['review_id']])
                llm_review = find_target_llm_review(output, item['paper_id'], item['review_id'])
                new_llm_review = str(arg_inference_next_round(f"papers_img/{item['paper_id']}.png", arg_input_data[item['paper_id']][item['review_id']]['aspects'], llm_review, str({item['field']: {index: item['llm_reviews'][turn]['llm_review']}}), arg_model, arg_processor, prompt_type, 'relevance', max_tokens=max_tokens, temperature=temperature))
    
                if new_llm_review != 'FORMAT ERROR':
                    new_llm_aspects = rap_inference(new_llm_review, rap_model, rap_tokenizer, aspect_categories)
                    pass_relevance = relevance_check(new_llm_aspects, postprocessed[item['paper_id']]['Reviews'][item['review_id']][item['field']][index]['aspect_coarse'], fine_to_coarse_aspect)
    
                    item['llm_reviews'][str(int(turn) + 1)] = copy.deepcopy(item['llm_reviews'][turn])
                    item['llm_reviews'][str(int(turn) + 1)]['llm_review'] = new_llm_review
                    item['llm_reviews'][str(int(turn) + 1)]['checks']['relevance'] = pass_relevance
                    
                    if pass_relevance == True:
                        pass_specification = specification_check(new_llm_aspects, postprocessed[item['paper_id']]['Reviews'][item['review_id']][item['field']][index]['aspect_fine'])
                        item['llm_reviews'][str(int(turn) + 1)]['checks']['specification'] = pass_specification
                    
                    if item['llm_reviews'][str(int(turn) + 1)]['checks']['specification'] == True:
                        pass_evidence = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'evidence')
                        item['llm_reviews'][str(int(turn) + 1)]['checks']['evidence'] = pass_evidence
    
                    if item['llm_reviews'][str(int(turn) + 1)]['checks']['evidence'].split(': ')[0] in ['PASS', 'MATCH', 'SUFFICIENT']:
                        pass_reasoning = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'reasoning')
                        item['llm_reviews'][str(int(turn) + 1)]['checks']['reasoning'] = pass_reasoning
    
                with open(f"judge-{experiment_id}.json", 'w') as file:
                    json.dump(judge, file, indent=4, ensure_ascii=False)
    
        # next round: specification
        for _, item in judge.items():
            turn = list(item['llm_reviews'].keys())[-1]
            if item['llm_reviews'][turn]['checks']['specification'] == False:
    
                _, index = locate_review(item['human_review'], postprocessed[item['paper_id']]['Reviews'][item['review_id']])
                llm_review = find_target_llm_review(output, item['paper_id'], item['review_id'])
                new_llm_review = str(arg_inference_next_round(f"papers_img/{item['paper_id']}.png", arg_input_data[item['paper_id']][item['review_id']]['aspects'], llm_review, str({item['field']: {index: item['llm_reviews'][turn]['llm_review']}}), arg_model, arg_processor, prompt_type, 'specification', str({item['field']: {index: postprocessed[item['paper_id']]['Reviews'][item['review_id']][item['field']][index]['aspect_fine']}}), max_tokens, temperature))
    
                if new_llm_review != 'FORMAT ERROR':
                    new_llm_aspects = rap_inference(new_llm_review, rap_model, rap_tokenizer, aspect_categories)
                    pass_specification = specification_check(new_llm_aspects, postprocessed[item['paper_id']]['Reviews'][item['review_id']][item['field']][index]['aspect_fine'])
    
                    item['llm_reviews'][str(int(turn) + 1)] = copy.deepcopy(item['llm_reviews'][turn])
                    item['llm_reviews'][str(int(turn) + 1)]['llm_review'] = new_llm_review
                    item['llm_reviews'][str(int(turn) + 1)]['checks']['specification'] = pass_specification
    
                    if pass_specification == True:
                        pass_evidence = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'evidence')
                        item['llm_reviews'][str(int(turn) + 1)]['checks']['evidence'] = pass_evidence
    
                    if item['llm_reviews'][str(int(turn) + 1)]['checks']['evidence'].split(': ')[0] in ['PASS', 'MATCH', 'SUFFICIENT']:
                        pass_reasoning = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'reasoning')
                        item['llm_reviews'][str(int(turn) + 1)]['checks']['reasoning'] = pass_reasoning
    
                with open(f"judge-{experiment_id}.json", 'w') as file:
                    json.dump(judge, file, indent=4, ensure_ascii=False)
    
        # next round: evidence
        for _, item in judge.items():
            turn = list(item['llm_reviews'].keys())[-1]
            if item['llm_reviews'][turn]['checks']['evidence'].split(': ')[0] in ['HALLUCINATED', 'INSUFFICIENT']:
                
                llm_review = find_target_llm_review(output, item['paper_id'], item['review_id'])
                new_llm_review = arg_inference_next_round(f"papers_img/{item['paper_id']}.png", arg_input_data[item['paper_id']][item['review_id']]['aspects'], llm_review, str({item['field']: {index: item['llm_reviews'][turn]['llm_review']}}), arg_model, arg_processor, prompt_type, 'evidence', max_tokens=max_tokens, temperature=temperature)
                
                if new_llm_review != 'FORMAT ERROR':
                    pass_evidence = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'evidence')
    
                    item['llm_reviews'][str(int(turn) + 1)] = copy.deepcopy(item['llm_reviews'][turn])
                    item['llm_reviews'][str(int(turn) + 1)]['llm_review'] = new_llm_review
                    item['llm_reviews'][str(int(turn) + 1)]['checks']['evidence'] = pass_evidence
    
                    if pass_evidence.split(': ')[0] in ['PASS', 'MATCH', 'SUFFICIENT']:
                        pass_reasoning = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'reasoning')
                        item['llm_reviews'][str(int(turn) + 1)]['checks']['reasoning'] = pass_reasoning
    
                with open(f"judge-{experiment_id}.json", 'w') as file:
                    json.dump(judge, file, indent=4, ensure_ascii=False)
    
        # next round: reasoning
        for _, item in judge.items():
            turn = list(item['llm_reviews'].keys())[-1]
            if item['llm_reviews'][turn]['checks']['reasoning'].split(': ')[0] in ['NO']:
                
                llm_review = find_target_llm_review(output, item['paper_id'], item['review_id'])
                new_llm_review = arg_inference_next_round(f"papers_img/{item['paper_id']}.png", arg_input_data[item['paper_id']][item['review_id']]['aspects'], llm_review, str({item['field']: {index: item['llm_reviews'][turn]['llm_review']}}), arg_model, arg_processor, prompt_type, 'reasoning', max_tokens=max_tokens, temperature=temperature)
                
                if new_llm_review != 'FORMAT ERROR':
                    pass_reasoning = evidence_or_reasoning_check(new_llm_review, item['human_review'], f"{item['field'].upper()}: {item['aspects']}", 'reasoning')
    
                    item['llm_reviews'][str(int(turn) + 1)] = copy.deepcopy(item['llm_reviews'][turn])
                    item['llm_reviews'][str(int(turn) + 1)]['llm_review'] = new_llm_review
                    item['llm_reviews'][str(int(turn) + 1)]['checks']['reasoning'] = pass_reasoning
    
                with open(f"judge-{experiment_id}.json", 'w') as file:
                    json.dump(judge, file, indent=4, ensure_ascii=False)
    
        with open('config.txt', 'a') as file:
            file.write(f'{experiment_id}\t{venue}\t{arg_model_name}\t{prompt_type}\t{type_of_labels}\t{adversarial}\to3-mini\t{temperature}\t{max_tokens}\t{seed}\t{device_name}\n')

        accelerator.free_memory()
        del accelerator
        del rap_model
        del arg_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
