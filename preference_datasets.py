import json

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple,Any
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import load_from_disk


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_stack_exchange(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.

       https://hf-mirror.com/datasets/HuggingFaceH4/stack-exchange-preferences/tree/main/data/3dprinting.stackexchange.com
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    #dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    dataset_path = "/home/hkx/data/work/hf_data_and_model/datas/stack-exchange-preferences/"
    #dataset_path = "/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/datas/stack-exchange-preferences/data/3dprinting.meta.stackexchange.com/"
    num_proc = 5
    dataset = datasets.load_dataset(path=dataset_path, cache_dir=None, num_proc=num_proc)['train']
    #dataset = load_from_disk(dataset_path)
    print(f'load data done, dataset:{dataset}, shape:{len(dataset)}')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    # test只取前1%的数据，train取剩下的99%的数据
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=num_proc)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i)) # 实际发现有些answer的分数相同

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs # 所遥answer两两组合，num = C(n,2)
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)]) # 获取score最高的回答作为sft的结果

    """
    {
    "responses":
    [
        " The problem is in the design of your bed. Let's start from the basic setup of a glass bed:\n\nThe heater element is usually mounted to a metal carrier, which is both spreading the thermal energy over the bed, but also is the structural element that is leveled against the carriage. Atop that comes the glass print surface.\n\nNow, once the heater element is turned on, the aluminium starts to expand and evens the distribution to the glass. As the glass has a much lower thermal expansion coefficient, it doesn't expand as fast. Because of this, the glass surface should  be glued to the bed or heater but held in position to the metal bed with a clip. This way the thermal and mechanical stress on the glass sheet is mitigated: The metal bed evens the heat transfer and the clip can move its position on the glass.",
        " I would be careful before trying another glass just hoping it will go better, since you haven't found the issue.\n\nI have a PCB heated bed in direct contact (PCB copper traces on top) a 2 mm glass (plain float glass, not hardened and not borosilicate). It never broke and I've been using it intensely for the last few months. My heated bed is very flat (even if it bends with the heat) and also clean: no residues which can push against the glass. Clean yours properly!\n\nAlso, how powerful is your heated bed? mine is about 120 W for 12x12 cm. If yours is too powerful, maybe you could slow down the heating by reducing the maximum duty cycle (you need maybe to recompile Marlin) or by increasing the temperature 10 °C at time. \n\nI also see that you use mirrors, maybe recovered from other applications. I bought the glass new, which is very cheap but it is also guaranteed defect free. Maybe yours had issues already.",
        " As manufactuer and 3D printing's fans, I think it's better to use custom tempered glass. It will be nice and flat and stiff. It's also easy to clean and holds up well. You can print on the bare glass with many materials or use various preparations like PVA (glue stick or white glue diluted with water are popular), hairspray, or others."
    ],
    "pairs": # 三个回答的顺序：0>1>2
    [
        [
            0,
            1
        ],
        [
            0,
            2
        ],
        [
            2,
            1
        ]
    ],
    "sft_target": " The problem is in the design of your bed. Let's start from the basic setup of a glass bed:\n\nThe heater element is usually mounted to a metal carrier, which is both spreading the thermal energy over the bed, but also is the structural element that is leveled against the carriage. Atop that comes the glass print surface.\n\nNow, once the heater element is turned on, the aluminium starts to expand and evens the distribution to the glass. As the glass has a much lower thermal expansion coefficient, it doesn't expand as fast. Because of this, the glass surface should  be glued to the bed or heater but held in position to the metal bed with a clip. This way the thermal and mechanical stress on the glass sheet is mitigated: The metal bed evens the heat transfer and the clip can move its position on the glass."
    }
    """

    print(f"converted sample data, prompt:{prompt} \nresult:\n{json.dumps(data[prompt], ensure_ascii=False)}")
    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_dolly(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading Dolly dataset ({split} split)...')
    # dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    dataset_path = '/research/cbim/medical/lh599/code/rl_4_llm/results/experiment_toxic/toxicplus_7b_100_t_0_d/data/experiment_toxic_neg_epoch_0_dataset_temp_0.7_0/'
    dataset = load_from_disk(dataset_path)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = ex['instruction'] + ex['context']
        prompt = f'\n\nHuman: {prompt}\n\nAssistant:'
        chosen_response = ex['response']
        rejected_response = ex['generated_response'].replace('<|endoftext|>', '')
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing Dolly', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'stack_exchange':
        data = get_stack_exchange(split, silent=silent, cache_dir=cache_dir)
    elif name == 'dolly':
        data = get_dolly(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer:PreTrainedTokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

        将batch内的样本List[Dict] 转为 Dict[str, Union[List, torch.Tensor]],即将许多样本合并成tensor, 并padding
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch:List[Dict[str, Any]]):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(example[k][::-1]) for example in batch] # prompt是先按时间逆序翻转, 对batch内每条样本均进行转换
                else:
                    to_pad = [torch.LongTensor(example[k]) for example in batch]
                padding_value = 0 if k.endswith('_attention_mask') else tokenizer.pad_token_id
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value) # 将一个batch中的所有有input_ids pad到相同的长度,最多是max_len=512, 在右边补0
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1]) # 对于prompt是按时间翻转后再padding的，因此需要翻转回来
            else: # 其它非ids类的值直接copy,如原始string
                padded_batch[k] = [example[k] for example in batch]

        return padded_batch
    return collate_fn


def tokenize_chosen_rejected_pair(prompt: str, chosen_answer: str, rejected_answer: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
       该函数只处理一对正负样本，里面只有一个正样本以及一个负样本
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen_answer, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected_answer, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen_tokens}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected_tokens}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id) # answer.input_ids后添加eos
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length: # 截断prompt
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()} # 只保留answer的前半部分
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens} # 注意：已经添加了EOS
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:] # 注意：label与原始的input一样,在计算loss前才向左shift
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids']) # prompt部分不计算loss
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]# 注意：label与原始的input一样,在计算loss前才向左shift
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}
    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen_answer
    batch['rejected'] = prompt + rejected_answer
    batch['chosen_response_only'] = chosen_answer
    batch['rejected_response_only'] = rejected_answer

    prompt_answers = {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}
    for k, toks in prompt_answers.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer:PreTrainedTokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(low=0, high=2**32, size=1000000))
        flat_data_list = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir).items():
                flat_data_list.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(int(next(permutation_seeds))):
                random.shuffle(flat_data_list)

        batch:List[Dict[str, Any]] = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data_list:
            if done:
                break
            if sft_mode: # 生成sft数据
                pos_neg_pair = tokenize_chosen_rejected_pair(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                pos_neg_pair = {k: v for k, v in pos_neg_pair.items() if 'rejected' not in k} # 不要rejected的数据
                batch.append(pos_neg_pair)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:# 生成dpo数据
                for p in pairs:
                    if done:
                        break
                    pos_neg_pair = tokenize_chosen_rejected_pair(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(pos_neg_pair) # 将一对正负样本组装成batch数据
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True

                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True

def tests():
    get_stack_exchange("train")

if __name__ == '__main__':
    tests()

