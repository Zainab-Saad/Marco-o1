import os
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from transformers import AutoTokenizer

from tree_search.mcts_nodes import *
from tree_search.utils import snake_to_pascal
from tree_search.utils.mcts_engine import MCTS_Engine
from tree_search.utils.model_IO.vllm_server_API import set_server

from tree_search.config import DataConfig
from tree_search.data import load_math, format_prompt
from tree_search.prompts import SYSTEM_COT_PROMPT, MATH_COT_PROMPT

args = argparse.ArgumentParser()
args.add_argument('--config', type=str, default='./tree_search/configs/math_config.json')
args.add_argument('--level', type=str, required=True, help='MATH level 1 to 5')
args.add_argument('--start', type=int, default=None)
args.add_argument('--end', type=int, default=None)
args.add_argument('--num-problems', type=int, default=None, help='limit for debug mode for testing')
args.add_argument('--workers', type=int, default=16, help='problems searched concurrently')
args.add_argument('--server-url', type=str, default='http://127.0.0.1:40000/v1/completions')
args.add_argument('--max-runtime-min', type=float, default=None, help='exit 99 with work remaining')
args = args.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_config_to_node_class(config):
    action_tree = {}
    for action_name, action_info in config['action_tree'].items():
        action_class = globals()[f'{snake_to_pascal(action_name)}Node']
        action_class.prefill_text = action_info['prefill_text']
        action_class.description = action_info['description']
        action_class.show_in_history = action_info['show_in_history']
        action_class.use_special_model = action_info['special_model']
        action_tree[action_name] = action_info['next_step']
    args.action_tree = action_tree

    for key in ['mode', 'max_rollout_time', 'max_tokens', 'output_tree', 'evaluate_func',
                'search_reward_threshold', 'use_for_wrong_answer', 'generate_func',
                'use_step', 'use_mini_step', 'use_function_call',
                'max_generations', 'max_depth', 'max_chain_tokens',
                'temperature', 'top_p', 'top_k', 'seed', 'teacher_model', 'tag_instructions']:
        setattr(args, key, config[key])
    args.output_folder = f"{config['output_folder']}_L{args.level}"
    args.mask_asking = False
    args.use_multi_turn = args.use_function_call
    args.use_tag = not (args.use_step or args.use_mini_step)

def build_prompt(tokenizer, question, system_prompt):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': format_prompt(MATH_COT_PROMPT, question)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True)
    return text + '<think>\n'

def main():
    config = load_config(args.config)
    load_config_to_node_class(config)
    set_server(args.server_url, args.teacher_model)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    args.tokenizer = tokenizer
    system_prompt = SYSTEM_COT_PROMPT + '\n\n' + args.tag_instructions

    problems = load_math(DataConfig(dataset='math', split='train'), levels=[args.level])
    data = [{
        'id': f'L{args.level}_{idx:04d}',
        'problem_idx': idx,
        'level': p['level'],
        'subject': p['subject'],
        'problem': p['question'],
        'solution': p['gold'],
        'prompt': build_prompt(tokenizer, p['question'], system_prompt),
    } for idx, p in enumerate(problems)]

    lo = args.start if args.start is not None else 0
    hi = args.end if args.end is not None else len(data)
    data = data[lo:hi]
    if args.num_problems is not None:
        data = data[:args.num_problems]

    out_dir = f'./output/{args.output_folder}'
    os.makedirs(f'{out_dir}/tree_output', exist_ok=True)
    save_file = os.path.basename(args.config).replace('_config.json', '')
    index_path = f'{out_dir}/index_{lo}_{hi}.jsonl'
    lock = threading.Lock()
    deadline = time.monotonic() + args.max_runtime_min * 60 if args.max_runtime_min else None
    skipped = []

    def is_done(d):
        return os.path.exists(f'{out_dir}/tree_output/{save_file}_{d["id"]}.json')

    def run_one(d):
        if deadline is not None and time.monotonic() > deadline:
            skipped.append(d['id'])
            return
        t0 = time.time()
        try:
            mcts = MCTS_Engine(max_rollout_depth=args.max_rollout_time, generate_func=args.generate_func,
                                max_new_tokens=args.max_tokens, args=args)
            info = mcts.do_rollout(d)
        except Exception as e:
            print(f'[ERROR] {d["id"]}: {e!r}', flush=True)
            return
        rec = {k: d[k] for k in ('id', 'problem_idx', 'level', 'subject', 'problem', 'solution')}
        rec.update(info)
        rec['elapsed_s'] = round(time.time() - t0, 1)
        with lock:
            with open(index_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    todo = [d for d in data if not is_done(d)]
    print(f'level {args.level} shard [{lo},{hi}): {len(data)} problems, {len(todo)} remaining', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(run_one, todo), total=len(todo)))

    if skipped:
        print(f'[deadline] {len(skipped)} problems not started; resubmit to continue', flush=True)
        sys.exit(99)
    print('Done.')

if __name__ == '__main__':
    main()