"""Immutable campaign configuration and path checks."""
import hashlib
import json
from pathlib import Path
from . import VERSION, METHOD
ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
KINDS = {'analytic', 'sft_soft', 'shaped', 'grpo', 'ctr', 'rlcd_lite', 'rewarding_doubt'}

def load_config(path, output=None):
    c = json.loads(Path(path).read_text())
    if output is not None:
        c['output'] = output
    if c['version'] != VERSION or c['method'] != METHOD or c['objective'] not in KINDS:
        raise ValueError('Unsupported training recipe')
    for k in ('model_path','data','init_checkpoint','final_eval_data','dev_data','media_root','output'):
        if c.get(k) is not None:
            p = Path(c[k])
            c[k] = str((p if p.is_absolute() else ROOT/p).resolve())
    if c['max_steps'] != 500 or c['save_every'] != 50:
        raise ValueError('Campaign budget is exactly 500 steps, checkpoints every 50')
    expected_world = 8 if c['objective'] in {'analytic','sft_soft'} else 4
    if c['world_size'] != expected_world:
        raise ValueError('Incorrect GPU count for recipe')
    if c['global_tokens_per_step'] != 32768 or c['gradient_reduction'] != 'global_question_mean':
        raise ValueError('Nonstandard optimizer budget')
    if c['data'] in (c['dev_data'],c['final_eval_data']):
        raise ValueError('Train/evaluation leakage')
    if c['objective'] != 'sft_soft' and not c['init_checkpoint']:
        raise ValueError('Continuation requires soft SFT initialization')
    if c['actor_mode'] not in {'sft_adapters','head'} or not 0 <= c['label_smoothing'] < 1:
        raise ValueError('Invalid actor or smoothing')
    return c

def check_paths(c):
    paths = [Path(c['model_path'])/'config.json', Path(c['data']), Path(c['dev_data']), Path(c['final_eval_data'])]
    if c['init_checkpoint']:
        paths += [Path(c['init_checkpoint'])/'checkpoint.pt',Path(c['init_checkpoint'])/'config.json']
        init = json.loads(paths[-1].read_text())
        for k in ('stage','projection_dim','lora_rank','lora_alpha'):
            if c[k] != init[k]:
                raise ValueError('Initialization architecture mismatch: '+k)
        p=Path(init['model_path'])
        if (p if p.is_absolute() else ROOT/p).resolve()!=Path(c['model_path']):
            raise ValueError('Initialization backbone mismatch')
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    return {'config':c,'inputs_exist':True}

