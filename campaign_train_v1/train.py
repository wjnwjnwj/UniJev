"""Analytic distribution-policy training. --check-data validates inputs on CPU only."""
import argparse
import json
from pathlib import Path
from .configuration import load_config, check_paths, sha256
from . import METHOD, VERSION


def validate_data(config):
    from .schema import read_jsonl
    from .report import expected_questions
    from collections import Counter
    import hashlib

    records = read_jsonl(config["data"])
    evaluation = read_jsonl(config["final_eval_data"])
    overlap = {r["group_id"] for r in records} & {r["group_id"] for r in evaluation}
    if overlap:
        raise ValueError(f"Train/evaluation group leakage: {len(overlap)} groups")

    def fingerprint(record):
        return hashlib.sha256(json.dumps(record["request"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    # Also reject identical requests with renamed group IDs.
    shared = {fingerprint(r) for r in records} & {fingerprint(r) for r in evaluation}
    if shared:
        raise ValueError(f"Train/evaluation request leakage: {len(shared)} requests")
    expected = expected_questions(records)
    for key, item in expected.items():
        if len(item["target"]) < 2:
            raise ValueError(f"Training question needs >=2 candidates: {key}")
    evaluation_questions = expected_questions(evaluation)
    counts = dict(Counter(item["modality"] for item in evaluation_questions.values()))
    if config.get("expected_eval_counts") is not None and counts != config["expected_eval_counts"]:
        raise ValueError(f"Evaluation modality counts mismatch: {counts}")
    init = json.loads((Path(config["init_checkpoint"]) / "config.json").read_text()) if config['init_checkpoint'] else {}
    from .configuration import ROOT
    original = Path(init["data"]) if init.get("data") else None
    original = (original if original.is_absolute() else ROOT / original).resolve() if original else None
    sft_overlap = None
    if original is not None and original.is_file():
        if original == Path(config["data"]).resolve():
            sft_overlap = len({r["group_id"] for r in records})
        else:
            sft_groups = {r["group_id"] for r in read_jsonl(original)}
            sft_overlap = len(sft_groups & {r["group_id"] for r in records})
    return records, {"train_records": len(records), "train_questions": len(expected),
                     "eval_records": len(evaluation), "eval_questions": len(evaluation_questions),
                     "eval_modality_counts": counts, "train_eval_group_overlap": 0,
                     "train_eval_request_overlap": 0, "sft_train_group_overlap": sft_overlap,
                     "independent_of_sft_training": sft_overlap == 0 if sft_overlap is not None else None}


def run(config):
    import random
    import time
    import torch
    from .modeling import build_model, build_compiler, optimizer_groups
    from .objectives import compute
    from .runtime import Context, load_initial_checkpoint, load_checkpoint, save_checkpoint, write_json, provenance, snapshot_sources
    context = Context(config['device'])
    try:
        config = dict(config, device=context.device)
        if context.world_size != config['world_size']:
            raise ValueError('World size differs from the frozen recipe')
        if context.device.startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(
                config['memory_mib']*1024**2/torch.cuda.get_device_properties(context.device).total_memory, context.device)
        output = Path(config['output'])
        resume = output / 'latest'
        resuming = resume.is_symlink()
        if (output/'finished.json').exists():
            raise FileExistsError('Run already completed')
        if not resuming:
            context.on_primary(lambda: output.mkdir(parents=True, exist_ok=False))
        records, data_audit = validate_data(config)
        _, dev_audit = validate_data(dict(config, final_eval_data=config['dev_data'], expected_eval_counts={'text':764}))
        data_audit['development'] = dev_audit
        torch.manual_seed(config['seed'])
        model = build_model(config)
        initial_progress = load_initial_checkpoint(model, config['init_checkpoint']) if config['init_checkpoint'] else None
        groups = optimizer_groups(model, config)
        parameters = [p for group in groups for p in group['params']]
        context.broadcast(parameters)
        compiler = build_compiler(config, Path(config['media_root']))
        optimizer = torch.optim.AdamW([{'params': g['params'], 'lr': g['lr']} for g in groups], weight_decay=config['weight_decay'])
        shuffle = random.Random(config['seed'])
        manifest = {'version': VERSION, 'method': METHOD, 'config': config,
                    'world_size': context.world_size, 'initial_progress': initial_progress,
                    'data_audit': data_audit, 'provenance': provenance(),
                    'input_sha256': {key: sha256(config[key]) for key in ('data', 'final_eval_data','dev_data')},
                    'init_sha256': sha256(Path(config['init_checkpoint']) / 'checkpoint.pt') if config['init_checkpoint'] else None,
                    'total_parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': {g['name']: sum(p.numel() for p in g['params']) for g in groups},
                    'optimizer_groups': [{'name': g['name'], 'lr': g['lr']} for g in groups]}
        if not resuming:
            context.on_primary(lambda: snapshot_sources(output))
            context.on_primary(lambda: write_json(output / 'config.json', config))
            context.on_primary(lambda: write_json(output / 'run_manifest.json', manifest))
        else:
            previous = json.loads((output/'run_manifest.json').read_text())
            for key in ('input_sha256','init_sha256','world_size'):
                if previous[key] != manifest[key]:
                    raise ValueError('Resume input changed: '+key)
            if previous['provenance']['source_sha256'] != manifest['provenance']['source_sha256']:
                raise ValueError('Resume source changed')
            if {k:v for k,v in previous['config'].items() if k!='device'} != {k:v for k,v in config.items() if k!='device'}:
                raise ValueError('Resume configuration changed')
        print(json.dumps({"event": "data_audit", **data_audit}), flush=True)
        model.train()
        step, epoch, pending_tokens, local_questions = 0, 0, 0, 0
        total_tokens, total_questions, stats_sum = 0, 0, {}
        started = time.monotonic()
        last_saved = -1
        resume_offset = 0
        if resuming:
            progress = load_checkpoint(model,resume)
            saved = torch.load(resume/'training_state.pt',map_location='cpu',weights_only=False)
            if not saved['resume_supported'] or saved['progress'] != progress:
                raise ValueError('Invalid resume state')
            optimizer.load_state_dict(saved['optimizer'])
            step, epoch = progress['step'],progress['epoch']
            total_tokens,total_questions = progress['total_tokens'],progress['total_questions']
            rs = saved['rank_states'][context.rank]
            resume_offset = rs['next_record_offset']
            # Replay only cheap Python shuffles; no model/data replay.
            for _ in range(epoch):
                dummy = list(range(len(records)))
                shuffle.shuffle(dummy)
            torch.set_rng_state(rs['torch_rng'])
            if context.device.startswith('cuda'):
                torch.cuda.set_rng_state(rs['cuda_rng'])
            last_saved = step
        else:
            # Identical model initialization, independent per-rank action samples.
            torch.manual_seed(config['seed'] + context.rank)
        # Reference-model anchors belong to the excluded campaign recipes; SFT trains without one.
        anchor = None
        while step < config['max_steps']:
            order = list(range(len(records)))
            shuffle.shuffle(order)
            # No duplicate padding: idle ranks participate in the same collectives.
            for offset in range(0, len(order), context.world_size):
                if offset < resume_offset:
                    continue
                position = offset + context.rank
                round_tokens = 0
                if position < len(order):
                    index = order[position]
                    candidate_rng = random.Random(config['seed'] + epoch * len(records) + index)
                    compiled = compiler.compile(records[index], rng=candidate_rng, labeled_only=True)
                    round_tokens = compiled.compute_tokens
                    for question in compiled.questions:
                        objective, stats = compute(model, anchor, compiler, records[index], question, config, step)
                        if not torch.isfinite(objective):
                            raise FloatingPointError(f'Nonfinite loss at record {index}')
                        objective.backward()
                        stats['loss'] = float(objective.detach())
                        for key, value in stats.items():
                            stats_sum[key] = stats_sum.get(key, 0.0) + value
                        local_questions += 1
                pending_tokens += int(context.sum(round_tokens))
                if pending_tokens < config['global_tokens_per_step']:
                    continue
                questions = int(context.sum(local_questions))
                context.mean_gradients(parameters, questions)
                norm = torch.nn.utils.clip_grad_norm_(parameters, config['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                model.project_parameters()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                total_tokens += pending_tokens
                total_questions += questions
                # Every rank uses the same metric names, even with an idle last rank.
                all_stats = context.gather(stats_sum)
                means = {key: sum(s.get(key, 0.0) for s in all_stats) / questions
                         for key in sorted(set().union(*(s.keys() for s in all_stats)))}
                entry = {'step': step, 'epoch': epoch, 'global_tokens': pending_tokens,
                         'global_questions': questions, 'gradient_norm': float(norm),
                         'total_tokens': total_tokens, 'total_questions': total_questions,
                         'elapsed_seconds': time.monotonic() - started, 'temperature': float(model.temperature.detach()), **means}
                if context.primary:
                    with (output / 'training_log.jsonl').open('a') as stream:
                        stream.write(json.dumps(entry, allow_nan=False) + '\n')
                    print(json.dumps(entry), flush=True)
                pending_tokens, local_questions, stats_sum = 0, 0, {}
                if step % config['save_every'] == 0 or step == config['max_steps']:
                    progress = {'step': step, 'epoch': epoch, 'total_tokens': total_tokens, 'total_questions': total_questions}
                    rank_states = context.gather({
                        "rank": context.rank, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if context.device.startswith("cuda") else None,
                        "shuffle_rng": shuffle.getstate(), "next_record_offset": offset + context.world_size})
                    context.on_primary(lambda: save_checkpoint(
                        output / f'step-{step:05d}', model, config, progress, optimizer, rank_states))
                    last_saved = step
                if step == config['max_steps']:
                    break
            epoch += 1
            resume_offset = 0
        if last_saved != config['max_steps']:
            raise RuntimeError('Fixed final checkpoint was not saved')
        context.on_primary(lambda: write_json(output / 'finished.json', {'steps': step, 'final_checkpoint': f'step-{step:05d}',
                                                                        'elapsed_seconds': time.monotonic() - started}))
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--check', action='store_true', help='Validate config and local paths without loading a model')
    parser.add_argument('--check-data', action='store_true', help='Also inspect full JSONL labels and split groups on CPU')
    args = parser.parse_args()
    config = load_config(args.config, args.output)
    checked = check_paths(config)
    if args.check or args.check_data:
        if args.check_data:
            _, checked['data_audit'] = validate_data(config)
        print(json.dumps(checked, indent=2))
        return
    run(config)


if __name__ == '__main__':
    main()
