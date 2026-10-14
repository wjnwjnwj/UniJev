"""Resumable immutable dev/test evaluation; raw predictions retained per run."""
import argparse
import json
import os
import time
from pathlib import Path
from .configuration import load_config,sha256
from .runtime import write_json
from .report import export
from .schema import read_jsonl

def paths(c,suite,initial=False):
    root=Path(c['output'])
    return root.parent.parent/'evaluations'/'initial_softlabel_2b'/suite if initial else root/'evaluation'/suite

def validate(c,suite,initial=False):
    out=paths(c,suite,initial)
    proof=json.loads((out/'completion.json').read_text())
    for name,digest in proof['files'].items():
        if sha256(out/name)!=digest:
            raise ValueError('Evaluation payload changed')
    data=c['dev_data'] if suite=='dev' else c['final_eval_data']
    checkpoint=Path(c['init_checkpoint']) if initial else Path(c['output'])/'step-00500'
    report=json.loads((out/'comparison_metrics.json').read_text())
    if report['data_sha256']!=sha256(data) or report['metadata']['checkpoint_sha256']!=sha256(checkpoint/'checkpoint.pt'):
        raise ValueError('Evaluation input changed')
    from .report import summarize_predictions
    rows=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()]
    counts={'text':764} if suite=='dev' else c['expected_eval_counts']
    if report['metrics']!=summarize_predictions(rows,read_jsonl(data),counts):
        raise ValueError('Metric recomputation mismatch')
    return report

def run(c,suite,initial=False):
    import torch
    from .modeling import build_model,build_compiler
    from .runtime import load_checkpoint,load_initial_checkpoint,provenance
    out=paths(c,suite,initial)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():
        validate(c,suite,initial)
        return
    torch.cuda.set_device(0)
    cap=int(os.environ.get('WJN_EVAL_MEMORY_MIB','24576'))
    torch.cuda.set_per_process_memory_fraction(cap*1024**2/torch.cuda.get_device_properties(0).total_memory)
    checkpoint=Path(c['init_checkpoint']) if initial else Path(c['output'])/'step-00500'
    if not initial and not (Path(c['output'])/'finished.json').is_file():
        raise ValueError('Final checkpoint not complete')
    data=c['dev_data'] if suite=='dev' else c['final_eval_data']
    fingerprint={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint/'checkpoint.pt'),
        'data_sha256':sha256(data),'suite':suite,'source_sha256':provenance()['source_sha256'],
        'config':c,'initial':initial}
    manifest=out/'run.json'
    if manifest.exists() and json.loads(manifest.read_text())!=fingerprint:
        raise ValueError('Evaluation resume fingerprint mismatch')
    write_json(manifest,fingerprint)
    torch.manual_seed(c['seed'])
    model=build_model(dict(c,device='cuda:0',gradient_checkpointing=False))
    (load_initial_checkpoint if initial else load_checkpoint)(model,checkpoint)
    model.eval()
    compiler=build_compiler(c,Path(c['media_root']))
    records=read_jsonl(data)
    prediction=out/'predictions.jsonl'
    seen=set()
    if prediction.exists():
        raw=prediction.read_bytes()
        if raw and not raw.endswith(b'\n'):
            # Preserve the torn write for audit; resume only fully written rows.
            (out/f'torn-{time.time_ns()}.bin').write_bytes(raw[raw.rfind(b'\n')+1:])
            prediction.write_bytes(raw[:raw.rfind(b'\n')+1])
        for line in prediction.read_text().splitlines():
            row=json.loads(line)
            key=(row['record_index'],row['qid'])
            if key in seen:
                raise ValueError('Duplicate resumed prediction')
            seen.add(key)
    with prediction.open('a',buffering=1) as stream,torch.no_grad():
        for index,record in enumerate(records):
            if all((index,qid) in seen for qid in record['request']['questions']):
                continue
            compiled=compiler.compile(record,labeled_only=True)
            for q in compiled.questions:
                if (index,q.qid) in seen:
                    continue
                p=model(q).float().softmax(-1).cpu().tolist()
                row={'record_index':index,'group_id':record['group_id'],'qid':q.qid,
                     'task':q.kind,'modality':compiled.modality,'target':dict(zip(q.keys,q.target)),
                     'probabilities':dict(zip(q.keys,p))}
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if index%20==0:
                print(json.dumps({'event':'eval_progress','suite':suite,'index':index,'total':len(records)}),flush=True)
    export(prediction,data,out,{'text':764} if suite=='dev' else c['expected_eval_counts'],
           metadata=dict(fingerprint,latency_table_eligible=False))
    names=['predictions.jsonl','comparison_metrics.json','comparison_metrics.csv','run.json']
    write_json(out/'completion.json',{'files':{n:sha256(out/n) for n in names},'finished_at':time.time()})
    validate(c,suite,initial)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--suite',choices=['dev','main'],required=True)
    p.add_argument('--initial',action='store_true')
    a=p.parse_args()
    run(load_config(a.config),a.suite,a.initial)
if __name__=='__main__':
    main()

