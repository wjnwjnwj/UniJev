"""Validate full-suite predictions and export metrics; never edit paper tables."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from .configuration import sha256
from .schema import read_jsonl, candidates, target_distribution


def modality(record):
    state = record['request']['state']
    kinds = set()
    if isinstance(state, dict):
        for message in state['messages']:
            if isinstance(message['content'], list):
                kinds.update(item['type'] for item in message['content'] if item['type'] in {'image_url', 'video_url'})
    return 'mixed' if len(kinds) > 1 else 'image' if 'image_url' in kinds else 'video' if 'video_url' in kinds else 'text'


def expected_questions(records):
    expected = {}
    for index, record in enumerate(records):
        for qid, question in record['request']['questions'].items():
            keys = [k for k, _ in candidates(question)]
            target = target_distribution(record.get('targets', {}).get(qid), keys)
            if target is not None:
                if target.count(1.0) != 1 or any(v not in (0, 1) for v in target):
                    raise ValueError('Full-suite metrics require one-hot targets')
                expected[(index, qid)] = {'group_id': record['group_id'], 'modality': modality(record),
                                          'target': dict(zip(keys, target)), 'task': question['type']}
    if not expected:
        raise ValueError('No labeled evaluation questions')
    return expected


def validate_predictions(rows, records, expected_counts=None):
    expected = expected_questions(records)
    seen, clean = set(), []
    for row in rows:
        if isinstance(row.get('record_index'), bool) or not isinstance(row.get('record_index'), int):
            raise ValueError('record_index must be an integer')
        key = (row['record_index'], row.get('qid'))
        if key in seen or key not in expected:
            raise ValueError(f'Duplicate or unexpected prediction: {key}')
        seen.add(key)
        truth = expected[key]
        for field in ('group_id', 'modality', 'target', 'task'):
            if row.get(field) != truth[field]:
                raise ValueError(f'Mismatched {field} for {key}')
        probs = row.get('probabilities', {})
        if set(probs) != set(truth['target']):
            raise ValueError(f'Candidate mismatch for {key}')
        values = list(probs.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError(f'Invalid probability distribution for {key}')
        # Canonical candidate and record order gives stable ties in argmax/AURC.
        probs = {k: probs[k] for k in truth['target']}
        answer = max(probs, key=probs.get)
        correct = max(truth['target'], key=truth['target'].get)
        clean.append(dict(row, probabilities=probs, answer=answer,
                          metrics={'accuracy': float(answer == correct)}))
    missing = set(expected) - seen
    if missing:
        raise ValueError(f'Incomplete evaluation: missing {len(missing)} of {len(expected)} questions')
    clean.sort(key=lambda row: (row['record_index'], str(row['qid'])))
    counts = dict(Counter(row['modality'] for row in clean))
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f'Modality counts mismatch: {counts} != {expected_counts}')
    return clean


def metrics(rows, bins=15):
    # Same conventions as compare_rlcd_runs.metrics / risk_coverage.aurc:
    # 15 equal-width ECE bins, NLL floor 1e-12, trapezoidal risk from (0,0).
    count = len(rows)
    pairs, nll, brier = [], 0.0, 0.0
    for row in rows:
        p, t = row['probabilities'], row['target']
        y = max(t, key=t.get)
        pairs.append((max(p.values()), float(max(p, key=p.get) == y)))
        nll -= math.log(max(p[y], 1e-12))
        brier += sum((p[k] - t[k]) ** 2 for k in p)
    ece = 0.0
    for b in range(bins):
        bucket = [(c, h) for c, h in pairs if b / bins < c <= (b + 1) / bins]
        if bucket:
            ece += abs(sum(h - c for c, h in bucket)) / count
    hits, previous, area = 0, 0.0, 0.0
    for i, (_, hit) in enumerate(sorted(pairs, key=lambda pair: -pair[0]), 1):
        hits += hit
        risk = 1 - hits / i
        area += (risk + previous) / (2 * count)
        previous = risk
    return {'questions': count, 'acc': sum(h for _, h in pairs) / count,
            'conf': sum(c for c, _ in pairs) / count, 'ece': ece,
            'nll': nll / count, 'brier': brier / count, 'aurc': area}


def summarize_predictions(rows, records, expected_counts=None):
    clean = validate_predictions(rows, records, expected_counts)
    summary = {'all': metrics(clean)}
    for name in sorted({row['modality'] for row in clean}):
        summary[name] = metrics([row for row in clean if row['modality'] == name])
    return summary


def export(predictions, data, output, expected_counts=None, metadata=None):
    rows = [json.loads(line) for line in Path(predictions).read_text().splitlines() if line.strip()]
    report = {'data': str(data), 'data_sha256': sha256(data),
              'predictions': str(predictions), 'predictions_sha256': sha256(predictions),
              'metadata': metadata or {},
              'metrics': summarize_predictions(rows, read_jsonl(data), expected_counts)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison_metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with (output / 'comparison_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['modality', 'questions', 'acc', 'conf', 'ece', 'nll', 'brier', 'aurc'])
        writer.writeheader()
        for name, result in report['metrics'].items():
            writer.writerow(dict(modality=name, **result))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.predictions, args.data, args.output), indent=2))


if __name__ == '__main__':
    main()
