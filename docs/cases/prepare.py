"""Export the README's recorded paper examples; no model inference is run."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEST = Path(__file__).resolve().parent
CASE_IDS = ['text_correct', 'image_correct', 'video_gallery_6', 'image_disagreement']
IMAGE_NAMES = {'image_correct': 'image-counting.jpg', 'image_disagreement': 'image-chart-error.png'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(paper):
    snapshot_path = paper / 'figures/appendix_data.json'
    snapshot = json.loads(snapshot_path.read_text())
    originals = {case['slug']: case for case in snapshot['cases']}
    records = []
    for case_id in CASE_IDS:
        original = originals[case_id]
        record = original['record']
        qid = original['qid']
        question = record['request']['questions'][qid]
        metadata = record.get('meta', {})
        item = {
            'id': case_id,
            'paper_case_id': original['display_id'],
            'record_id': original['record_id'],
            'modality': original['modality'],
            'evaluation_split': original['evaluation_split'],
            'source_dataset': metadata.get('source_repo', metadata.get('domain')),
            'upstream_source_split': metadata.get('source_split'),
            'upstream_terms': metadata.get('license', 'See upstream dataset terms'),
            'question_id': qid,
            'context': original.get('display_state') if isinstance(original.get('display_state'), str) else '',
            'question': question['instructions'],
            'candidates': question['criteria'],
            'target_probabilities': record['targets'][qid]['probabilities'],
            'predictions': {},
            'media': [],
        }
        for method in ['SFT', 'CORA']:
            prediction = original['predictions'][method]
            item['predictions'][method] = {
                'selected_answer': prediction['answer'],
                'candidate_probabilities': prediction['probabilities'],
                'accuracy': prediction['metrics']['accuracy'],
                'confidence': prediction['metrics']['confidence'],
            }
        for media in original['media']:
            source = paper / 'figures' / media['file']
            digest = sha256(source)
            if digest != media['sha256']:
                raise ValueError(f'Source frame hash mismatch: {source.name}')
            item['media'].append({
                'paper_file': 'figures/' + media['file'],
                'source_sha256': digest,
                'timestamp_seconds': media.get('timestamp_seconds'),
            })
        if case_id in IMAGE_NAMES:
            destination = DEST / IMAGE_NAMES[case_id]
            shutil.copyfile(paper / 'figures' / original['media'][0]['file'], destination)
            item['preview'] = destination.name
        records.append(item)

    video = originals['video_gallery_6']
    frames = []
    durations = []
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    for i, media in enumerate(video['media']):
        with Image.open(paper / 'figures' / media['file']) as source:
            source = source.convert('RGB')
            source.thumbnail((640, 352), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (640, 396), '#24392e')
            canvas.paste(source, ((640-source.width)//2, (352-source.height)//2))
        draw = ImageDraw.Draw(canvas)
        timestamp = media['timestamp_seconds']
        draw.text((16, 365), f"Sampled frame {i+1}/6", font=font, fill='#edf5e9')
        draw.text((340, 365), f'Source time: {timestamp:06.2f} s', font=font, fill='#edf5e9')
        frames.append(canvas)
        interval = video['media'][i+1]['timestamp_seconds'] - timestamp if i+1 < len(video['media']) else 1.0
        durations.append(max(10, round(interval*100)*10))
    indexed = [frame.quantize(colors=128) for frame in frames]
    indexed[0].save(DEST / 'video-scene.gif', save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, disposal=2, optimize=False)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        concat = []
        for i, (frame, duration) in enumerate(zip(frames, durations)):
            frame.save(temp / f'{i}.png')
            concat.extend([f"file '{i}.png'", f'duration {duration/1000:.3f}'])
        concat.append(f"file '{len(frames)-1}.png'")
        (temp / 'frames.txt').write_text('\n'.join(concat))
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(temp / 'frames.txt'),
            '-c:v', 'libx264', '-crf', '25', '-preset', 'fast', '-r', '25',
            '-pix_fmt', 'yuv420p', '-t', f'{sum(durations)/1000:.3f}',
            '-movflags', '+faststart', str(DEST / 'video-scene.mp4'),
        ], check=True)
    next(r for r in records if r['id']=='video_gallery_6')['preview'] = 'video-scene.gif'
    outputs = []
    for name in [*IMAGE_NAMES.values(), 'video-scene.gif', 'video-scene.mp4']:
        path = DEST / name
        outputs.append({'file': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    result = {
        'source': 'newpaper/figures/appendix_data.json',
        'source_sha256': sha256(snapshot_path),
        'model_scale': '2B',
        'models': {'SFT': 'Soft-label SFT initialization', 'CORA': '100 supervised calibration updates'},
        'video_transform': 'Six original sampled frames with timestamp captions, source intervals rounded to 10 ms; final frame held for 1 s; no audio or interpolated motion.',
        'assets': outputs,
        'cases': records,
    }
    (DEST / 'predictions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(f'Exported {len(records)} recorded cases and {len(outputs)} media assets.')
    for item in outputs:
        print(f"{item['file']}: {item['bytes']/1024:.1f} KiB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=DEST.parents[2] / 'newpaper')
    args = parser.parse_args()
    export(args.paper_dir.resolve())
