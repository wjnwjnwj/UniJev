# README case sources

The gallery uses four recorded evaluation cases from the latest UniJev manuscript (`newpaper/figures/appendix_data.json`), with the paper's original media. It compares the local **2B soft-label SFT initialization** and **CORA (2B), after 100 calibration updates**. No new inference is run and no probabilities are synthesized.

| README example | Paper case | Source dataset | Evaluation split |
| --- | --- | --- | --- |
| Text: science question | T1 / `text_correct` | [SciQ](https://huggingface.co/datasets/allenai/sciq), `sciq/test/178` | `text_test764` |
| Image: bowl counting | I3 / `image_correct` | [VQAv2](https://huggingface.co/datasets/Multimodal-Fatima/VQAv2_train), `vqav2:132654000` | `main3124` |
| Video: scene recognition | V6 / `video_gallery_6` | NExT-QA via the OneThinker video mixture, record `9020076077` | `main3124` |
| Image: remaining chart error | I7 / `image_disagreement` | [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), record ending in `:4607` | `main3124` |

`main3124` is the paper's held-out trimodal test suite; `text_test764` is its text evaluation. An upstream dataset's original split name is retained in the snapshot and is separate from the paper's source-group train/test partition.

[predictions.json](predictions.json) preserves the exact candidate probabilities, selected answers, correctness labels, original record identifiers, input hashes, and output asset hashes. README tables display those probabilities as percentages rounded to one decimal place; rounded entries may not sum to exactly 100%.

## Media

- `image-counting.jpg` and `image-chart-error.png` are byte-for-byte copies of the corresponding images in the latest paper.
- `video-scene.gif` and `video-scene.mp4` replay the six frames supplied for paper case V6. Each frame includes its original source timestamp. Inter-frame durations follow the source timestamps, rounded to 10 ms; the final frame is held for one second so it remains readable. No audio or interpolated motion is added.
- The GIF renders inline in GitHub Markdown. The MP4 is provided as a relative file link, without relying on an HTML video player or a temporary website URL.

Media retain their upstream provenance and terms. This gallery does not assign a new license to the source datasets. It illustrates both successful predictions and a remaining error; aggregate calibration should be assessed using the full evaluation results.

## Regenerate the assets

The gallery is self-contained for readers. To update it from a new paper snapshot, run the exporter with Python 3, Pillow, and FFmpeg installed:

```bash
python docs/cases/prepare.py --paper-dir /path/to/newpaper
```

The exporter validates source image hashes, writes the four gallery assets and the probability snapshot, and does not modify training data or weights. After regeneration, update the README tables to match the new saved probabilities.
