"""Compile a record once per state; append one scoring suffix per question.

The state — any mix of text, images and video — is tokenized once through the
Qwen3.5 processor (text-only states skip the visual tower entirely).
Each question then appends a plain-text block:

    <|im_start|>user
    Task: choice
    Question: <instructions>
    Candidates:
    A: <description>
    ...
    Decision:

The decision head reads the hidden state at each candidate's last token and at
the final "Decision:" token. Score questions never see their level IDs and are
scored one level per branch, so no level ever attends to another.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json
import tarfile

import numpy as np
import torch

from .schema import candidates, target_distribution


def load_tar_video(path, max_frames=None):
    """Load an easyvideor1 frame-cache tar: JPEG frames + sampling metadata.

    Returns (frames ndarray [T,H,W,C], metadata dict for VideoMetadata) with at
    most max_frames uniformly chosen frames; timestamps stay exact because the
    original frame indices and fps travel along in the metadata.
    """
    with tarfile.open(path) as tar:
        meta = json.load(tar.extractfile("metadata.json"))
        if meta.get("format") != "easyvideor1_preprocessed_video":
            raise ValueError(f"Unknown cache format in {path}")
        count = meta["num_frames"]
        source = meta["metadata"]
        if max_frames and count > max_frames:
            chosen = np.linspace(0, count - 1, max_frames).round().astype(int).tolist()
        else:
            chosen = list(range(count))
        frames = []
        for index in chosen:
            data = tar.extractfile(f"frames/{index:06d}.jpg").read()
            from PIL import Image
            frames.append(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))
    metadata = {"total_num_frames": source["total_num_frames"], "fps": source["fps"],
                "frames_indices": [int(source["frames_indices"][i]) for i in chosen],
                "video_backend": "tar_cache"}
    return np.stack(frames), metadata


@dataclass
class Branch:
    inputs: dict
    candidate_positions: list
    decision_position: int
    length: int


@dataclass
class Question:
    qid: str
    kind: str
    keys: list
    descriptions: list
    branches: list
    target: list | None
    is_text: bool


@dataclass
class CompiledState:
    questions: list
    logical_tokens: int
    compute_tokens: int
    media: list
    modality: str = "text"  # text | image | video | mixed


class Compiler:
    def __init__(self, processor, media_root=".", max_length=32768, media_kwargs=None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.media_root = Path(media_root)
        self.max_length = max_length
        self.media_kwargs = media_kwargs or {}
        self.verify_hashes = True  # training audits assets once at startup and turns this off

    def _messages(self, state):
        messages = [{"role": "user", "content": state}] if isinstance(state, str) else deepcopy(state["messages"])
        media = []
        for message in messages:
            if message["role"] not in {"user", "assistant", "system"}:
                raise ValueError("Unsupported state role")
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            for item in message["content"]:
                kind = item["type"]
                if kind == "text":
                    # Prevent user text from injecting Qwen control tokens.
                    if any(t in item["text"] for t in self.tokenizer.all_special_tokens):
                        raise ValueError("Reserved tokenizer control token in state")
                    continue
                if kind not in {"image_url", "video_url"}:
                    raise ValueError(f"Unsupported content type: {kind}")
                url = item[kind]["url"]
                if "://" in url:
                    raise ValueError("Materialize remote media locally before training")
                path = Path(url)
                path = path if path.is_absolute() else self.media_root / path
                path = path.resolve(strict=True)
                if kind == "video_url" and path.suffix == ".tar":
                    # easyvideor1 frame cache: decode selected JPEGs, keep exact
                    # source timestamps via VideoMetadata instead of re-decoding
                    # the original mp4.
                    frames, metadata = load_tar_video(path, self.media_kwargs.get("max_frames"))
                    media.append({"path": str(path), "type": kind, "cache": True,
                                  "sampling": dict(metadata, sampled_frames=len(frames))})
                    item.clear()
                    item.update(type="video", video=frames)
                    item["__metadata__"] = metadata
                else:
                    media.append({"path": str(path), "type": kind})
                    item.clear()
                    item.update(type=kind.removesuffix("_url"), path=str(path))
        return messages, media

    def _encode_state(self, state):
        messages, media = self._messages(state)
        cache_metadata = []
        for message in messages:
            for item in message["content"] if isinstance(message["content"], list) else []:
                if "__metadata__" in item:
                    cache_metadata.append(item.pop("__metadata__"))
        kwargs = dict(self.media_kwargs, return_metadata=True)
        if cache_metadata:
            if len(cache_metadata) != sum(1 for m in media if m["type"] == "video_url"):
                raise ValueError("Mixing frame-cache tars and raw videos in one state is unsupported")
            from transformers.video_utils import VideoMetadata
            kwargs.update(do_sample_frames=False,
                          video_metadata=[VideoMetadata(**m) for m in cache_metadata])
            kwargs.pop("fps", None)  # frames are pre-sampled; fps would resample
        base = dict(self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", processor_kwargs=kwargs))
        # Frame sampling metadata is audit information, never a backbone kwarg.
        metadata = base.pop("video_metadata", None)
        if metadata is not None:
            for entry, meta in zip([m for m in media if m["type"] == "video_url"], metadata):
                entry["sampling"] = dict(meta) if isinstance(meta, dict) else vars(meta)
                sampling = entry["sampling"]
                indices = sampling.get("frames_indices")
                if indices is not None and sampling.get("fps"):
                    sampling["frames_indices"] = [int(i) for i in indices]
                    sampling["timestamps_seconds"] = [int(i) / sampling["fps"] for i in indices]
        for name in ("image_grid_thw", "video_grid_thw"):
            if name in base:
                grid = base[name]
                media.append({name: grid.tolist(), "visual_tokens": int(grid.prod(-1).sum()) // 4})
        return base, media

    def compile(self, record, rng=None, labeled_only=False):
        selected = []
        for qid, question in record["request"]["questions"].items():
            pairs = candidates(question)
            target = record.get("targets", {}).get(qid)
            distribution = target_distribution(target, [k for k, _ in pairs])
            if labeled_only and distribution is None:
                continue
            if rng is not None and question["type"] == "choice":
                pairs = list(pairs)
                rng.shuffle(pairs)
                distribution = target_distribution(target, [k for k, _ in pairs])
            selected.append((qid, question, pairs, distribution))
        if not selected:
            return CompiledState([], 0, 0, [])
        state = record["request"]["state"]
        base, media = self._encode_state(state)
        # Hash only assets the record explicitly pins; hashing every video at
        # scale would read terabytes for no verification value.
        expected = {str((self.media_root / a["path"]).resolve()): a["sha256"]
                    for a in record.get("assets", [])} if self.verify_hashes else {}
        for entry in media:
            path = entry.get("path")
            if path in expected:
                entry["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if expected[path] != entry["sha256"]:
                    raise ValueError(f"Media hash mismatch: {path}")
        base_length = base["input_ids"].shape[1]
        kinds = {m["type"] for m in media if "type" in m}
        if not kinds:
            modality = "text"
        elif kinds == {"image_url"}:
            modality = "image"
        elif kinds == {"video_url"}:
            modality = "video"
        else:
            modality = "mixed"
        is_text = modality == "text"
        logical_tokens, compute_tokens = base_length, 0
        compiled = []
        for qid, question, pairs, distribution in selected:
            keys = [k for k, _ in pairs]
            branches = []
            groups = [[pair] for pair in pairs] if question["type"] == "score" else [pairs]
            for group in groups:
                suffix = []

                def append(text):
                    suffix.extend(self.tokenizer.encode(text, add_special_tokens=False))

                for value in [question["instructions"]] + [v for pair in group for v in pair]:
                    if any(t in value for t in self.tokenizer.all_special_tokens):
                        raise ValueError("Reserved tokenizer control token in question/criteria")
                # Canonical segment tokenization makes endpoints exact; no string searching.
                append("<|im_start|>user\n")
                append("Task: " + question["type"] + "\nQuestion: " + question["instructions"] + "\nCandidates:\n")
                positions = []
                for key, description in group:
                    # Score never sees its level ID, nor any other level description.
                    append((key + ": " if question["type"] != "score" else "") + description)
                    positions.append(base_length + len(suffix) - 1)
                    append("\n")
                append("Decision:")
                length = base_length + len(suffix)
                if length > self.max_length:
                    raise ValueError(f"{qid}: {length} tokens exceed max_length={self.max_length}; no truncation")
                inputs = dict(base)
                extension = torch.tensor([suffix], dtype=base["input_ids"].dtype)
                inputs["input_ids"] = torch.cat([base["input_ids"], extension], dim=1)
                inputs["attention_mask"] = torch.cat(
                    [base["attention_mask"], torch.ones_like(extension)], dim=1)
                # Any other sequence-aligned integer tensor (e.g. mm_token_type_ids,
                # which 3D RoPE indexes) is padded with zeros: the suffix is text.
                for name, value in base.items():
                    if name in ("input_ids", "attention_mask"):
                        continue
                    if (isinstance(value, torch.Tensor) and value.ndim == 2
                            and value.shape[0] == 1 and value.shape[1] == base_length
                            and not torch.is_floating_point(value)):
                        inputs[name] = torch.cat(
                            [value, torch.zeros_like(extension)], dim=1)
                branches.append(Branch(inputs, positions, length - 1, length))
                logical_tokens += len(suffix)
                compute_tokens += length
            compiled.append(Question(qid, question["type"], keys, [d for _, d in pairs],
                                     branches, distribution, is_text))
        return CompiledState(compiled, logical_tokens, compute_tokens, media, modality)
