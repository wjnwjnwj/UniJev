"""Qwen3.5 multimodal backbone + shared bilinear decision head, and its builder."""
import math

import torch
from torch import nn



class DecisionHead(nn.Module):
    def __init__(self, hidden_size, projection_dim=256):
        super().__init__()
        self.decision = nn.Linear(hidden_size, projection_dim, bias=False)
        self.candidate = nn.Linear(hidden_size, projection_dim, bias=False)
        self.scale = math.sqrt(projection_dim)

    def forward(self, hidden, positions, decision_position):
        # Compute head and losses in FP32 even with a BF16 backbone.
        d = self.decision(hidden[0, decision_position].float())
        c = self.candidate(hidden[0, positions].float())
        return (c * d).sum(-1) / self.scale


class DecisionModel(nn.Module):
    """One forward pass per branch; logits over candidates, no token generation."""

    def __init__(self, backbone, projection_dim=256):
        super().__init__()
        self.backbone = backbone
        self.head = DecisionHead(backbone.config.text_config.hidden_size, projection_dim)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "deterministic", False):
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        return self

    def forward(self, question):
        device = next(self.backbone.parameters()).device
        outputs = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            outputs.append(self.head(hidden, branch.candidate_positions, branch.decision_position))
        return torch.cat(outputs)


def build_model(config):
    from transformers import Qwen3_5ForConditionalGeneration
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config.get("dtype", "bf16")]
    # Load the exact published architecture before extracting its backbone.
    # Loading the bare Qwen3_5Model can trigger a language-prefix remapping in
    # Transformers 5.4 and silently initialize the language tower from scratch.
    container, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_path"], dtype=dtype, attn_implementation=config.get("attn_implementation", "sdpa"),
        local_files_only=True, output_loading_info=True)
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete pretrained model load: {loading_info}")
    backbone = container.model
    del container  # Never execute or retain the generation/vocabulary output layer.
    backbone.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    targets = [name for name, module in backbone.language_model.named_modules()
               if isinstance(module, nn.Linear)]
    backbone.language_model = get_peft_model(backbone.language_model, LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=0.0, target_modules=targets, bias="none"))
    backbone.visual.merger.requires_grad_(True)
    if config["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = DecisionModel(backbone, config.get("projection_dim", 256))
    model.deterministic = config.get("deterministic", True)
    model.delta_parameter_names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
    model.to(config.get("device", "cuda"))
    return model


def build_compiler(config, media_root="."):
    from transformers import AutoProcessor
    from .compiler import Compiler
    processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
    return Compiler(processor, media_root, config.get("max_length", 32768), config.get("media_kwargs"))


def optimizer_groups(model, config):
    groups = {"head": [], "lora": [], "merger": []}
    rates = {"head": 2e-4, "lora": 5e-5, "merger": 1e-5}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            group = "head"
        elif "lora_" in name:
            group = "lora"
        elif ".visual.merger." in name:
            group = "merger"
        else:
            raise ValueError(f"Unclassified trainable parameter: {name}")
        groups[group].append((name, parameter))
    return [{"params": [p for _, p in parameters], "names": [n for n, _ in parameters],
             "lr": config.get(f"{name}_lr", rates[name]), "name": name}
            for name, parameters in groups.items() if parameters]

