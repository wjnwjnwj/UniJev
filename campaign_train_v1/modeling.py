"""SFT initialization plus a deterministic distribution actor."""
import math
import torch
from torch import nn
from .base_model import build_model as build_sft_model, build_compiler, optimizer_groups as sft_groups


class AnalyticPolicy(nn.Module):
    def __init__(self, base, actor_mode="temperature", temperature_min=0.05, temperature_max=20.0):
        super().__init__()
        if actor_mode not in {"temperature", "head", "sft_adapters"}:
            raise ValueError("actor_mode must be temperature, head or sft_adapters")
        if not 0 < temperature_min <= 1 <= temperature_max or temperature_min == temperature_max:
            raise ValueError("Temperature bounds must contain 1 and have positive width")
        self.base = base
        # Capture all SFT delta parameters BEFORE freezing. They must all be saved,
        # even when only temperature/head is optimized.
        self.sft_parameter_names = tuple(n for n, p in base.named_parameters() if p.requires_grad)
        self.actor_mode = actor_mode
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        if actor_mode != "sft_adapters":
            base.requires_grad_(False)
        if actor_mode == "head":
            base.head.requires_grad_(True)
        self.log_temperature = nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=next(base.parameters()).device),
            requires_grad=actor_mode == "temperature")
        self.train()

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, question):
        return self.base(question) / self.temperature

    def train(self, mode=True):
        super().train(mode)
        if self.actor_mode != "sft_adapters":
            self.base.eval()
        # Deterministic actor: no dropout exploration. Keep adapter mode training
        # enabled so backbone gradient checkpointing can work.
        for module in self.base.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        return self

    @torch.no_grad()
    def project_parameters(self):
        if self.actor_mode == "temperature":
            self.log_temperature.clamp_(math.log(self.temperature_min), math.log(self.temperature_max))


def build_model(config):
    base_config = dict(config)
    if config["actor_mode"] != "sft_adapters":
        base_config["gradient_checkpointing"] = False
    base = build_sft_model(base_config)
    return AnalyticPolicy(base, config["actor_mode"], config["temperature_min"], config["temperature_max"])


def optimizer_groups(model, config):
    if model.actor_mode == "temperature":
        return [{"params": [model.log_temperature], "names": ["log_temperature"],
                 "lr": config["temperature_lr"], "name": "temperature"}]
    return sft_groups(model.base, config)
