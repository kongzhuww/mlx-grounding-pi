"""Strict local-only loading of converted MLX checkpoints."""

import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .config import load_config
from .model import Model


def load_model(path, *, lazy=False):
    path = Path(path)
    config = load_config(path / "config.json")
    settings = config.get("mlx_grounding_pi", {})
    if settings.get("format_version") != 1:
        raise ValueError("expected a converted MLX checkpoint; run mlx_grounding_pi.convert first")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    model = Model(config)
    quant = settings.get("quantization")
    if quant:
        modules = set(quant["modules"])
        known_modules = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
        if not modules <= set(known_modules) or any(
            not p.startswith("language_model.") for p in modules
        ):
            raise ValueError("invalid language quantization module list")
        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant["mode"],
            class_predicate=lambda p, m: p in modules,
        )
    expected = {k: v.shape for k, v in tree_flatten(model.parameters())}
    if set(index["weight_map"]) != set(expected):
        raise ValueError("converted checkpoint tensor names do not match model parameters")
    seen = set()
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard:
            raise ValueError("invalid shard filename")
        values = mx.load(str(path / shard))
        wanted = {k for k, v in index["weight_map"].items() if v == shard}
        if set(values) != wanted:
            raise ValueError(f"shard contents do not match weight index: {shard}")
        for name, value in values.items():
            if value.shape != expected[name] or name in seen:
                raise ValueError(f"wrong shape or duplicate tensor: {name}")
            seen.add(name)
        model.load_weights(list(values.items()), strict=False)
        if not lazy:
            mx.eval(*values.values())
        del values
    if seen != set(expected):
        raise ValueError("incomplete checkpoint")
    model.eval()
    return model, config
