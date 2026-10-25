"""GroundingPI composition using the unmodified mlx-lm Qwen3 implementation.

Only batch size 1 is supported. See loading.py and generate.py for local
checkpoint loading and greedy image grounding.
"""

import mlx.core as mx
from mlx import nn
from mlx_lm.models.qwen3 import Model as Qwen3
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args

from .config import VisionConfig
from .projector import Projector
from .vision import VisionTower


class Visual(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vision_tower = VisionTower(cfg)
        self.projector = Projector(cfg)

    def __call__(self, patches, grids):
        # Match official get_image_features(): FP32 processor pixels must not
        # promote the BF16 vision tower to mixed/F32 activations in MLX.
        patches = patches.astype(self.vision_tower.patch_embed.proj.weight.dtype)
        return mx.concatenate(self.projector(self.vision_tower(patches, grids)))


class Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.image_token_id = config["image_token_id"]
        vision = VisionConfig.from_dict(config["vision_config"])
        if vision.out_hidden_size != config["text_config"]["hidden_size"]:
            raise ValueError("vision output must match Qwen3 hidden size")
        self.visual = Visual(vision)
        self.language_model = Qwen3(Qwen3Args.from_dict(config["text_config"]))

    @property
    def layers(self):
        return self.language_model.layers

    def __call__(
        self, input_ids, cache=None, pixel_values=None, image_grid_thw=None, logits_to_keep=0
    ):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("only nonempty [1, sequence_length] input IDs are supported")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together")
        embeddings = None
        cached = cache is not None and any(getattr(c, "offset", 0) != 0 for c in cache)
        positions = [
            i for i, token in enumerate(input_ids[0].tolist()) if token == self.image_token_id
        ]
        if positions and pixel_values is None and not cached:
            raise ValueError("image placeholders in prefill require image inputs")
        if pixel_values is not None:
            if cached:
                raise ValueError("images may only be supplied on the initial prefill")
            features = self.visual(pixel_values, image_grid_thw)
            if len(positions) != features.shape[0]:
                raise ValueError("image placeholder count does not match projected feature count")
            embeddings = self.language_model.model.embed_tokens(input_ids)
            embeddings[0, mx.array(positions)] = features.astype(embeddings.dtype)
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if logits_to_keep:
            hidden = self.language_model.model(input_ids, cache=cache, input_embeddings=embeddings)
            hidden = hidden[:, -logits_to_keep:]
            if self.language_model.args.tie_word_embeddings:
                return self.language_model.model.embed_tokens.as_linear(hidden)
            return self.language_model.lm_head(hidden)
        return self.language_model(input_ids, cache=cache, input_embeddings=embeddings)
