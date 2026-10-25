from pathlib import Path

import pytest

from mlx_grounding_pi.config import VisionConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_vision():
    return VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        qkv_hidden_size=16,
        patch_size=2,
        init_pos_emb_height=2,
        init_pos_emb_width=3,
        projector_hidden_size=32,
        out_hidden_size=12,
    )


@pytest.fixture
def tiny_config(tiny_vision):
    from dataclasses import asdict

    return {
        "image_token_id": 31,
        "vision_config": asdict(tiny_vision),
        "text_config": {
            "model_type": "qwen3",
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
        },
    }
