"""Single-image NumPy/Pillow preprocessing; no Torch or remote-code dependency.

Pixel utilities are deliberately shared with the audited, vendored upstream
implementation to preserve resize/alpha/patch ordering. Tokenization/chat-template
integration uses the checkpoint's saved tokenizer and chat template.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ._vendor.media_utils import (
    TransparentBgConfig,
    fill_transparent_bg_with,
    image_to_np,
    navit_patchify,
    navit_resize_image,
    normalize,
)


@dataclass(frozen=True)
class ImageInputs:
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int]
    num_tokens: int


class Processor:
    """Local-only checkpoint tokenizer, saved chat template and one image."""

    def __init__(self, path, *, max_output_tokens=None):
        from transformers import AutoTokenizer

        self.path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            local_files_only=True,
            trust_remote_code=False,
        )
        template = self.path / "chat_template.jinja"
        if template.is_file():
            self.tokenizer.chat_template = template.read_text()
        if not self.tokenizer.chat_template:
            raise ValueError("checkpoint must provide its own chat template")
        self.image_processor = ImageProcessor.from_config(
            self.path / "preprocessor_config.json",
            max_output_tokens=max_output_tokens,
        )
        config = json.loads((self.path / "config.json").read_text())
        self.image_token_id = config["image_token_id"]
        if self.tokenizer.convert_tokens_to_ids("<|image_pad|>") != self.image_token_id:
            raise ValueError("tokenizer and model disagree on image token ID")

    def __call__(self, image, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty text")
        inputs = self.image_processor(image)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prompt.count("<|image_pad|>") != 1:
            raise ValueError("expected exactly one image placeholder in the rendered prompt")
        prompt = prompt.replace("<|image_pad|>", "<|image_pad|>" * inputs.num_tokens)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
        if np.count_nonzero(encoded["input_ids"] == self.image_token_id) != inputs.num_tokens:
            raise ValueError("tokenizer did not preserve adjacent image placeholder tokens")
        return {
            "input_ids": encoded["input_ids"],
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
            "prompt": prompt,
            "image": inputs,
        }


class ImageProcessor:
    def __init__(self, media_proc_cfg: dict, max_output_tokens: int | None = None):
        self.cfg = dict(media_proc_cfg)
        self.patch_size = int(self.cfg["patch_size"])
        self.merge_size = int(self.cfg["merge_kernel_size"])
        self.patch_limit = int(self.cfg["in_patch_limit"])
        self.max_output_tokens = self.patch_limit // self.merge_size**2
        if max_output_tokens is not None:
            if type(max_output_tokens) is not int or max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be a positive integer")
            self.max_output_tokens = min(max_output_tokens, self.max_output_tokens)
            self.patch_limit = min(self.patch_limit, self.max_output_tokens * self.merge_size**2)
        if self.cfg.get("fixed_output_tokens") is not None:
            raise ValueError("fixed_output_tokens has not been validated")
        if self.patch_limit < self.merge_size**2:
            raise ValueError("patch budget must fit at least one merged token")

    @classmethod
    def from_config(cls, path: str | Path, **kwargs):
        return cls(json.loads(Path(path).read_text())["media_proc_cfg"], **kwargs)

    def resize_config(self, width: int, height: int) -> dict:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        limit = self.patch_limit
        while limit > 0:
            result = navit_resize_image(
                width,
                height,
                self.patch_size,
                self.merge_size,
                limit,
                int(self.cfg["patch_limit_on_one_side"]),
                None,
            )
            if result["num_tokens"] <= self.max_output_tokens:
                return result
            reduced = int(limit * self.max_output_tokens / result["num_tokens"])
            limit = min(limit - 1, reduced)
        raise ValueError("unable to satisfy image token budget")

    def __call__(self, image: Image.Image | str | Path) -> ImageInputs:
        if not isinstance(image, Image.Image):
            with Image.open(image) as source:
                return self(source.copy())
        original_size = image.size
        resize = self.resize_config(*image.size)
        background = self.cfg.get("transparent_bg_config")
        background = TransparentBgConfig(**background) if background else None
        stage = self.cfg.get("transparent_bg_fill_stage", "before_resize")
        if stage not in {"before_resize", "after_resize"}:
            raise ValueError("unsupported alpha compositing stage")
        if stage == "before_resize":
            image = fill_transparent_bg_with(image, background)
        pixels = image_to_np(
            image,
            (resize["new_width"], resize["new_height"]),
            transparent_bg_config=background,
            transparent_bg_fill_stage=stage,
        )
        pixels = np.pad(pixels, ((0, resize["pad_height"]), (0, resize["pad_width"]), (0, 0)))
        pixels = normalize(
            pixels, np.array(self.cfg["image_mean"]), 1.0 / np.array(self.cfg["image_std"])
        )
        patches = navit_patchify(pixels[None], self.patch_size)
        return ImageInputs(
            pixel_values=patches["pixel_values"],
            image_grid_thw=patches["grid_thw"][None],
            original_size=original_size,
            resized_size=(resize["new_width"], resize["new_height"]),
            padding=(resize["pad_width"], resize["pad_height"]),
            num_tokens=resize["num_tokens"],
        )
