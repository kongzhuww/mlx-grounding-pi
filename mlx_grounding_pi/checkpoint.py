"""Bounded safetensors header inspection; no tensor payload or model allocation."""

import json
import math
import re
import struct
from urllib.request import Request, urlopen

from .config import VisionConfig

MAX_HEADER_BYTES = 2 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}


def read_http_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=45) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if (
            response.status != 206
            or not match
            or tuple(map(int, match.groups()[:2])) != (start, end)
        ):
            raise ValueError(
                "server did not honor the requested byte range; refusing full download"
            )
        payload = response.read(end - start + 2)
        if len(payload) != end - start + 1:
            raise ValueError("unexpected range response length")
        return payload, int(match[3])


def fetch_header(url: str) -> dict:
    # Distinct queries prevent proxy/CDN caches from reusing the initial 8-byte response.
    sep = "&" if "?" in url else "?"
    first, total = read_http_range(url + sep + "header_range=prefix", 0, 7)
    length = struct.unpack("<Q", first)[0]
    if not 2 <= length <= MAX_HEADER_BYTES or length + 8 > total:
        raise ValueError("invalid or oversized safetensors header")
    raw, total2 = read_http_range(url + sep + "header_range=json", 8, 7 + length)
    if total != total2:
        raise ValueError("file size changed between range requests")
    header = json.loads(raw)
    validate_header(header, total - length - 8)
    return {"header_bytes": length, "file_bytes": total, "tensors": header}


def validate_header(header: dict, data_bytes: int):
    offset = 0
    for key, value in sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    ):
        shape, dtype = value["shape"], value["dtype"]
        if any(type(d) is not int or d < 0 for d in shape) or dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported shape or dtype: {key}")
        start, end = value["data_offsets"]
        if start != offset or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"invalid tensor byte offsets: {key}")
        offset = end
    if offset != data_bytes:
        raise ValueError("header does not cover the complete tensor payload")


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Exact unquantized source tensor schema, without allocating 4.84B parameters."""
    v = VisionConfig.from_dict(config["vision_config"])
    t = config["text_config"]
    h, q, m = v.hidden_size, v.qkv_hidden_size, v.intermediate_size
    result = {
        "model.visual.vision_tower.patch_embed.proj.weight": (h, 3, v.patch_size, v.patch_size),
        "model.visual.vision_tower.patch_embed.pos_emb.weight": (
            v.init_pos_emb_height,
            v.init_pos_emb_width,
            h,
        ),
        "model.visual.vision_tower.encoder.final_layernorm.weight": (h,),
        "model.visual.projector.pre_norm.weight": (h,),
        "model.visual.projector.pre_norm.bias": (h,),
        "model.visual.projector.fc1.weight": (v.projector_hidden_size, h * v.spatial_merge_size**2),
        "model.visual.projector.fc2.weight": (v.out_hidden_size, v.projector_hidden_size),
        "model.visual.projector.post_norm.weight": (v.out_hidden_size,),
    }
    if v.patch_embed_proj_bias:
        result["model.visual.vision_tower.patch_embed.proj.bias"] = (h,)
    for i in range(v.num_hidden_layers):
        prefix = f"model.visual.vision_tower.encoder.blocks.{i}."
        for key, shape in {
            "norm0.weight": (h,),
            "norm1.weight": (h,),
            "wqkv.weight": (3 * q, h),
            "wo.weight": (h, q),
            "mlp.fc0.weight": (m, h),
            "mlp.fc1.weight": (h, m),
        }.items():
            result[prefix + key] = shape
        if v.attn_bias:
            result[prefix + "wqkv.bias"] = (3 * q,)
            result[prefix + "wo.bias"] = (h,)
        if v.linear_bias:
            result[prefix + "mlp.fc0.bias"] = (m,)
            result[prefix + "mlp.fc1.bias"] = (h,)
    d, head, f = t["hidden_size"], t["head_dim"], t["intermediate_size"]
    result["model.language_model.embed_tokens.weight"] = (t["vocab_size"], d)
    result["model.language_model.norm.weight"] = (d,)
    if not t["tie_word_embeddings"]:
        result["lm_head.weight"] = (t["vocab_size"], d)
    for i in range(t["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{i}."
        for name in ["input_layernorm", "post_attention_layernorm"]:
            result[prefix + name + ".weight"] = (d,)
        for name in ["q_norm", "k_norm"]:
            result[prefix + "self_attn." + name + ".weight"] = (head,)
        for name, shape in {
            "q_proj": (t["num_attention_heads"] * head, d),
            "k_proj": (t["num_key_value_heads"] * head, d),
            "v_proj": (t["num_key_value_heads"] * head, d),
            "o_proj": (d, t["num_attention_heads"] * head),
        }.items():
            result[prefix + "self_attn." + name + ".weight"] = shape
        for name, shape in {"gate_proj": (f, d), "up_proj": (f, d), "down_proj": (d, f)}.items():
            result[prefix + "mlp." + name + ".weight"] = shape
    return result


def verify_schema(config: dict, index: dict, headers: dict) -> dict:
    expected = expected_shapes(config)
    if set(expected) != set(index["weight_map"]):
        raise ValueError("checkpoint tensor names differ from the configured architecture")
    observed, parameters, payload_bytes = set(), 0, 0
    for shard, wrapper in headers.items():
        validate_header(wrapper["tensors"], wrapper["file_bytes"] - wrapper["header_bytes"] - 8)
        for name, spec in wrapper["tensors"].items():
            if name == "__metadata__":
                continue
            if name in observed or index["weight_map"].get(name) != shard:
                raise ValueError(f"unexpected or duplicate tensor in shard: {name}")
            if tuple(spec["shape"]) != expected[name]:
                raise ValueError(f"shape mismatch: {name}: {spec['shape']} != {expected[name]}")
            if spec["dtype"] != "BF16":
                raise ValueError(f"expected original BF16 weights: {name}")
            observed.add(name)
            parameters += math.prod(spec["shape"])
            a, b = spec["data_offsets"]
            payload_bytes += b - a
    if observed != set(expected):
        raise ValueError("not all checkpoint tensors were inspected")
    if (
        parameters != index["metadata"]["total_parameters"]
        or payload_bytes != index["metadata"]["total_size"]
    ):
        raise ValueError("checkpoint totals disagree with index metadata")
    return {
        "verified_tensors": len(observed),
        "parameters": parameters,
        "payload_bytes": payload_bytes,
    }
