#!/usr/bin/env python3
"""Convert a Laya checkpoint (safetensors + config + tokenizer) to a single GGUF file.

    python3 convert_laya_gguf.py --model-dir /path/to/checkpoint --out model-f16.gguf
    python3 convert_laya_gguf.py --model-dir /path/to/checkpoint --out model-q8.gguf --quantize q8_0
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
from safetensors import safe_open

# -------------------------------------------------------------------------- GGUF writer (minimal, self-contained)
# We avoid importing from llama.cpp's gguf-py to keep this script dependency-free.

GGUF_MAGIC = 0x46554747  # 'GGUF'
GGUF_VERSION = 3

# ggml type enum (subset we use)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

TYPE_SIZE = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_Q8_0: 34}  # Q8_0: 32 bytes data + 2 bytes scale per block of 32
BLOCK_SIZE = {GGML_TYPE_F32: 1, GGML_TYPE_F16: 1, GGML_TYPE_Q8_0: 32}

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def _write_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _write_kv(f, key: str, val):
    _write_str(f, key)
    if isinstance(val, str):
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        _write_str(f, val)
    elif isinstance(val, float):
        f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
        f.write(struct.pack("<f", val))
    elif isinstance(val, int):
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", val))
    elif isinstance(val, list) and all(isinstance(v, float) for v in val):
        f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
        f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
        f.write(struct.pack("<Q", len(val)))
        for v in val:
            f.write(struct.pack("<f", v))
    elif isinstance(val, list) and all(isinstance(v, str) for v in val):
        f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        f.write(struct.pack("<Q", len(val)))
        for v in val:
            _write_str(f, v)
    else:
        raise ValueError(f"unsupported metadata type for key {key}: {type(val)}")


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize fp32 array to Q8_0 format: per-block-of-32 scale + int8 weights."""
    flat = data.astype(np.float32).flatten()
    n = len(flat)
    # pad to multiple of 32
    if n % 32 != 0:
        flat = np.concatenate([flat, np.zeros(32 - n % 32, dtype=np.float32)])
    blocks = flat.reshape(-1, 32)
    out = bytearray()
    for block in blocks:
        amax = np.abs(block).max()
        scale = amax / 127.0 if amax != 0 else 0.0
        inv_scale = 1.0 / scale if scale != 0 else 0.0
        quants = np.round(block * inv_scale).clip(-128, 127).astype(np.int8)
        out += struct.pack("<f", scale)  # fp32 scale -- WAIT, Q8_0 uses fp16 scale
        # Actually Q8_0 format: 2 bytes (fp16 scale) + 32 bytes (int8 data) = 34 bytes per block
        # Let me fix this
    # Redo properly
    out = bytearray()
    for block in blocks:
        amax = np.abs(block).max()
        scale = amax / 127.0 if amax != 0 else 0.0
        inv_scale = 1.0 / scale if scale != 0 else 0.0
        quants = np.round(block * inv_scale).clip(-128, 127).astype(np.int8)
        # fp16 scale
        scale_f16 = np.float16(scale)
        out += scale_f16.tobytes()  # 2 bytes
        out += quants.tobytes()     # 32 bytes
    return bytes(out)


def write_gguf(path: str, metadata: dict, tensors: list):
    """Write a GGUF file. tensors: [(name, np_array, ggml_type), ...]"""
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))    # n_tensors
        f.write(struct.pack("<Q", len(metadata)))   # n_kv

        # Metadata KV
        for k, v in metadata.items():
            _write_kv(f, k, v)

        # Tensor info
        tensor_data_list = []
        offset = 0
        for name, arr, dtype in tensors:
            _write_str(f, name)
            ndim = len(arr.shape)
            f.write(struct.pack("<I", ndim))
            # GGUF stores shape in ne order: ne[0] is the contiguous (fast) dimension.
            # numpy row-major: last dim is contiguous. So reverse the shape for GGUF.
            gguf_shape = list(reversed(arr.shape))
            for d in gguf_shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype))
            f.write(struct.pack("<Q", offset))

            # Compute data size
            n_elements = int(np.prod(arr.shape))
            bs = BLOCK_SIZE[dtype]
            n_blocks = (n_elements + bs - 1) // bs
            data_size = n_blocks * TYPE_SIZE[dtype]
            tensor_data_list.append((arr, dtype, data_size))

            # Align offset to 32 bytes
            offset += data_size
            offset = (offset + 31) & ~31

        # Padding to align tensor data start to 32 bytes
        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b'\x00' * (aligned - pos))

        # Tensor data
        for arr, dtype, data_size in tensor_data_list:
            if dtype == GGML_TYPE_F32:
                f.write(arr.astype(np.float32).tobytes())
            elif dtype == GGML_TYPE_F16:
                f.write(arr.astype(np.float16).tobytes())
            elif dtype == GGML_TYPE_Q8_0:
                f.write(quantize_q8_0(arr))
            # Pad to 32-byte alignment
            pos = f.tell()
            aligned = (pos + 31) & ~31
            f.write(b'\x00' * (aligned - pos))

    print(f"  wrote {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


# -------------------------------------------------------------------------- conversion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", choices=["f16", "f32", "q8_0"], default="f16",
                    help="weight dtype: f16 (default), f32, or q8_0")
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_dir, "rl_agent_config.json")
    sf_path = os.path.join(args.model_dir, "model.safetensors")
    enc_cfg_path = os.path.join(args.model_dir, "encoder", "config.json")
    tok_path = os.path.join(args.model_dir, "tokenizer", "tokenizer.json")

    for p in [cfg_path, sf_path, enc_cfg_path]:
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(enc_cfg_path) as f:
        enc_cfg = json.load(f)

    n_layers = enc_cfg.get("num_hidden_layers", 22)
    n_heads = enc_cfg.get("num_attention_heads", 12)
    hidden = enc_cfg.get("hidden_size", 768)
    intermediate = enc_cfg.get("intermediate_size", 1152)
    vocab = enc_cfg.get("vocab_size", 256000)
    max_pos = enc_cfg.get("max_position_embeddings", 8192)
    head_layers = cfg.get("head_layers", 2)
    temperatures = cfg.get("temperature", [1.0, 1.0, 1.0])

    print(f"[config] encoder: {n_layers} layers, {hidden}d, {n_heads} heads, vocab {vocab}")
    print(f"         intermediate: {intermediate} (GeGLU: Wi=[{intermediate*2},{hidden}])")
    print(f"         head: {head_layers} layers, temperature: {temperatures}")
    print(f"         quantize: {args.quantize}")

    # Build metadata
    metadata = {
        "general.architecture": "laya",
        "general.name": cfg.get("model_name", "laya"),
        "laya.hidden_size": hidden,
        "laya.num_layers": n_layers,
        "laya.num_heads": n_heads,
        "laya.intermediate_size": intermediate,
        "laya.vocab_size": vocab,
        "laya.max_position_embeddings": max_pos,
        "laya.head_layers": head_layers,
        "laya.max_len": cfg.get("max_len", 1024),
        "laya.head_max_len": cfg.get("head_max_len", 256),
        "laya.temperatures": [float(t) for t in temperatures],
        "laya.local_attention": enc_cfg.get("local_attention", 128),
        "laya.global_attn_every_n": enc_cfg.get("global_attn_every_n_layers", 3),
        "laya.rope_theta": float(enc_cfg.get("rope_parameters", {}).get("full_attention", {}).get("rope_theta", 160000)),
    }

    # Load tokenizer model for embedding in GGUF
    if os.path.exists(tok_path):
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_json = f.read()
        metadata["tokenizer.json"] = tok_json

    # Choose dtype
    dtype_map = {"f16": GGML_TYPE_F16, "f32": GGML_TYPE_F32, "q8_0": GGML_TYPE_Q8_0}
    target_dtype = dtype_map[args.quantize]
    # Small tensors (norms, biases, embeddings <1024 elements, temperature) stay f32
    SMALL_THRESHOLD = 1024

    # Load and convert tensors
    sf = safe_open(sf_path, framework="np")
    tensors = []
    total_params = 0
    for name in sorted(sf.keys()):
        arr = sf.get_tensor(name)
        total_params += arr.size

        # ggml convention: ne[0] is the contiguous (fast) dimension, and ggml_mul_mat(a, b)
        # computes a^T @ b. For a PyTorch weight [out_dim, in_dim]:
        #   - ggml reads the bytes as ne[0]=first_dim_of_stored_shape, ne[1]=second
        #   - numpy row-major: last dim is contiguous → ne[0] maps to in_dim
        #   - So WITHOUT transposing, the raw bytes give ggml ne=[in_dim, out_dim]
        #   - ggml_mul_mat(W, x): W^T @ x. W ne=[in_dim, out_dim], x ne=[in_dim, L]
        #     → output ne=[out_dim, L]. Correct!
        # For get_rows(emb, ids): emb ne=[hidden, vocab]. get_rows takes ne[0]-length
        #   rows → returns [hidden, n_ids]. We need emb stored as [hidden, vocab] in ggml.
        #   PyTorch emb is [vocab, hidden]. So we DO need to transpose it for ggml_get_rows.
        #   But since numpy row-major [vocab, hidden] stores hidden as contiguous → ggml reads
        #   ne=[hidden, vocab]. That's exactly what we want! No transpose needed!
        # CONCLUSION: do NOT transpose anything. Just write the numpy array as-is.

        # Keep small tensors in f32 for precision
        if arr.size <= SMALL_THRESHOLD or "norm" in name or name == "temperature":
            dt = GGML_TYPE_F32
        else:
            dt = target_dtype
        tensors.append((name, arr, dt))

    print(f"[tensors] {len(tensors)} tensors, {total_params:,} params")

    write_gguf(args.out, metadata, tensors)
    print("[done]")


if __name__ == "__main__":
    main()
