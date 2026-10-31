#!/usr/bin/env python3
"""CPU-only DeepSeek-V4.1 DSpark sidecar, using the main GGUF's conversion rules.

Reference: vcruz305/llama.cpp@2076412598e99127ea2554467b2b7530856895d3,
conversion/{base,deepseek}.py: repack_mxfp4_blocks, DeepseekV4Model.tensor_force_quant,
DeepseekV41Model.dequant_model. No torch, model loading, or GPU runtime is used.

Common mtp.K tensors become blk.(40+K).<the existing main-layer name>.
DSpark-only tensors use blk.(40+K).nextn.*; this is a sidecar, not a standalone model.
Run --self-test for the small format check; conversion verifies every output hash,
shape and type and compares all common tensors with the actual main GGUF headers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf

Q = gguf.GGMLQuantizationType
ROOT = Path("/daten/inferssd-models/DeepSeek-V4.1-Flash-1017-MTP")
MAIN = Path("/daten/inferssd-models/DeepSeek-V4.1-Flash-1017-GGUF/Q8_0")
CONVERTER_REV = "2076412598e99127ea2554467b2b7530856895d3"

COMMON = {
    "attn.attn_sink": "attn_sinks.weight",
    "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight",
    "attn.wkv.weight": "attn_kv.weight",
    "attn.wo_a.weight": "attn_output_a.weight",
    "attn.wo_b.weight": "attn_output_b.weight",
    "attn.wq_a.weight": "attn_q_a.weight",
    "attn.wq_b.weight": "attn_q_b.weight",
    "attn_norm.weight": "attn_norm.weight",
    "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.gate.bias_vl": "exp_probs_b_vl.bias",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    **{f"hc_{part}_{kind}": f"hc_{part}_{kind}.weight"
       for part in ("attn", "ffn") for kind in ("base", "fn", "scale")},
}
SPECIAL = {
    "mtp.0.main_proj.weight": "blk.40.nextn.main_proj.weight",
    "mtp.0.main_norm.weight": "blk.40.nextn.main_norm.weight",
    "mtp.2.norm.weight": "blk.42.nextn.shared_head_norm.weight",
    "mtp.2.markov_head.embed.weight": "blk.42.nextn.markov_embd.weight",
    "mtp.2.markov_head.head.weight": "blk.42.nextn.markov_output.weight",
    "mtp.2.confidence_head.proj.weight": "blk.42.nextn.confidence_proj.weight",
}
EXPERTS = {"w1": "ffn_gate_exps.weight", "w2": "ffn_down_exps.weight",
           "w3": "ffn_up_exps.weight"}


def fp8_lut():
    """IEEE-like E4M3FN, including subnormals, signed zero, and the two NaNs."""
    b = np.arange(256, dtype=np.uint8)
    e, m = (b >> 3) & 15, b & 7
    a = np.where(e == 0, m.astype(np.float32) * 2**-9,
                 (1 + m.astype(np.float32) / 8) * np.exp2(e.astype(np.float32) - 7))
    a = np.where(b & 128, -a, a).astype(np.float32)
    a[(b & 127) == 127] = np.nan
    return a


FP8 = fp8_lut()


def repack_mxfp4(packed, scales):
    """Same nibble permutation and unchanged E8M0 byte as pinned base.py:743."""
    rows, cols = packed.shape
    if cols % 16 or scales.shape != (rows, cols // 16):
        raise ValueError("MXFP4 weight/scale shape mismatch")
    src = packed.reshape(rows, -1, 16)
    vals = np.stack((src & 15, src >> 4), axis=-1).reshape(rows, -1, 32)
    qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
    return np.concatenate((scales[:, :, None], qs), axis=-1).reshape(rows, -1)


def source_tensors(original):
    index = json.loads((original / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {k: v for k, v in index.items() if k.startswith("mtp.")}
    if not selected:
        raise ValueError("No mtp.* tensors in index")
    tensors = {}
    for shard in sorted(set(selected.values())):
        path = original / shard
        with gguf.utility.SafetensorsLocal(path) as entries:
            end = max(t.data_range.offset + t.data_range.size for t in entries.values())
            if path.stat().st_size != end:
                raise ValueError(f"Incomplete shard: {path} (expected {end} bytes)")
            for name, tensor in entries.items():
                if selected.get(name) == shard:
                    tensors[name] = tensor
    if tensors.keys() != selected.keys():
        raise ValueError("Index and shard tensors differ")
    return tensors


def raw(tensor):
    dtype = {"F32": "<f4", "BF16": "<u2", "F8_E4M3": "u1",
             "F8_E8M0": "u1", "U8": "u1", "I8": "u1"}[tensor.dtype]
    if math.prod(tensor.shape) * np.dtype(dtype).itemsize != tensor.data_range.size:
        raise ValueError("Safetensors shape/byte length mismatch")
    return tensor.mmap_bytes().view(dtype).reshape(tensor.shape)


def make_plan(tensors, config):
    hp = config["text_config"]
    if (hp["num_hidden_layers"], hp["num_nextn_predict_layers"],
        hp["dspark_n_routed_experts"], config["quantization_config"]["weight_block_size"],
        config["quantization_config"]["scale_fmt"]) != (40, 3, 128, [32, 32], "ue8m0"):
        raise ValueError("This converter supports the V4.1 1017 DSpark layout only")
    plan, consumed = [], set()
    for stage in range(3):
        for proj, target in EXPERTS.items():
            names = [f"mtp.{stage}.ffn.experts.{i}.{proj}.weight" for i in range(128)]
            shape = tensors[names[0]].shape
            for name in names:
                scale = name.removesuffix("weight") + "scale"
                if (tensors[name].shape != shape or tensors[name].dtype not in ("U8", "I8") or
                    tensors[scale].dtype != "F8_E8M0" or
                    tensors[scale].shape != (shape[0], shape[1] // 16)):
                    raise ValueError(f"Invalid expert tensor: {name}")
                consumed.update((name, scale))
            plan.append(dict(name=f"blk.{40 + stage}.{target}", sources=names,
                             shape=(128, shape[0], shape[1] * 2), qtype=Q.MXFP4,
                             analogue=f"blk.0.{target}"))
    for name, tensor in sorted(tensors.items()):
        if name in consumed:
            continue
        if name.endswith(".scale"):
            continue  # accounted for below, otherwise rejected by the consumed-set check
        _, stage, suffix = name.split(".", 2)
        if suffix in COMMON:
            target = f"blk.{40 + int(stage)}.{COMMON[suffix]}"
            analogue = f"blk.0.{COMMON[suffix]}"
        else:
            target, analogue = SPECIAL[name], None  # unknown source names must fail
        shape = tensor.shape
        if tensor.dtype == "F8_E4M3":
            scale = name.removesuffix("weight") + "scale"
            if tensors[scale].dtype != "F8_E8M0" or tensors[scale].shape != tuple((s + 31) // 32 for s in shape):
                raise ValueError(f"Invalid FP8 scale for {name}")
            consumed.add(scale)
            qtype = Q.Q8_0
        elif tensor.dtype == "F32" or len(shape) <= 1 or target.endswith("_norm.weight"):
            qtype = Q.F32
        elif tensor.dtype == "BF16":
            qtype = Q.BF16
        else:
            raise ValueError(f"Unsupported dtype for {name}: {tensor.dtype}")
        plan.append(dict(name=target, sources=[name], shape=shape, qtype=qtype, analogue=analogue))
        consumed.add(name)
    if consumed != tensors.keys():
        raise ValueError(f"Unconverted tensors: {tensors.keys() - consumed}")
    return plan


def convert_tensor(item, tensors):
    qtype, shape = item["qtype"], item["shape"]
    name = item["sources"][0]
    if qtype == Q.MXFP4:
        out = np.empty((shape[0], shape[1], shape[2] // 32 * 17), dtype=np.uint8)
        for i, source in enumerate(item["sources"]):
            out[i] = repack_mxfp4(raw(tensors[source]), raw(tensors[source.removesuffix("weight") + "scale"]))
        return out
    weight = raw(tensors[name])
    if qtype == Q.BF16:
        return weight.view(np.uint8)  # preserve the checkpoint BF16 bits
    if qtype == Q.F32:
        if tensors[name].dtype == "BF16":
            return (weight.astype(np.uint32) << 16).view(np.float32)
        if tensors[name].dtype != "F32":
            raise ValueError(f"Expected F32 or BF16: {name}")
        return weight
    scales = raw(tensors[name.removesuffix("weight") + "scale"])
    if np.any(scales == 255):
        raise ValueError(f"NaN E8M0 scale: {name}")
    out = np.empty((shape[0], shape[1] // 32 * 34), dtype=np.uint8)
    # ponytail: row chunks bound scratch memory; no parallel worker pool beside production.
    for start in range(0, shape[0], 256):
        stop = min(start + 256, shape[0])
        values = FP8[weight[start:stop]]
        scale = np.exp2(scales[start // 32:(stop + 31) // 32].astype(np.float32) - 127)
        values *= np.repeat(np.repeat(scale, 32, axis=0), 32, axis=1)[:stop-start, :shape[1]]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite FP8 weight: {name}")
        out[start:stop] = gguf.quantize(values, Q.Q8_0)
    return out


def main_headers(path):
    files = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No main GGUF: {path}")
    headers = {}
    for file in files:
        reader = gguf.GGUFReader(file)
        for tensor in reader.tensors:
            if tensor.name in headers:
                raise ValueError(f"Duplicate main tensor: {tensor.name}")
            headers[tensor.name] = (tuple(map(int, reversed(tensor.shape))), tensor.tensor_type)
    return headers


def check_analogues(plan, headers):
    for item in plan:
        if not item["analogue"]:
            continue
        shape, qtype = headers[item["analogue"]]
        if qtype == Q.MXFP4 or item["analogue"].split(".", 2)[2] in (
                "ffn_gate_inp.weight", "exp_probs_b.bias", "exp_probs_b_vl.bias"):
            if shape[0] != 384:
                raise ValueError(f"Expected 384 main experts: {item['analogue']}")
            shape = (128, *shape[1:])
        if tuple(item["shape"]) != shape or item["qtype"] != qtype:
            raise ValueError(f"Main analogue mismatch: {item['name']}")


def digest(data):
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast("B")).hexdigest()


def verify(path, plan, manifest):
    reader = gguf.GGUFReader(path)
    if len(reader.tensors) != len(plan):
        raise ValueError("Output tensor count mismatch")
    records = {r["name"]: r for r in manifest["tensors"]}
    for tensor, item in zip(reader.tensors, plan):
        if (tensor.name != item["name"] or tensor.tensor_type != item["qtype"] or
            tuple(reversed(tensor.shape)) != tuple(item["shape"]) or
            digest(tensor.data) != records[tensor.name]["sha256"]):
            raise ValueError(f"Output verification failed: {tensor.name}")
    return dict(Counter(t.tensor_type.name for t in reader.tensors))


def self_test():
    # Each code 0..15 twice: independent scalar reference for both nibble orders.
    codes = np.arange(32, dtype=np.uint8) % 16
    packed = (codes[::2] | (codes[1::2] << 4))[None, :]
    result = repack_mxfp4(packed, np.array([[127]], dtype=np.uint8))
    expected = bytes([127] + [int(codes[j]) | (int(codes[j + 16]) << 4) for j in range(16)])
    assert result.tobytes() == expected
    assert FP8[1] == 2**-9 and FP8[8] == 2**-6 and FP8[126] == 448
    assert np.signbit(FP8[128]) and np.isnan(FP8[127]) and FP8[254] == -448
    for qt in (Q.F32, Q.BF16, Q.Q8_0, Q.MXFP4):
        block, size = gguf.GGML_QUANT_SIZES[qt]
        assert gguf.quant_shape_to_byte_shape((2, 5120), qt) == (2, 5120 // block * size)

    def source(dtype, array):
        return SimpleNamespace(dtype=dtype, shape=array.shape,
                               data_range=SimpleNamespace(size=array.nbytes),
                               mmap_bytes=lambda: array.view(np.uint8).reshape(-1))

    # Cross the 256-row chunk boundary and vary scales in BOTH dimensions.
    # E8M0 zero encodes 2**-127, not zero (an exponent-bit reinterpretation fails).
    assert np.exp2(np.float32(-127)) == np.float32(2**-127)
    w = (np.arange(288 * 64).reshape(288, 64) % 127).astype(np.uint8)
    s = (np.arange(18).reshape(9, 2) % 5 + 125).astype(np.uint8)
    tensors = {"w.weight": source("F8_E4M3", w), "w.scale": source("F8_E8M0", s)}
    item = dict(sources=["w.weight"], qtype=Q.Q8_0, shape=w.shape)
    expected = FP8[w].copy()
    for i in range(9):
        for j in range(2):
            expected[i*32:(i+1)*32, j*32:(j+1)*32] *= np.float32(2.0 ** (int(s[i, j]) - 127))
    assert np.array_equal(convert_tensor(item, tensors), gguf.quantize(expected, Q.Q8_0))
    bf = np.array([[0x3F80, 0xC000, 0x0000, 0x8000]], dtype="<u2")
    tensors = {"w.weight": source("BF16", bf)}
    assert convert_tensor(dict(sources=["w.weight"], qtype=Q.BF16, shape=bf.shape), tensors).tobytes() == bf.tobytes()
    assert np.array_equal(convert_tensor(dict(sources=["w.weight"], qtype=Q.F32, shape=bf.shape), tensors),
                          np.array([[1, -2, 0, -0.0]], dtype=np.float32))
    print("self-test OK: E4M3FN, MXFP4 nibble order, FP8 32x32 scales/chunks, BF16/F32, GGUF shapes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "original")
    parser.add_argument("--output", type=Path, default=ROOT / "DeepSeek-V4.1-Flash-1017-MTP-Q8_0.gguf")
    parser.add_argument("--main-model", type=Path, default=MAIN)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    config = json.loads((args.original / "config.json").read_text())
    tensors = source_tensors(args.original)
    plan = make_plan(tensors, config)
    check_analogues(plan, main_headers(args.main_model))
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        print("verified", verify(args.output, plan, manifest))
        return
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("Output/manifest already exists; use --verify-only or a new path")
    temporary = args.output.with_suffix(".gguf.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = gguf.GGUFWriter(temporary, "deepseek41")
    writer.add_name("DeepSeek-V4.1-Flash-1017 DSpark MTP sidecar")
    writer.add_string("general.type", "mtp-sidecar")
    writer.add_string("deepseek41.mtp.format", "inferssd-dspark-v1")
    writer.add_uint32("deepseek41.mtp.base_layer", 40)
    writer.add_uint32("deepseek41.mtp.layer_count", 3)
    writer.add_string("deepseek41.mtp.config", json.dumps(config, sort_keys=True))
    writer.add_string("deepseek41.mtp.converter_reference", CONVERTER_REV)
    writer.add_string("deepseek41.mtp.shared_tensors", "token_embd.weight,output.weight")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    for item in plan:
        block, size = gguf.GGML_QUANT_SIZES[item["qtype"]]
        writer.add_tensor_info(item["name"], item["shape"], np.dtype("float32"),
                               math.prod(item["shape"]) // block * size, raw_dtype=item["qtype"])
    manifest = dict(format="inferssd-dspark-v1", converter_reference=CONVERTER_REV,
                    source_tensor_count=len(tensors), output_tensor_count=len(plan),
                    config_sha256=hashlib.sha256((args.original / "config.json").read_bytes()).hexdigest(),
                    index_sha256=hashlib.sha256((args.original / "model.safetensors.index.json").read_bytes()).hexdigest(),
                    tensors=[])
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for i, item in enumerate(plan, 1):
            data = convert_tensor(item, tensors)
            manifest["tensors"].append({**item, "qtype": item["qtype"].name, "sha256": digest(data)})
            writer.write_tensor_data(data)
            print(f"[{i}/{len(plan)}] {item['name']} {item['shape']} {item['qtype'].name}", flush=True)
            del data
        writer.close()
        manifest["types"] = verify(temporary, plan, manifest)
        temporary.rename(args.output)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        writer.close()
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size,
                      "source_tensors": len(tensors), "output_tensors": len(plan),
                      "types": manifest["types"]}), flush=True)


if __name__ == "__main__":
    main()
