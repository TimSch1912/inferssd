#!/usr/bin/env python3
"""Record real-model token IDs; compare flag-off/on files. Uses only the Python stdlib.

Record after starting the server: python3 tests/dspark_equivalence.py --record off.json
Repeat with --mtp-sidecar, then: python3 tests/dspark_equivalence.py --compare off.json on.json
"""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path


def post(base, path, value):
    request = urllib.request.Request(base + path, json.dumps(value).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def record(base):
    cases = [
        ("greedy", 0, 42, "Erkläre in einfachen Worten, wie ein Regenbogen entsteht.", 96),
        ("seeded", 0.8, 1234, "Write a Python function that merges two sorted lists. Explain the edge cases.", 96),
        ("grammar", 0.7, 7, 'Return a JSON object with exactly the keys "color" and "count".', 32),
    ]
    results = []
    for name, temperature, seed, prompt, count in cases:
        rendered = post(base, "/apply-template", {"messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True})
        tokens = rendered["tokens"]
        body = dict(prompt=tokens, n_predict=count, temperature=temperature, seed=seed, top_k=0,
            top_p=0.95, min_p=0, repeat_penalty=1.05, cache_prompt=False, return_tokens=True,
            n_probs=5, post_sampling_probs=False)
        if name == "grammar":
            body["grammar"] = 'root ::= "{" ws "\\\"color\\\"" ws ":" ws "\\\"blue\\\"" ws "," ws "\\\"count\\\"" ws ":" ws [0-9] "}"\nws ::= [ \\t\\n]*'
        start = time.monotonic()
        reply = post(base, "/completion", body)
        ids = reply.get("tokens")
        if not ids or not all(type(t) is int for t in ids):
            raise ValueError(f"{name}: no authoritative output token IDs")
        probabilities = reply["completion_probabilities"]
        assert len(probabilities) == len(ids)
        for p in probabilities:
            assert math.isfinite(p["logprob"]) and p["logprob"] <= 0
            assert all(math.isfinite(v["logprob"]) and v["logprob"] <= 0 for v in p["top_logprobs"])
            values = [math.exp(v["logprob"]) for v in p["top_logprobs"]]
            assert values and all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            assert values == sorted(values, reverse=True) and sum(values) <= 1.00001
        row = dict(name=name, request=body, tokens=ids, content=reply["content"], probabilities=probabilities,
            timings=reply["timings"], wall_s=time.monotonic()-start)
        results.append(row)
        print(name, len(ids), reply["timings"], flush=True)
    return results


def compare(paths):
    a, b = (json.loads(Path(p).read_text()) for p in paths)
    if len(a) != len(b):
        raise AssertionError("different case counts")
    for left, right in zip(a, b):
        if left["request"] != right["request"]:
            raise AssertionError("different inputs/samplers")
        x, y = left["tokens"], right["tokens"]
        match = next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
        print(f'{left["name"]}: common={match} off={len(x)} on={len(y)}')
        if x != y:
            raise AssertionError(f'{left["name"]}: token mismatch at {match}')
    print("exact token-ID equivalence: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record", type=Path)
    action.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.record:
        args.record.write_text(json.dumps(record(args.url), indent=2, ensure_ascii=False)+"\n")
    else:
        compare(args.compare)
