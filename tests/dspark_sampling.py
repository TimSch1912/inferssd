#!/usr/bin/env python3
"""Small real-server categorical sampler check, stdlib only. Record off/on, then compare."""
import argparse
import collections
import json
import math
from pathlib import Path
from dspark_equivalence import post


def record(base, count):
    # Fixed prefix forces the random choice into a speculative verification step.
    prompt = post(base, '/apply-template', {'messages': [{'role': 'user', 'content':
        'Choose A, B, C or D at random; all four choices are equally welcome. ' +
        'Context: a red blue green yellow board. ' * 32}], 'add_generation_prompt': True})['tokens']
    rows = []
    for seed in range(count):
        reply = post(base, '/completion', dict(prompt=prompt, n_predict=12, seed=seed,
            temperature=2.0, top_k=0, top_p=1.0, min_p=0, cache_prompt=False,
            return_tokens=True, n_probs=8, post_sampling_probs=True,
            grammar='root ::= "Answer: " [ABCD]'))
        text = reply['content']
        assert text in ['Answer: '+c for c in 'ABCD'], text
        probabilities = reply['completion_probabilities']
        assert len(probabilities) == len(reply['tokens'])
        assert all(row['top_probs'] for row in probabilities)
        rows.append(dict(seed=seed, content=text, tokens=reply['tokens'], probabilities=probabilities))
        print(seed, text, flush=True)
    return rows


def compare(left, right):
    assert len(left) == len(right) and len(left) >= 32
    a, b = (collections.Counter(row['content'][-1] for row in r) for r in (left, right))
    n = len(left)
    assert len(a) >= 2 and len(b) >= 2, 'degenerate categorical test'
    # Four conservative binomial bounds: sanity check, not a universal equivalence proof.
    for c in 'ABCD':
        p = (a[c]+b[c])/(2*n)
        bound = 4*math.sqrt(max(p*(1-p), 1/n)*2/n)+1/n
        assert abs(a[c]-b[c])/n <= bound, (c,a,b,bound)
    print(json.dumps(dict(samples=n, off=a, on=b,
        total_variation=sum(abs(a[c]-b[c]) for c in 'ABCD')/(2*n)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8090')
    p.add_argument('--count', type=int, default=64)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--record', type=Path)
    g.add_argument('--compare', nargs=2, type=Path)
    args = p.parse_args()
    if args.record:
        args.record.write_text(json.dumps(record(args.url,args.count),ensure_ascii=False,indent=2)+'\n')
    else:
        compare(*(json.loads(path.read_text()) for path in args.compare))
