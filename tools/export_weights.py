#!/usr/bin/env python3
"""학습한 망을 브라우저가 읽을 JSON 으로 뺀다.

정책망은 39 → 128 → 128 → 12 짜리 작은 망이라, 가중치를 통째로 실어도
100KB 대다. 브라우저에서 라이브러리 없이 행렬곱 세 번이면 추론이 끝난다.
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aiarena.train.ppo import Policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--digits", type=int, default=5, help="소수 자릿수 (파일 크기)")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    p = Policy(); p.load_state_dict(ck["model"]); p.eval()
    d = p.state_dict()

    def arr(t):
        return [round(float(v), args.digits) for v in t.flatten()]

    out = {
        "shape": [39, 128, 128, 12],
        "w0": arr(d["body.0.weight"]), "b0": arr(d["body.0.bias"]),
        "w1": arr(d["body.2.weight"]), "b1": arr(d["body.2.bias"]),
        "w2": arr(d["pi.weight"]),     "b2": arr(d["pi.bias"]),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{args.out}  {os.path.getsize(args.out)/1024:.0f}KB")


if __name__ == "__main__":
    main()
