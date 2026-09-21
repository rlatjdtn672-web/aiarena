#!/usr/bin/env python3
"""여러 봇을 한 표로 비교한다.

손코딩 기준선과 학습한 망을 같은 상대들에게 붙여서 줄줄이 세운다.
리그 순위표의 축소판이고, "내 봇이 기준선을 넘었나"를 한눈에 본다.

    python tools/compare.py                                   # 손코딩 4종만
    python tools/compare.py runs/clone/cloned.pt runs/clone_ft/best.pt
    python tools/compare.py --label 베낀망 runs/clone/cloned.pt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiarena.core.spec import SCENARIOS
from aiarena.core.match import run_match
from aiarena.core.world import RESULT_P0_WIN, RESULT_P1_WIN
from aiarena.bots.baselines import BASELINES, BASELINE_KR

FOES = ["rush", "kite", "spread"]     # 정지는 서로 시간초과라 뺀다


def score(scenario, bot, episodes, seed):
    cells, tw, tl, tt = [], 0, 0, 0
    for f in FOES:
        w = l = t = 0
        for e in range(episodes):
            r = run_match(scenario, bot, BASELINES[f], seed=seed + e * 7919)
            if r.result == RESULT_P0_WIN:
                w += 1
            elif r.result == RESULT_P1_WIN:
                l += 1
            else:
                t += 1
        cells.append((w, l, t))
        tw += w; tl += l; tt += t
    return cells, tw, tl, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", help="비교할 .pt 파일들")
    ap.add_argument("--scenario", default="raider_vs_biter")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=99000)
    ap.add_argument("--label", action="append", default=[], help="체크포인트 이름 (순서대로)")
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    total = args.episodes * len(FOES)

    entries = [(f"손코딩 {BASELINE_KR[k]}", BASELINES[k]) for k in ("rush", "kite", "spread")]

    if args.checkpoints:
        import torch
        from aiarena.train.ppo import Policy, make_bot
        for i, path in enumerate(args.checkpoints):
            ck = torch.load(path, weights_only=False)
            p = Policy(); p.load_state_dict(ck["model"]); p.eval()
            name = args.label[i] if i < len(args.label) else os.path.basename(os.path.dirname(path))
            entries.append((f"★ {name}", make_bot(p, greedy=True)))

    print("=" * 74)
    print(f"판 {sc.key} · 상대마다 {args.episodes}판 · 합계 {total}판")
    print("=" * 74)
    print("  " + "".ljust(18) + "".join(f"{BASELINE_KR[f]:>12}" for f in FOES) + f"{'합계':>14}")

    for name, bot in entries:
        cells, tw, tl, tt = score(sc, bot, args.episodes, args.seed)
        row = "  " + name.ljust(18)
        for (w, l, t) in cells:
            row += f"{w:>5}:{l:<5}"
        row += f"{tw:>7}/{total}  {tw/total*100:5.1f}%"
        print(row)

    print()
    print("  ※ '정지'는 양쪽이 서로 안 움직여 시간초과가 나므로 상대에서 뺐다")


if __name__ == "__main__":
    main()
