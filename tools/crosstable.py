#!/usr/bin/env python3
"""기준선 4종을 전부 맞붙여 대전표를 뽑는다.

이 표가 판의 품질 검사지다. 두 가지를 본다:

  1. 균형 칸이 있는가 — 어느 한 칸이 40~60% 사이여야 한다.
     전부 0:24 같은 표면 실력이 승부를 못 가르는 판이라 리그를 못 연다.
  2. 원본 판의 표와 구조가 같은가 — 어느 전략이 어느 전략을 이기는 순서가 같아야 한다.

사용:
    python tools/crosstable.py                    # 기본 판, 24판씩
    python tools/crosstable.py --episodes 12      # 빠르게
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiarena.core.spec import SCENARIOS
from aiarena.core.match import run_match
from aiarena.core.world import RESULT_P0_WIN, RESULT_P1_WIN
from aiarena.bots.baselines import BASELINES, BASELINE_KR


def duel(scenario, f0, f1, episodes, seed):
    w0 = w1 = to = 0
    steps = 0
    for e in range(episodes):
        r = run_match(scenario, f0, f1, seed=seed + e * 7919)
        if r.result == RESULT_P0_WIN:
            w0 += 1
        elif r.result == RESULT_P1_WIN:
            w1 += 1
        else:
            to += 1
        steps += r.steps
    return w0, w1, to, steps / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="raider_vs_biter")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="*", default=[3001, 4242])
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    names = list(BASELINES)
    per_seed = max(1, args.episodes // len(args.seeds))

    print("=" * 78)
    print(f"판: {sc.key}   "
          f"{sc.p0_unit}×{sc.p0_count} (P0, 왼쪽)  vs  {sc.p1_unit}×{sc.p1_count} (P1, 오른쪽)")
    print(f"판단주기 {sc.frame_skip}프레임 · 최대 {sc.max_steps}판단 · "
          f"{per_seed * len(args.seeds)}판씩 (시드 {len(args.seeds)}종)")
    print("=" * 78)

    t0 = time.time()
    header = "  " + "P0 \\ P1".ljust(10) + "".join(f"{BASELINE_KR[n]:>14}" for n in names)
    print(header)
    cells = {}
    for n0 in names:
        row = "  " + BASELINE_KR[n0].ljust(10)
        for n1 in names:
            w0 = w1 = to = 0
            for sd in args.seeds:
                a, b, c, _ = duel(sc, BASELINES[n0], BASELINES[n1], per_seed, sd)
                w0 += a; w1 += b; to += c
            cells[(n0, n1)] = (w0, w1, to)
            mark = "*" if to > (w0 + w1 + to) / 2 else " "
            row += f"{w0:>5}:{w1:<4}{mark}   "
        print(row)
    print("   (* = 시간초과가 절반을 넘은 칸)")

    # 균형 칸 찾기
    print()
    total = per_seed * len(args.seeds)
    best = None
    for (n0, n1), (w0, w1, to) in cells.items():
        decided = w0 + w1
        if decided < total * 0.5:       # 대부분 시간초과면 균형이 아니라 교착이다
            continue
        rate = w0 / decided
        gap = abs(rate - 0.5)
        if best is None or gap < best[0]:
            best = (gap, n0, n1, w0, w1, to, rate)

    if best is None:
        print("★ 균형 칸 없음 — 전부 일방적이거나 교착이다. 이 판은 아직 리그용이 아니다.")
    else:
        _, n0, n1, w0, w1, to, rate = best
        verdict = "합격" if 0.40 <= rate <= 0.60 else "아직"
        print(f"★ 가장 팽팽한 칸: (P0 {BASELINE_KR[n0]}, P1 {BASELINE_KR[n1]}) = "
              f"{w0}:{w1} (시간초과 {to})  →  P0 승률 {rate*100:.1f}%  [{verdict}]")
    print(f"\n걸린 시간 {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
