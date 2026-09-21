#!/usr/bin/env python3
"""엔진 자가 검사. 고친 뒤에는 이걸 먼저 돌린다.

리그는 결정론 위에 서 있다. 같은 봇, 같은 씨앗인데 결과가 다르면
순위표가 의미를 잃는다. 그래서 제일 먼저 검사한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiarena.core.spec import SCENARIOS, OBS_DIM, N_ACTIONS
from aiarena.core.match import run_match
from aiarena.core.world import World, RESULT_TIMEOUT
from aiarena.core.obs import build_obs
from aiarena.bots.baselines import BASELINES

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else '실패'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    sc = SCENARIOS["raider_vs_biter"]
    print("=" * 66)
    print("엔진 자가 검사")
    print("=" * 66)

    # 1. 결정론 — 같은 씨앗이면 같은 판
    same = True
    for seed in (1, 77, 3001):
        a = run_match(sc, BASELINES["kite"], BASELINES["rush"], seed=seed)
        b = run_match(sc, BASELINES["kite"], BASELINES["rush"], seed=seed)
        if (a.result, a.steps, a.frames, a.survivors) != (b.result, b.steps, b.frames, b.survivors):
            same = False
    check("결정론: 같은 씨앗 = 같은 결과", same)

    # 2. 씨앗이 다르면 판도 달라야 한다 (안 그러면 N판이 사실상 1판이다)
    outs = {run_match(sc, BASELINES["kite"], BASELINES["rush"], seed=s).frames
            for s in range(100, 130)}
    check("씨앗이 다르면 판도 다르다", len(outs) > 3, f"서로 다른 길이 {len(outs)}종")

    # 3. 관측 모양
    w = World(sc)
    w.reset(7)
    o0 = build_obs(w, 0)
    o1 = build_obs(w, 1)
    check("관측 모양 P0", o0.shape == (sc.p0_count, OBS_DIM), str(o0.shape))
    check("관측 모양 P1", o1.shape == (sc.p1_count, OBS_DIM), str(o1.shape))

    # 4. 죽은 유닛 줄은 0이다
    w.teams[1][0].alive = False
    o1 = build_obs(w, 1)
    check("죽은 유닛 관측은 전부 0", float(abs(o1[0]).sum()) == 0.0)

    # 5. 아무 행동이나 넣어도 안 터진다 (참가자 봇은 뭘 낼지 모른다)
    def wild(obs, mask):
        return [(i * 37 + 11) for i in range(len(obs))]   # 범위 밖 정수
    try:
        r = run_match(sc, wild, wild, seed=5)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print("      예외:", e)
    check("범위 밖 행동을 내도 판이 끝난다", ok)

    # 6. 아무것도 안 하면 시간초과 (양쪽 다 정지끼리)
    r = run_match(sc, BASELINES["stop"], BASELINES["stop"], seed=3001)
    check("정지끼리는 시간초과", r.result == RESULT_TIMEOUT, f"결과 {r.result}")

    # 7. 속도: 리그는 판을 많이 돌려야 한다
    import time
    t0 = time.time()
    N = 50
    for e in range(N):
        run_match(sc, BASELINES["kite"], BASELINES["rush"], seed=1000 + e)
    dt = time.time() - t0
    check("속도", dt < 10.0, f"{N}판 {dt:.2f}초 = 판당 {dt/N*1000:.0f}ms")

    print()
    if FAIL:
        print(f"★ {len(FAIL)}개 실패: {', '.join(FAIL)}")
        return 1
    print("★ 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
