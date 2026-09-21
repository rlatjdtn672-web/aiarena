"""봇 둘을 붙여서 판을 돌린다.

봇 계약은 이것뿐이다:

    bot(obs, mask) -> 행동 리스트

  obs  : (유닛 수, 39) float32. 죽은 유닛 줄은 0
  mask : 길이 = 유닛 수. 살아 있으면 1
  반환 : 길이 = 유닛 수의 정수 리스트 (0~11)

이 계약만 지키면 강화학습이든 if 문 덩어리든 상관없다.
판이 바뀌어도 계약은 안 바뀐다 — 유닛 수와 관측 길이만 달라진다.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence
import numpy as np

from .spec import ScenarioSpec
from .obs import build_obs, alive_mask
from .world import (
    World, RESULT_RUNNING, RESULT_P0_WIN, RESULT_P1_WIN, RESULT_TIMEOUT,
)

Bot = Callable[[np.ndarray, Sequence[int]], Sequence[int]]


@dataclass
class MatchResult:
    result: int                 # RESULT_* 중 하나
    steps: int
    frames: int
    survivors: tuple            # (P0 생존 수, P1 생존 수)
    misfire: tuple              # (P0 헛방, P1 헛방)
    trace: Optional[List[dict]] = field(default=None)

    @property
    def p0_win(self) -> bool:
        return self.result == RESULT_P0_WIN

    @property
    def p1_win(self) -> bool:
        return self.result == RESULT_P1_WIN

    @property
    def timeout(self) -> bool:
        return self.result == RESULT_TIMEOUT


def _snapshot(w: World) -> dict:
    return {
        "f": w.frame,
        "u": [
            [
                {"i": u.idx, "x": round(u.x, 1), "y": round(u.y, 1),
                 "hp": round(u.hp, 1), "a": 1 if u.alive else 0}
                for u in team
            ]
            for team in w.teams
        ],
    }


def run_match(
    scenario: ScenarioSpec,
    bot0: Bot,
    bot1: Bot,
    seed: int,
    strict_fire: bool = True,
    record: bool = False,
) -> MatchResult:
    """판 하나. 같은 씨앗이면 언제 돌려도 같은 결과가 나온다."""
    w = World(scenario)
    w.reset(seed)
    trace: Optional[List[dict]] = [] if record else None

    while True:
        if record:
            trace.append(_snapshot(w))

        o0, m0 = build_obs(w, 0), alive_mask(w, 0)
        o1, m1 = build_obs(w, 1), alive_mask(w, 1)
        a0 = bot0(o0, m0)
        a1 = bot1(o1, m1)
        w.apply_actions(0, a0, strict_fire=strict_fire)
        w.apply_actions(1, a1, strict_fire=strict_fire)

        r = w.advance()
        if r != RESULT_RUNNING:
            if record:
                trace.append(_snapshot(w))
            return MatchResult(
                result=r,
                steps=w.step_count,
                frames=w.frame,
                survivors=(len(w.alive_units(0)), len(w.alive_units(1))),
                misfire=tuple(w.misfire),
                trace=trace,
            )


def run_series(
    scenario: ScenarioSpec,
    bot0_factory: Callable[[], Bot],
    bot1_factory: Callable[[], Bot],
    episodes: int,
    seed: int = 12345,
) -> dict:
    """같은 조합으로 여러 판. 씨앗은 판마다 다르게 굴린다.

    봇을 함수가 아니라 '만드는 함수'로 받는 이유: 판 사이에 상태를 남기는
    봇(예: 직전 행동을 기억하는 봇)이 있어도 판마다 새로 시작하게 하기 위해서다.
    """
    w0 = w1 = to = 0
    steps_sum = 0
    for e in range(episodes):
        r = run_match(scenario, bot0_factory(), bot1_factory(), seed=seed + e * 7919)
        if r.p0_win:
            w0 += 1
        elif r.p1_win:
            w1 += 1
        else:
            to += 1
        steps_sum += r.steps
    return {
        "p0_win": w0, "p1_win": w1, "timeout": to,
        "episodes": episodes,
        "avg_steps": steps_sum / episodes if episodes else 0.0,
    }
