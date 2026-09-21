"""학습용 환경.

한쪽(P0)을 학습시키고, 상대(P1)는 정해진 봇이 잡는다.
관측과 행동은 봇 계약과 똑같다 — 학습한 망을 그대로 봇으로 낼 수 있다는 뜻이다.
"""
from typing import Callable, Optional, Sequence, Tuple
import numpy as np

from .spec import ScenarioSpec
from .obs import build_obs, alive_mask
from .reward import RewardConfig
from .world import (
    World, RESULT_RUNNING, RESULT_P0_WIN, RESULT_P1_WIN, RESULT_TIMEOUT,
)


class ArenaEnv:
    """판 하나를 학습용으로 감싼 것.

    step() 한 번 = 판단 한 번. 팀 전체가 한꺼번에 행동한다.
    보상은 팀 전체에 하나로 준다 (유닛별로 쪼개지 않는다).
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        foe_bot: Callable,
        reward: Optional[RewardConfig] = None,
        learn_team: int = 0,
    ):
        self.sc = scenario
        self.foe_bot = foe_bot
        self.rc = reward or RewardConfig()
        self.team = learn_team
        self.foe_team = 1 - learn_team
        self.w = World(scenario)
        self._prev_my_hp = 0.0
        self._prev_foe_hp = 0.0
        self._prev_my_n = 0
        self._prev_foe_n = 0

    # ── 상태 요약 ───────────────────────────────────────────────────
    def _hp_sum(self, team: int) -> float:
        return sum(u.hp for u in self.w.teams[team] if u.alive)

    def _count(self, team: int) -> int:
        return len(self.w.alive_units(team))

    def _snapshot(self) -> None:
        self._prev_my_hp = self._hp_sum(self.team)
        self._prev_foe_hp = self._hp_sum(self.foe_team)
        self._prev_my_n = self._count(self.team)
        self._prev_foe_n = self._count(self.foe_team)

    # ── 표준 인터페이스 ─────────────────────────────────────────────
    def reset(self, seed: int) -> np.ndarray:
        self.w.reset(seed)
        self._snapshot()
        return build_obs(self.w, self.team)

    def step(self, actions: Sequence[int]) -> Tuple[np.ndarray, float, bool, dict]:
        # 상대 행동
        fo = build_obs(self.w, self.foe_team)
        fm = alive_mask(self.w, self.foe_team)
        fa = self.foe_bot(fo, fm)

        self.w.apply_actions(self.team, actions)
        self.w.apply_actions(self.foe_team, fa)
        result = self.w.advance()

        # 보상: 이번 판단 동안 벌어진 일로 센다
        rc = self.rc
        my_hp, foe_hp = self._hp_sum(self.team), self._hp_sum(self.foe_team)
        my_n, foe_n = self._count(self.team), self._count(self.foe_team)
        my_max = self.w.teams[self.team][0].spec.hp * len(self.w.teams[self.team])
        foe_max = self.w.teams[self.foe_team][0].spec.hp * len(self.w.teams[self.foe_team])

        r = 0.0
        r += (self._prev_foe_hp - foe_hp) / foe_max * rc.enemy_hp
        r -= (self._prev_my_hp - my_hp) / my_max * rc.ally_hp
        r += (self._prev_foe_n - foe_n) * rc.kill
        r -= (self._prev_my_n - my_n) * rc.death
        r -= rc.step

        done = result != RESULT_RUNNING
        if done:
            # ★ 승/패/시간초과는 서로 배타다. 더하지 않는다.
            win = RESULT_P0_WIN if self.team == 0 else RESULT_P1_WIN
            if result == win:
                r += rc.win
            elif result == RESULT_TIMEOUT:
                r -= rc.timeout
            else:
                r -= rc.lose

        self._snapshot()
        obs = build_obs(self.w, self.team)
        info = {
            "result": result,
            "steps": self.w.step_count,
            "my_alive": my_n,
            "foe_alive": foe_n,
            "misfire": self.w.misfire[self.team],
        }
        return obs, r, done, info

    @property
    def n_units(self) -> int:
        sc = self.sc
        return sc.p0_count if self.team == 0 else sc.p1_count
