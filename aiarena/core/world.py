"""전투 판 하나를 굴리는 엔진.

설계 원칙 세 가지:
  1. 결정론. 같은 씨앗이면 언제 어디서 돌려도 같은 판이 나온다. 리그의 생명이다.
  2. 판은 데이터다. 이 파일은 어떤 유닛이 나오는지 모른다. ScenarioSpec 만 받는다.
  3. 명령은 판단 주기마다, 물리는 프레임마다. 둘을 섞지 않는다.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence
import math
import random

from .spec import (
    ScenarioSpec, UnitSpec, DIRS, MOVE_PX, N_ACTIONS,
    ACT_MOVE_LO, ACT_MOVE_HI, ACT_ATTACK_NEAR, ACT_ATTACK_WEAK, ACT_RETREAT,
)

# 명령 종류
ORD_STOP = 0      # 명령 없음. 스스로 적을 찾아 쫓아가서 싸운다 (아래 _stop_behavior 참고)
ORD_MOVE = 1      # 목표 지점으로 간다. 가는 동안은 쏘지 않는다
ORD_ATTACK = 2    # 지정한 적을 쏜다. 사거리 밖이면 아무것도 하지 않는다 (따라가지 않는다)

# 이 속력보다 빠르게 움직이는 동안은 공격이 나가지 않는다 (px/frame).
# 원본 대전표에 맞추기 위해 스캔으로 정한 값이다 — tools/tune_fire.py 참고.
import os as _os
FIRE_SPEED_MAX = float(_os.environ.get("AIARENA_FIRE_SPEED_MAX", "2.0"))

# 판 결과
RESULT_RUNNING = 0
RESULT_P0_WIN = 1
RESULT_P1_WIN = 2
RESULT_TIMEOUT = 3


@dataclass
class Unit:
    idx: int          # 팀 안에서의 번호 (죽어도 유지된다 — 관측 자리가 고정되어야 하므로)
    team: int
    spec: UnitSpec
    x: float
    y: float
    hp: float
    cd: int = 0
    alive: bool = True
    order: int = ORD_STOP
    tx: float = 0.0           # ORD_MOVE 목표
    ty: float = 0.0
    target: Optional["Unit"] = None      # ORD_ATTACK 대상
    attacker: Optional["Unit"] = None    # 마지막으로 나를 때린 적 (반격용)
    speed_now: float = 0.0               # 현재 속력 (크기만)
    heading: float = 0.0                 # 현재 진행 방향 (라디안)

    @property
    def half(self) -> float:
        return self.spec.half


def box_distance(a: Unit, b: Unit) -> float:
    """두 유닛의 충돌 박스 사이 최단 거리. 겹쳐 있으면 0이다.

    원본 엔진이 사거리를 잴 때 쓰는 방식이라 그대로 맞췄다. 중심 거리로 재면
    덩치 큰 유닛의 사거리가 실제보다 짧아져서 근접 유닛이 영원히 못 때린다.
    """
    dx = abs(a.x - b.x) - (a.half + b.half)
    dy = abs(a.y - b.y) - (a.half + b.half)
    dx = dx if dx > 0.0 else 0.0
    dy = dy if dy > 0.0 else 0.0
    return math.hypot(dx, dy)


def center_distance(a: Unit, b: Unit) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class World:
    """판 하나. reset() 으로 새 판을 열고, step() 으로 판단 한 번을 흘린다."""

    def __init__(self, scenario: ScenarioSpec):
        self.sc = scenario
        self.teams: List[List[Unit]] = [[], []]
        self.step_count = 0
        self.frame = 0
        self.result = RESULT_RUNNING
        self.misfire = [0, 0]     # 사거리 밖인데 쏘라고 한 횟수
        self._rng = random.Random(0)

    # ── 판 열기 ──────────────────────────────────────────────────────────
    def reset(self, seed: int) -> None:
        sc = self.sc
        self._rng = random.Random(seed)
        gap = self._rng.randint(sc.gap_lo, sc.gap_hi)
        spread = self._rng.randint(sc.spread_lo, sc.spread_hi)

        self.teams = [[], []]
        for team, (unit_key, count, sign) in enumerate(
            ((sc.p0_unit, sc.p0_count, -1), (sc.p1_unit, sc.p1_count, +1))
        ):
            from .spec import UNITS
            spec = UNITS[unit_key]
            y0 = sc.cy - (count - 1) * spread / 2.0
            for i in range(count):
                self.teams[team].append(Unit(
                    idx=i, team=team, spec=spec,
                    x=float(sc.cx + sign * gap // 2),
                    y=float(y0 + i * spread),
                    hp=spec.hp,
                ))
        self.step_count = 0
        self.frame = 0
        self.result = RESULT_RUNNING
        self.misfire = [0, 0]

    # ── 조회 ────────────────────────────────────────────────────────────
    def alive_units(self, team: int) -> List[Unit]:
        return [u for u in self.teams[team] if u.alive]

    def nearest_foe(self, u: Unit) -> Optional[Unit]:
        best, best_d = None, 1e18
        for z in self.teams[1 - u.team]:
            if not z.alive:
                continue
            d = center_distance(u, z)
            if d < best_d:
                best, best_d = z, d
        return best

    def weakest_foe(self, u: Unit) -> Optional[Unit]:
        best, best_hp = None, 1e18
        for z in self.teams[1 - u.team]:
            if not z.alive:
                continue
            if z.hp < best_hp:
                best, best_hp = z, z.hp
        return best

    def in_range(self, u: Unit, t: Optional[Unit]) -> bool:
        if t is None or not t.alive:
            return False
        return box_distance(u, t) <= u.spec.range

    # ── 명령 주기 ────────────────────────────────────────────────────────
    def apply_actions(self, team: int, actions: Sequence[int], strict_fire: bool = True) -> None:
        """판단 한 번. 죽은 유닛의 행동은 무시한다."""
        units = self.teams[team]
        for i, u in enumerate(units):
            if not u.alive:
                continue
            a = int(actions[i]) % N_ACTIONS

            if a == 0:
                u.order, u.target = ORD_STOP, None
                continue

            if ACT_MOVE_LO <= a <= ACT_MOVE_HI:
                dx, dy = DIRS[a - 1]
                u.order = ORD_MOVE
                u.tx, u.ty = u.x + dx * MOVE_PX, u.y + dy * MOVE_PX
                u.target = None
                continue

            if a == ACT_RETREAT:
                z = self.nearest_foe(u)
                if z is None:
                    u.order, u.target = ORD_STOP, None
                    continue
                dx, dy = u.x - z.x, u.y - z.y
                L = max(1.0, math.hypot(dx, dy))
                u.order = ORD_MOVE
                u.tx, u.ty = u.x + dx / L * MOVE_PX, u.y + dy / L * MOVE_PX
                u.target = None
                continue

            # 공격 명령
            z = self.nearest_foe(u) if a == ACT_ATTACK_NEAR else self.weakest_foe(u)
            if strict_fire and not self.in_range(u, z):
                # 사거리 밖에서 쏘라고 했다 = 헛방. 제자리에 선다.
                # 이게 없으면 공격 명령이 알아서 접근해버려서, 움직이는 법을 배울 이유가 사라진다.
                self.misfire[team] += 1
                u.order, u.target = ORD_STOP, None
                continue
            u.order, u.target = ORD_ATTACK, z

    # ── 물리 프레임 ──────────────────────────────────────────────────────
    def tick(self) -> None:
        """프레임 하나. 이동 → 공격 → 쿨다운 → 겹침 해소 순서다."""
        order = [u for t in self.teams for u in t if u.alive]

        for u in order:
            if u.order == ORD_MOVE:
                self._move(u)
            elif u.order == ORD_ATTACK:
                self._brake(u)   # 쏘려면 서야 한다. 미끄러지는 유닛은 서는 데도 시간이 든다

        for u in order:
            if not u.alive:
                continue
            if u.order == ORD_ATTACK:
                self._try_fire(u, u.target)
            elif u.order == ORD_STOP:
                self._stop_behavior(u)

        for u in order:
            if u.cd > 0:
                u.cd -= 1

        self._separate()
        self._clamp()
        self.frame += 1

    def _stop_behavior(self, u: Unit) -> None:
        """명령이 없는 유닛이 스스로 하는 일.

        원본 게임의 '정지'가 제자리에 못 박혀 있는 게 아니라는 점이 중요하다.
        사거리에 적이 들어오면 쏘고, 안 들어오면 **쫓아간다.** 그리고 멀리서 맞으면
        때린 놈을 향해 달려간다. 이게 없으면 사거리 긴 쪽이 일방적으로 이겨서
        판이 통째로 무너진다.
        """
        t = self._auto_target(u)
        if t is not None:
            self._brake(u)
            self._try_fire(u, t)
            return
        # 사거리 밖 — 쫓아갈 대상을 고른다. 나를 때린 놈이 1순위다.
        # 단 무한정 쫓지는 않는다. 획득 범위를 크게 벗어나면 포기한다 —
        # 이걸 안 걸면 '명령 없음'이 사람이 짠 추격보다 똑똑해져서 판이 거꾸로 선다.
        chase = u.spec.acquire * float(_os.environ.get("AIARENA_CHASE_MULT", "2.0"))
        z = None
        if u.attacker is not None and u.attacker.alive and box_distance(u, u.attacker) <= chase:
            z = u.attacker
        else:
            u.attacker = None
            best, best_d = None, 1e18
            for f in self.teams[1 - u.team]:
                if not f.alive:
                    continue
                d = box_distance(u, f)
                if d <= u.spec.acquire and d < best_d:
                    best, best_d = f, d
            z = best
        if z is None:
            self._brake(u)
            return
        self._steer(u, z.x, z.y)

    def _clamp(self) -> None:
        w, h = self.sc.map_w, self.sc.map_h
        for team in self.teams:
            for u in team:
                if not u.alive:
                    continue
                half = u.half
                if u.x < half:
                    u.x = half
                elif u.x > w - half:
                    u.x = w - half
                if u.y < half:
                    u.y = half
                elif u.y > h - half:
                    u.y = h - half

    def _move(self, u: Unit) -> None:
        self._steer(u, u.tx, u.ty)

    def _steer(self, u: Unit, tx: float, ty: float) -> None:
        """목표 지점을 향해 한 프레임 움직인다.

        걸어다니는 유닛(accel = 0)은 즉시 방향을 바꾸고 즉시 최고속으로 간다.

        미끄러지는 유닛은 다르다. 가고 싶은 방향으로 **선회**한다 — 뒤로 가려고
        멈췄다가 다시 가속하는 게 아니라, 속력을 유지한 채 방향을 돌린다.
        이게 무빙샷이 성립하는 이유다. 멈췄다 가는 모델로 만들면 도망치는 쪽이
        매번 재가속 손해를 봐서 사거리가 긴 쪽이 오히려 잡힌다.
        """
        spd = u.spec.speed
        acc = u.spec.accel
        dx, dy = tx - u.x, ty - u.y
        d = math.hypot(dx, dy)

        if acc <= 0.0:
            if d <= 1e-6:
                u.speed_now = 0.0
                return
            u.heading = math.atan2(dy, dx)
            step = spd if d > spd else d
            u.speed_now = step
            u.x += math.cos(u.heading) * step
            u.y += math.sin(u.heading) * step
            return

        if d <= 1e-6:
            self._brake(u)
            return

        want = math.atan2(dy, dx)
        turn = math.radians(u.spec.turn)
        diff = (want - u.heading + math.pi) % (2 * math.pi) - math.pi
        if diff > turn:
            diff = turn
        elif diff < -turn:
            diff = -turn
        u.heading += diff

        u.speed_now = min(spd, u.speed_now + acc)
        step = u.speed_now
        if step >= d:
            u.x, u.y = tx, ty
            u.speed_now = 0.0
            return
        u.x += math.cos(u.heading) * step
        u.y += math.sin(u.heading) * step

    def _brake(self, u: Unit) -> None:
        """명령이 없어 서려는 중. 미끄러지는 유닛은 바로 못 선다."""
        acc = u.spec.accel
        if acc <= 0.0 or u.speed_now <= acc:
            u.speed_now = 0.0
            return
        u.speed_now -= acc
        u.x += math.cos(u.heading) * u.speed_now
        u.y += math.sin(u.heading) * u.speed_now

    def _auto_target(self, u: Unit) -> Optional[Unit]:
        """제자리에 선 유닛이 스스로 고르는 표적 — 사거리 안에서 가장 가까운 적."""
        best, best_d = None, 1e18
        for z in self.teams[1 - u.team]:
            if not z.alive:
                continue
            d = box_distance(u, z)
            if d <= u.spec.range and d < best_d:
                best, best_d = z, d
        return best

    def _try_fire(self, u: Unit, t: Optional[Unit]) -> None:
        if t is None or not t.alive or u.cd > 0:
            return
        # ★ 달리면서는 못 쏜다. 서야 방아쇠가 당겨진다.
        # 이게 무빙샷의 진짜 대가다 — 한 방 쏘려면 서는 시간과 다시 붙이는 시간을 낸다.
        # 이 조건이 없으면 사거리 긴 쪽이 최고속으로 달아나며 쏴서 영원히 안 잡힌다.
        if u.speed_now > FIRE_SPEED_MAX:
            return
        if box_distance(u, t) > u.spec.range:
            return
        t.hp -= u.spec.damage
        t.attacker = u          # 맞은 쪽은 때린 놈을 기억한다 (명령 없을 때 반격하러 간다)
        u.cd = u.spec.cooldown
        if t.hp <= 0.0:
            t.alive = False
            t.hp = 0.0

    def _separate(self) -> None:
        """겹친 유닛을 밀어낸다. 한 번만 푼다 — 완전히 안 겹치게 하면
        떼로 몰려오는 쪽이 벽에 막힌 것처럼 굳어버린다."""
        units = [u for t in self.teams for u in t if u.alive]
        n = len(units)
        for i in range(n):
            a = units[i]
            for j in range(i + 1, n):
                b = units[j]
                rx = (a.half + b.half)
                dx, dy = b.x - a.x, b.y - a.y
                ox = rx - abs(dx)
                oy = rx - abs(dy)
                if ox <= 0.0 or oy <= 0.0:
                    continue
                # 덜 파고든 축으로만 민다 (박스 충돌의 최소 이동 거리)
                if ox < oy:
                    push = ox / 2.0
                    s = 1.0 if dx >= 0 else -1.0
                    a.x -= push * s
                    b.x += push * s
                else:
                    push = oy / 2.0
                    s = 1.0 if dy >= 0 else -1.0
                    a.y -= push * s
                    b.y += push * s

    # ── 판단 한 번 = frame_skip 프레임 ───────────────────────────────────
    def advance(self) -> int:
        for _ in range(self.sc.frame_skip):
            self.tick()
            r = self._check_end()
            if r != RESULT_RUNNING:
                self.result = r
                return r
        self.step_count += 1
        if self.step_count >= self.sc.max_steps:
            self.result = RESULT_TIMEOUT
            return RESULT_TIMEOUT
        return RESULT_RUNNING

    def _check_end(self) -> int:
        a0 = any(u.alive for u in self.teams[0])
        a1 = any(u.alive for u in self.teams[1])
        if not a1 and not a0:
            return RESULT_TIMEOUT      # 동시 전멸은 무승부로 본다
        if not a1:
            return RESULT_P0_WIN
        if not a0:
            return RESULT_P1_WIN
        return RESULT_RUNNING
