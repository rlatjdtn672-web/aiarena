"""World 의 상태를 봇이 보는 숫자로 바꾼다.

관측은 (유닛 수, OBS_DIM) 모양이다. 죽은 유닛의 줄은 전부 0이다.
줄 번호는 판이 끝날 때까지 고정된다 — 죽었다고 밀려 올라오면 봇이 누가 누군지 잃는다.
"""
from typing import List
import numpy as np

from .spec import (
    OBS_DIM, SELF_DIM, FOE_DIM, ALLY_DIM, N_FOE_OBS, N_ALLY_OBS,
)
from .world import World, Unit, box_distance, center_distance


def build_obs(w: World, team: int) -> np.ndarray:
    units = w.teams[team]
    me_spec = units[0].spec
    foes = w.teams[1 - team]
    foe_max_hp = foes[0].spec.hp if foes else 1.0
    cd_max = max(1, me_spec.cooldown)
    cx, cy = w.sc.cx, w.sc.cy

    out = np.zeros((len(units), OBS_DIM), dtype=np.float32)
    for i, u in enumerate(units):
        if not u.alive:
            continue
        o = out[i]
        o[0] = u.hp / me_spec.hp
        o[1] = min(u.cd, cd_max) / cd_max
        o[2] = 1.0
        o[3] = (u.x - cx) / 512.0
        o[4] = (u.y - cy) / 512.0

        # 가까운 적부터. 정렬 기준은 중심 거리다 (사거리 판정은 박스 거리라 따로 넣는다)
        zs = sorted(
            ((center_distance(u, z), z) for z in foes if z.alive),
            key=lambda p: p[0],
        )
        for k in range(min(N_FOE_OBS, len(zs))):
            d, z = zs[k]
            q = SELF_DIM + k * FOE_DIM
            o[q + 0] = (z.x - u.x) / 256.0
            o[q + 1] = (z.y - u.y) / 256.0
            o[q + 2] = d / 256.0
            o[q + 3] = z.hp / foe_max_hp
            o[q + 4] = 1.0
            o[q + 5] = 1.0 if box_distance(u, z) <= me_spec.range else 0.0

        c = 0
        base = SELF_DIM + N_FOE_OBS * FOE_DIM
        for j, m in enumerate(units):
            if j == i or not m.alive or c >= N_ALLY_OBS:
                continue
            q = base + c * ALLY_DIM
            o[q + 0] = (m.x - u.x) / 256.0
            o[q + 1] = (m.y - u.y) / 256.0
            o[q + 2] = m.hp / me_spec.hp
            o[q + 3] = 1.0
            c += 1
    return out


def alive_mask(w: World, team: int) -> List[int]:
    return [1 if u.alive else 0 for u in w.teams[team]]
