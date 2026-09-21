"""예제 봇. 이 파일을 복사해서 고치면 된다.

여기 있는 건 '무빙샷'이다 — 쏠 수 있으면 쏘고, 쿨다운 도는 동안은 물러난다.
기본 봇 넷 중에서는 제일 강하지만 완벽하지 않다. 표를 보면 여섯 마리가
벌려서 달려들면 진다. 그 칸을 이기는 게 숙제다.
"""
import math

# 행동 번호
STOP = 0
ATTACK_NEAR = 9
ATTACK_WEAK = 10
RETREAT = 11

# 관측에서 자주 쓰는 자리
MY_HP = 0
MY_COOLDOWN = 1          # 0 이면 지금 쏠 수 있다
FOE_DX = 5               # 가장 가까운 적의 상대 x (×256 하면 픽셀)
FOE_DY = 6
FOE_DIST = 7
FOE_HP = 8
FOE_IN_RANGE = 10        # 1 이면 사거리 안

DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def move_toward(dx, dy):
    """(dx, dy) 방향에 가장 가까운 이동 행동 번호."""
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return STOP
    dx, dy = dx / n, dy / n
    best, best_i = -9.0, 3
    for i, (ax, ay) in enumerate(DIRS):
        an = math.hypot(ax, ay)
        d = (dx * ax + dy * ay) / an
        if d > best:
            best, best_i = d, i + 1
    return best_i


def act(obs, mask):
    out = []
    for i in range(len(obs)):
        if not mask[i]:            # 죽은 유닛
            out.append(STOP)
            continue
        row = obs[i]
        in_range = row[FOE_IN_RANGE] > 0.5
        can_fire = row[MY_COOLDOWN] < 0.1

        if in_range and can_fire:
            out.append(ATTACK_NEAR)
        elif in_range:
            out.append(RETREAT)
        else:
            out.append(move_toward(row[FOE_DX], row[FOE_DY]))
    return out
