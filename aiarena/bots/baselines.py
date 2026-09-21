"""기준선 봇 4종.

리그 순위표의 좌표축이다. 참가자 봇이 잘하는지는 이 넷과 붙여서 정한다.
넷 다 관측만 보고 판단한다 — 엔진 내부를 들여다보지 않는다. 참가자와 같은 조건이다.

원본 실험의 손코딩 정책을 그대로 옮겼다. 숫자 하나도 바꾸지 않았다.
"""
from typing import List, Sequence
import numpy as np

from ..core.spec import (
    dir_action, SELF_DIM,
    ACT_ATTACK_NEAR, ACT_RETREAT,
)

# 첫 번째(가장 가까운) 적 정보의 위치
_FOE_DX = SELF_DIM + 0      # 5
_FOE_DY = SELF_DIM + 1      # 6
_FOE_IN_RANGE = SELF_DIM + 5  # 10
_MY_CD = 1


def _read_foe(row: np.ndarray):
    """가장 가까운 적의 상대 위치(픽셀)와 사거리 안 여부."""
    return row[_FOE_DX] * 256.0, row[_FOE_DY] * 256.0, row[_FOE_IN_RANGE] > 0.5


def stop(obs: np.ndarray, mask: Sequence[int]) -> List[int]:
    """가만히 선다. 사거리 안에 적이 들어오면 알아서 쏜다.

    아무것도 안 하는 것 같지만 미러전에서는 이게 제일 강할 때가 있다 —
    모두가 서 있으면 아무도 이동 손해를 안 보기 때문이다.
    """
    return [0] * len(obs)


def rush(obs: np.ndarray, mask: Sequence[int]) -> List[int]:
    """사거리에 들어오면 쏘고, 아니면 가장 가까운 적에게 직진한다."""
    out = []
    for i in range(len(obs)):
        if not mask[i]:
            out.append(0)
            continue
        ex, ey, in_range = _read_foe(obs[i])
        out.append(ACT_ATTACK_NEAR if in_range else dir_action(ex, ey))
    return out


def kite(obs: np.ndarray, mask: Sequence[int]) -> List[int]:
    """무빙샷. 쏠 수 있으면 쏘고, 쿨다운 도는 동안은 물러난다.

    사거리가 긴 쪽이 짧은 쪽을 상대로 쓰는 기본기다.
    사거리가 같거나 짧으면 그냥 도망만 다니다 시간이 끝난다.
    """
    out = []
    for i in range(len(obs)):
        if not mask[i]:
            out.append(0)
            continue
        ex, ey, in_range = _read_foe(obs[i])
        cd = obs[i][_MY_CD]
        if in_range and cd < 0.1:
            out.append(ACT_ATTACK_NEAR)
        elif in_range:
            out.append(ACT_RETREAT)
        else:
            out.append(dir_action(ex, ey))
    return out


def spread(obs: np.ndarray, mask: Sequence[int]) -> List[int]:
    """돌진하되 홀짝으로 위아래를 벌려서 접근한다.

    한 줄로 달려가면 범위 공격에 한꺼번에 맞고, 뭉치면 뒤쪽 유닛이 앞을 못 지나간다.
    """
    out = []
    for i in range(len(obs)):
        if not mask[i]:
            out.append(0)
            continue
        ex, ey, in_range = _read_foe(obs[i])
        if in_range:
            out.append(ACT_ATTACK_NEAR)
        else:
            out.append(dir_action(ex, ey + (60 if i % 2 else -60)))
    return out


BASELINES = {
    "stop": stop,
    "rush": rush,
    "kite": kite,
    "spread": spread,
}

# 한국어 이름 — 화면과 문서에 쓴다
BASELINE_KR = {
    "stop": "정지",
    "rush": "돌진",
    "kite": "무빙샷",
    "spread": "벌리기",
}
