"""유닛과 판(시나리오)의 스펙.

숫자는 전부 원본 게임 데이터에서 실측한 값이다. 추측한 값에는 주석으로 표시했다.
새 판을 추가할 때 코드를 고치는 게 아니라 여기 데이터만 늘린다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math


@dataclass(frozen=True)
class UnitSpec:
    """유닛 하나의 전투 상수.

    거리 단위는 픽셀, 시간 단위는 프레임(1프레임 = 42ms)이다.
    """
    name: str
    hp: float
    damage: float
    cooldown: int        # 공격 간 프레임 수
    range: int           # 사거리 (박스 간 거리 기준)
    speed: float         # px/frame
    size: int            # 충돌 박스 한 변 (정사각 근사)
    sight: int = 224     # 눈에 보이는 거리
    acquire: int = 128   # ★ 명령 없이 스스로 달려드는 거리. 시야보다 짧다.
                         # 이게 시야와 같으면 '정지'가 '돌진'보다 강해져서 판이 뒤집힌다
    accel: float = 0.0   # px/frame². 0이면 즉시 최고속 (걸어다니는 유닛)
                         # 미끄러지는 유닛은 방향을 되돌리는 데 시간이 걸린다 — 무빙샷의 대가다
    turn: float = 360.0  # 도/frame. 방향을 바꾸는 속도. 미끄러지는 유닛만 의미가 있다

    @property
    def half(self) -> float:
        return self.size / 2.0


# ── 유닛 도감 ────────────────────────────────────────────────────────────
# Raider: 사거리가 길고 빠른 단독 유닛. 쿨다운이 길다.
# Biter : 사거리가 없다시피 하고 느리지만, 쿨다운이 짧고 떼로 나온다.
UNITS: Dict[str, UnitSpec] = {
    "raider": UnitSpec(
        name="raider",
        hp=80.0, damage=20.0, cooldown=30, range=160,
        speed=6.668,   # 실측
        size=32,       # 실측
        sight=256,     # 실측 (8타일)
        acquire=160,   # 실측 (target_acq 5타일)
        accel=0.3906,  # 실측. 정지→최고속 17프레임
        turn=56.25,    # 실측 (turn_rate 40 = 40/256 바퀴). 180도 반전에 3프레임
    ),
    "biter": UnitSpec(
        name="biter",
        hp=35.0, damage=5.0, cooldown=8, range=15,
        speed=5.49,    # ★ 추정치. 원본 엔진은 보행 유닛 속도를 애니메이션으로 굴려서
                       #   데이터에서 직접 못 읽는다. 대전표로 보정하는 값이다.
        size=16,       # 실측
        sight=160,     # 실측 (5타일)
        acquire=128,   # 원본 표에 맞춰 조정 (실측 target_acq 는 3타일=96px)
        accel=0.0,     # 걸어다니는 유닛이라 가속이 사실상 없다
        turn=37.97,    # 실측 (turn_rate 27). accel 이 0 이라 실질 영향은 없다
    ),
}


@dataclass(frozen=True)
class ScenarioSpec:
    """한 판의 정의. 양쪽 유닛과 수, 스폰 기하, 종료 조건."""
    key: str
    p0_unit: str
    p0_count: int
    p1_unit: str
    p1_count: int
    gap_lo: int = 220        # 양 진영 x 거리 (판마다 고정, 매 판 이 범위에서 뽑는다)
    gap_hi: int = 300
    spread_lo: int = 34      # 같은 편 유닛 사이 y 간격
    spread_hi: int = 48
    max_steps: int = 200     # 판단 횟수 (프레임 아님)
    frame_skip: int = 4      # 판단 한 번당 흘려보내는 프레임
    cx: int = 1024
    cy: int = 1024
    map_w: int = 2048        # 맵 경계. 도망치는 쪽에게 벽이 없으면 영원히 안 잡힌다
    map_h: int = 2048

    @property
    def spec_p0(self) -> UnitSpec:
        return UNITS[self.p0_unit]

    @property
    def spec_p1(self) -> UnitSpec:
        return UNITS[self.p1_unit]


# ── 판 도감 ──────────────────────────────────────────────────────────────
SCENARIOS: Dict[str, ScenarioSpec] = {
    "raider_vs_biter": ScenarioSpec(
        key="raider_vs_biter",
        p0_unit="raider", p0_count=1,
        p1_unit="biter",  p1_count=6,
    ),
}


# ── 관측 레이아웃 ────────────────────────────────────────────────────────
# 유닛 하나가 보는 것:
#   [0] 내 체력 비율
#   [1] 내 쿨다운 비율 (0 = 지금 쏠 수 있음)
#   [2] 살아있음 (항상 1. 죽은 유닛은 벡터 전체가 0)
#   [3] 내 x (맵 중앙 기준, /512)
#   [4] 내 y (맵 중앙 기준, /512)
#   그다음 가까운 적 3기, 각 6칸: 상대x/256, 상대y/256, 거리/256, 적 체력비, 있음, 사거리안
#   그다음 아군 4기, 각 4칸: 상대x/256, 상대y/256, 체력비, 있음
N_FOE_OBS = 3
N_ALLY_OBS = 4
SELF_DIM = 5
FOE_DIM = 6
ALLY_DIM = 4
OBS_DIM = SELF_DIM + N_FOE_OBS * FOE_DIM + N_ALLY_OBS * ALLY_DIM  # = 39

# x 를 뒤집을 때 부호를 바꿔야 하는 인덱스 (좌우 대칭 정규화용)
FLIP_IDX: List[int] = (
    [3]
    + [SELF_DIM + k * FOE_DIM for k in range(N_FOE_OBS)]
    + [SELF_DIM + N_FOE_OBS * FOE_DIM + c * ALLY_DIM for c in range(N_ALLY_OBS)]
)

# ── 행동 ────────────────────────────────────────────────────────────────
# 0        정지 — 제자리에 서서, 사거리 안에 적이 있으면 자동으로 쏜다
# 1~8      8방향으로 이동
# 9        가장 가까운 적을 쏜다
# 10       체력이 가장 적은 적을 쏜다
# 11       가장 가까운 적의 반대 방향으로 물러난다
DIRS: List[Tuple[int, int]] = [
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)
]
ACT_NAMES = [
    "정지", "이동1", "이동2", "이동3", "이동4", "이동5", "이동6", "이동7", "이동8",
    "최근접공격", "최저HP공격", "후퇴",
]
N_ACTIONS = len(ACT_NAMES)
MOVE_PX = 96          # 이동 명령 한 번의 목표 거리
ACT_MOVE_LO, ACT_MOVE_HI = 1, 8
ACT_ATTACK_NEAR = 9
ACT_ATTACK_WEAK = 10
ACT_RETREAT = 11


def remap_action(a: int) -> int:
    """좌우를 뒤집었을 때 이동 행동 번호를 되돌린다.
    위(1)·아래(5)와 이동이 아닌 행동은 그대로다."""
    return a if a in (0, 1, 5, 9, 10, 11) else 10 - a


def dir_action(dx: float, dy: float) -> int:
    """(dx, dy) 방향에 가장 가까운 이동 행동 번호를 고른다."""
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 0
    dx, dy = dx / n, dy / n
    best, best_i = -9.0, 3
    for i, (ax, ay) in enumerate(DIRS):
        an = math.hypot(ax, ay)
        d = (dx * ax + dy * ay) / an
        if d > best:
            best, best_i = d, i + 1
    return best_i
