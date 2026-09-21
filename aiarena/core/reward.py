"""점수 주는 방식.

**여기가 이 판에서 유일하게 창작인 자리다.**

강화학습은 "어떻게 싸워라"를 가르치는 게 아니다. 점수만 주고, 점수가 높은 행동이
살아남게 둘 뿐이다. 그래서 점수를 어떻게 주느냐가 AI 의 성격을 통째로 정한다.

- 적 체력만 크게 쳐주면 → 무작정 달려들어 같이 죽는다
- 내 체력을 크게 쳐주면 → 도망만 다니다 시간이 끝난다
- 시간 끄는 벌점이 작으면 → 도망이 이득이 되어버린다

아래 숫자를 바꿔가며 돌려보는 게 이 판의 본론이다.
"""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """점수표. 숫자를 바꾸면 AI 성격이 바뀐다."""

    enemy_hp: float = 3.0      # 적 체력을 깎았을 때 (전체 대비 비율에 곱한다)
    ally_hp: float = 0.3       # 내 체력이 깎였을 때 빼는 점수
    kill: float = 2.0          # 적 하나를 잡을 때마다
    death: float = 1.0         # 내 유닛 하나가 죽을 때마다
    win: float = 10.0          # 전멸시켰을 때
    lose: float = 10.0         # 전멸당했을 때 빼는 점수
    timeout: float = 25.0      # ★ 시간만 끌고 끝났을 때 빼는 점수.
                               #   이게 작으면 "안 싸우고 버티기"가 최적해가 된다
    step: float = 0.001        # 판단 한 번마다 빼는 아주 작은 점수 (빨리 끝내라는 압력)

    def as_dict(self):
        return {k: getattr(self, k) for k in
                ("enemy_hp", "ally_hp", "kill", "death", "win", "lose", "timeout", "step")}


# 미리 만들어둔 점수표 몇 가지 — 성격이 어떻게 달라지는지 비교용
PRESETS = {
    "기본": RewardConfig(),
    "겁쟁이": RewardConfig(ally_hp=3.0, death=5.0, timeout=2.0),      # 내 몸 사리기
    "돌격대": RewardConfig(enemy_hp=6.0, ally_hp=0.0, death=0.0),     # 내 피해를 안 센다
    "속전속결": RewardConfig(step=0.02, timeout=40.0),                 # 빨리 끝내라
}
