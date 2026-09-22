#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
점수 진단기 — 돌리기 전에 사고를 잡습니다.

순수 함수만 있습니다. 네트워크도 파일도 건드리지 않습니다.
입력은 dict, 출력은 [{"level","title","body","fix"}] 리스트입니다.

여기 있는 규칙은 전부 실제로 났던 사고에서 나왔습니다.
  research_star/실패기_마린강화학습.md   사고 ①~⑪
  research_star/결과_마린vs럴커.md       사고 ①~⑥
주석에 붙은 (실패기 ③) 같은 표시가 그 사고 번호입니다.

  python3.11 diagnose.py    ← 회귀 시험. 하나라도 틀리면 여기서 죽습니다.
                               (studio.py 는 뜨기 전에 이 파일을 import 합니다)
"""

import math

# ─────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────

LN12 = math.log(12.0)          # 행동 12종 균등분포의 탐험 정도 = 2.4849

# UI 값 = CLI 정수 ÷ 이 나눗수. bwmicro_sp.cpp 167~177 줄에서 그대로 옮겼습니다.
# ★ r-step 만 1000 으로 나눕니다. 나머지는 전부 100 입니다.
DIVISOR = {
    "r-enemy-hp": 100.0, "r-ally-hp": 100.0, "r-kill": 100.0, "r-death": 100.0,
    "r-win": 100.0, "r-lose": 100.0, "r-timeout": 100.0, "r-misfire": 100.0,
    "r-step": 1000.0,
}

# 계약서 '사용자에게 보이는 손잡이' 표의 기본값
DEFAULT_SLIDERS = {
    "r-enemy-hp": 300, "r-ally-hp": 30, "r-kill": 200, "r-death": 100,
    "r-win": 1000, "r-lose": 1000, "r-timeout": 2500, "r-misfire": 0,
    "r-step": 1, "ent": 0.01, "gamma": 0.999, "frame-skip": 4,
}

# 계약서 '확정된 판': 벌처 1기(왼쪽·P0) vs 저글링 6마리(오른쪽·P1)
DEFAULT_SETUP = {
    "allies": 1, "enemies": 6, "max_steps": 200,
    # 헛방 벌점 누적을 어림잡을 때 쓰는 두 비율입니다. (럴커 ③)
    "p_attack": 0.68,        # 공격 행동(9·10)을 누르는 비율. 럴커 편 실측 68%
    "p_out_of_range": 0.50,  # 그중 사거리 밖이라 헛방이 되는 비율. 학습 초반 반반
}

# '싸우다 전멸' 을 대표할 때 쓰는 적 체력 소모 비율.
# 0.0 = 한 대도 못 때리고 전멸, 1.0 = 적 체력을 다 깎고도 전멸.
LOSE_F = 0.5

# ── 임계값 ───────────────────────────────────────────────────────────────
GAMMA_DISC_WARN = 0.35   # γ^Tmax 가 이 밑이면 끝 벌점이 흐려집니다 (럴커 ①)
GAMMA_DISC_BAD = 0.20    # γ=0.99, Tmax=200 → 0.134 가 여기 걸립니다
ENT_WARN = 1.20          # 탐험 정도 (실패기 ⑤: 0.484 에서 붕괴)
ENT_BAD = 0.80
SCALE_WARN = 40.0        # 점수 스케일 (럴커 ④: −31~+100 이 너무 컸다)
SCALE_BAD = 100.0
RATIO_WARN = 3.0         # 승리 점수와 패배 점수가 3배 밖으로 벌어지면 경보

# 스텝당 팀 평균 이동 픽셀의 '서 있음' 임계 (셀프플레이 신규 위험).
# 근거: 판단 주기 4프레임에서 한 스텝을 온전히 걸으면
#   벌처  flingy.top_speed 1707/256 = 6.67 px/프레임 × 4 = 26.7 px/스텝
#   마린  실측 4.000 px/프레임(실패기 부수 항목) × 4 = 16.0 px/스텝
#   저글링도 마린과 같은 iscript 구동이라 16~22 px/스텝
# 계약서 metrics 예시의 정상값도 move0=31.2 / move1=48.9 입니다.
# 그래서 한 스텝에 프레임당 1.25px(= 걸음의 1/5) 도 못 움직였으면 서 있는 것으로 봅니다.
MOVE_STOP_PER_FRAME = 1.25   # × frame_skip → 판단주기 4 에서 5.0 px/스텝


# ─────────────────────────────────────────────────────────────────────────
# 닫힌 식 — bwmicro_sp.cpp 의 compute_rewards(401~421줄) + term(653줄)에서 유도
# ─────────────────────────────────────────────────────────────────────────
#
#   매 스텝:  적 HP 1기분(정규화 1.0)을 깎으면 +Rd, 적 1기 죽으면 +Rk
#             내 HP 1기분을 잃으면 −Ra, 내 1기 죽으면 −Rx
#             무조건 −Rs, 헛방 1회마다 −Rm
#   종료:     이기면 +Rw / 지면 −Rl / 시간초과면 −Rt   (셋은 서로 배타적입니다)
#
#   무손실 승리    W    = Ne·(Rd + Rk) − Tmax·Rs + Rw
#   싸우다 전멸    L(f) = f·Ne·Rd − Na·(Ra + Rx) − Tmax·Rs − Rl
#   도망(시간초과) F    = −Tmax·Rs − Rt
#
# ★ Tmax·Rs 는 판이 최대 길이까지 갔을 때의 시간 벌점입니다. 승리는 보통 더 일찍
#   끝나므로 W 는 조금 보수적으로(작게) 나옵니다.

def _sliders(sliders):
    """UI 값(실수)으로 바꾼 손잡이 묶음을 돌려줍니다."""
    s = dict(DEFAULT_SLIDERS)
    s.update(sliders or {})
    v = {k: float(s.get(k, DEFAULT_SLIDERS[k])) / d for k, d in DIVISOR.items()}
    v["ent"] = float(s.get("ent", 0.01))
    v["gamma"] = float(s.get("gamma", 0.999))
    v["frame-skip"] = int(s.get("frame-skip", 4))
    return v


def _setup(setup):
    st = dict(DEFAULT_SETUP)
    st.update(setup or {})
    if "max-steps" in st:
        st["max_steps"] = st["max-steps"]
    return st


def _counts(st, side):
    """그 쪽에서 본 (내 유닛 수, 적 유닛 수). 셀프플레이라 오른쪽은 뒤집힙니다."""
    a, e = int(st["allies"]), int(st["enemies"])
    return (a, e) if side == "P0" else (e, a)


def totals(sliders, setup, side="P0", f=LOSE_F):
    """세 가지 결말의 총점을 돌려줍니다. {"win","lose","flee"}

    side="P0" 은 왼쪽(벌처), "P1" 은 오른쪽(저글링) 시점입니다.
    lose 는 적 체력을 f 만큼 깎고 전멸했을 때의 값입니다."""
    v = _sliders(sliders)
    st = _setup(setup)
    na, ne = _counts(st, side)
    tmax = int(st["max_steps"])
    step = tmax * v["r-step"]
    return {
        "win": ne * (v["r-enemy-hp"] + v["r-kill"]) - step + v["r-win"],
        "lose": f * ne * v["r-enemy-hp"] - na * (v["r-ally-hp"] + v["r-death"])
                - step - v["r-lose"],
        "flee": -step - v["r-timeout"],
    }


def _item(level, title, body, fix=None):
    return {"level": level, "title": title, "body": body, "fix": fix}


SIDE_LABEL = {"P0": "왼쪽", "P1": "오른쪽"}


def _who(st, side):
    name = st.get("ally" if side == "P0" else "enemy")
    return f"{SIDE_LABEL[side]}({name})" if name else SIDE_LABEL[side]


# ─────────────────────────────────────────────────────────────────────────
# 돌리기 전 진단
# ─────────────────────────────────────────────────────────────────────────

def diagnose_before(sliders, setup):
    """점수를 돌리기 전에 봅니다. 반환은 [{"level","title","body","fix"}]"""
    v = _sliders(sliders)
    st = _setup(setup)
    tmax = int(st["max_steps"])
    step = tmax * v["r-step"]
    out = []

    # ① 도망이 이득인가 (실패기 ③ — "도망이 싸우다 지는 것보다 8.9점 이득이었다")
    #    셀프플레이라 양쪽에 각각 겁니다. 유닛 수가 다르면 값도 다릅니다.
    #    처방은 한 번 눌러 양쪽이 다 고쳐지도록 두 쪽 중 큰 값을 씁니다.
    need_ui = max(-totals(sliders, setup, s, 0.0)["lose"] - step + 2.0
                  for s in ("P0", "P1"))          # 2점 여유
    need_ui = max(0, int(math.ceil(need_ui)))
    fix = {
        "label": f"'시간만 끌면'을 {need_ui}점으로 올리기",
        "patch": {"r-timeout": min(6000, need_ui * 100)},
    }
    for side in ("P0", "P1"):
        na, ne = _counts(st, side)
        t1 = totals(sliders, setup, side, 1.0)
        t5 = totals(sliders, setup, side, 0.5)
        t0 = totals(sliders, setup, side, 0.0)
        flee = t1["flee"]
        who = _who(st, side)
        if flee > t1["lose"]:
            out.append(_item(
                "danger", f"{who}: 도망이 제일 이득입니다",
                f"적을 전부 깎고 전멸해도 {t1['lose']:+.1f}점인데 그냥 시간만 끌면 "
                f"{flee:+.1f}점입니다. {flee - t1['lose']:+.1f}점 이득이라 얘는 "
                f"싸우지 않는 쪽을 고릅니다. 이대로 돌리면 도망 봇이 나옵니다.", fix))
        elif flee > t5["lose"]:
            out.append(_item(
                "danger", f"{who}: 도망이 싸우는 것보다 낫습니다",
                f"적 체력을 절반 깎고 전멸하면 {t5['lose']:+.1f}점, 시간만 끌면 "
                f"{flee:+.1f}점입니다. 도망이 {flee - t5['lose']:+.1f}점 이득입니다.", fix))
        elif flee > t0["lose"]:
            out.append(_item(
                "warn", f"{who}: 못 싸울 땐 도망이 이득입니다",
                f"한 대도 못 때리고 전멸하면 {t0['lose']:+.1f}점, 시간만 끌면 "
                f"{flee:+.1f}점입니다. 아직 아무것도 못 하는 학습 초반에 도망부터 "
                f"배울 수 있습니다.", fix))
        else:
            out.append(_item(
                "ok", f"{who}: 도망은 손해입니다",
                f"이기면 {t1['win']:+.1f} / 싸우다 전멸 {t5['lose']:+.1f} / "
                f"시간만 끌면 {flee:+.1f}점입니다. 도망이 제일 나쁩니다.", None))

    # ② 감가율이 벌점을 지우는가 (럴커 ① — 0.99^200 = 0.134)
    disc = v["gamma"] ** tmax
    seen = v["r-timeout"] * disc
    fix_g = {"label": "나중 챙기기를 0.999로 올리기", "patch": {"gamma": 0.999}}
    if disc < GAMMA_DISC_BAD:
        out.append(_item(
            "danger", "끝에 주는 벌점이 지워집니다",
            f"나중 챙기기 {v['gamma']}, 한 판 {tmax}스텝이면 {v['gamma']}^{tmax} = "
            f"{disc:.3f} 입니다. 판 시작 시점에서 시간초과 벌점 {-v['r-timeout']:.1f}점이 "
            f"{-seen:.2f}점으로 쪼그라듭니다. 반면 유닛 하나 죽는 "
            f"{-(v['r-ally-hp'] + v['r-death']):.1f}점은 그 자리에서 들어옵니다. "
            f"계산상 도망이 이득이 됩니다.", fix_g))
    elif disc < GAMMA_DISC_WARN:
        out.append(_item(
            "warn", "끝에 주는 벌점이 흐려집니다",
            f"{v['gamma']}^{tmax} = {disc:.3f} 이라 시간초과 벌점이 판 시작 시점엔 "
            f"{-seen:.2f}점으로 보입니다. 판이 길수록 끝 벌점이 안 먹습니다.", fix_g))
    else:
        out.append(_item(
            "ok", "끝 벌점이 판 처음까지 닿습니다",
            f"{v['gamma']}^{tmax} = {disc:.3f}. 시간초과 벌점이 판 시작 시점에서도 "
            f"{-seen:.1f}점으로 보입니다.", None))

    # ③ 헛방 벌점이 상금을 넘는가 (럴커 ③ — 제일 위험합니다)
    #    "헛방 벌점이 판당 20점 누적 → 방아쇠를 아예 안 당김"
    pa = float(st["p_attack"])
    po = float(st["p_out_of_range"])
    fix_m = {"label": "헛방 벌점을 0으로 끄기", "patch": {"r-misfire": 0}}
    for side in ("P0", "P1"):
        na, ne = _counts(st, side)
        cost = na * tmax * pa * po * v["r-misfire"]
        w = totals(sliders, setup, side, 1.0)["win"]
        who = _who(st, side)
        if v["r-misfire"] <= 0:
            continue
        if cost > 0.5 * w:
            out.append(_item(
                "danger", f"{who}: 헛방 벌점이 상금을 잡아먹습니다",
                f"유닛 {na}기 × {tmax}스텝 × 공격 {pa:.0%} × 사거리밖 {po:.0%} × "
                f"{v['r-misfire']:.2f}점 = 판당 {cost:.1f}점입니다. "
                f"이겨서 받는 {w:+.1f}점의 절반을 넘습니다. "
                f"얘는 방아쇠를 아예 안 당기는 쪽을 배웁니다.", fix_m))
        elif cost > 0.2 * w:
            out.append(_item(
                "warn", f"{who}: 헛방 벌점이 셉니다",
                f"판당 {cost:.1f}점이 헛방으로 빠집니다. 이겨서 받는 {w:+.1f}점의 "
                f"{cost / w:.0%} 입니다. 공격을 덜 누르게 될 수 있습니다.", fix_m))

    # ④ 점수 스케일이 얘한테 너무 큰가 (럴커 ④ — −31~+100 이 너무 컸다)
    for side in ("P0", "P1"):
        t1 = totals(sliders, setup, side, 1.0)
        t0 = totals(sliders, setup, side, 0.0)
        w, l = t1["win"], min(t0["lose"], t1["flee"])
        scale = max(abs(w), abs(l))
        who = _who(st, side)
        if scale > SCALE_BAD:
            out.append(_item(
                "danger", f"{who}: 점수 폭이 너무 큽니다",
                f"제일 좋은 결말 {w:+.1f} ~ 제일 나쁜 결말 {l:+.1f}점입니다. "
                f"{SCALE_BAD:.0f}점을 넘으면 얘가 한 판에 크게 흔들려서 "
                f"잘하던 것까지 무너집니다.", None))
        elif scale > SCALE_WARN:
            out.append(_item(
                "warn", f"{who}: 점수 폭이 큽니다",
                f"제일 좋은 결말 {w:+.1f} ~ 제일 나쁜 결말 {l:+.1f}점입니다. "
                f"{SCALE_WARN:.0f}점 안으로 줄이는 편이 안전합니다.", None))
        ratio = abs(w) / max(1e-9, abs(l))
        if ratio > RATIO_WARN or ratio < 1.0 / RATIO_WARN:
            out.append(_item(
                "warn", f"{who}: 상금과 벌점이 한쪽으로 쏠렸습니다",
                f"이기면 {w:+.1f}, 최악이면 {l:+.1f}점이라 {ratio:.1f}배 차이입니다. "
                f"한쪽만 크면 얘가 그쪽만 피하거나 그쪽만 쫓습니다.", None))

    # ⑤ 과제가 애초에 가능한가 (실패기 ④ — 손코딩 최적해도 승률 0% 였다)
    a, e = int(st["allies"]), int(st["enemies"])
    out.append(_item(
        "info", "손코딩 균형을 먼저 재세요",
        f"{a} 대 {e} 판이 애초에 붙을 만한지는 점수만 봐서는 알 수 없습니다. "
        f"'붙여보기'로 손코딩 4종 대전표(16칸)를 먼저 재세요. 한쪽이 전부 이기면 "
        f"얘가 아무리 배워도 그림이 안 나옵니다. 지난번엔 손으로 짠 최적 전략도 "
        f"한 판을 못 이기는 판을 두 달 붙잡고 있었습니다.", None))

    return out


# ─────────────────────────────────────────────────────────────────────────
# 돌리는 중 진단
# ─────────────────────────────────────────────────────────────────────────

def _upds(recent):
    return [r for r in (recent or []) if r.get("t") == "upd"]


def _avg(rows, key, default=0.0):
    xs = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    return sum(xs) / len(xs) if xs else default


def _avg_act0(rows, key):
    xs = [float(r[key][0]) for r in rows
          if isinstance(r.get(key), (list, tuple)) and r[key]]
    return sum(xs) / len(xs) if xs else 0.0


def _last_gen(recent):
    g = None
    for r in (recent or []):
        if r.get("t") == "gen" and isinstance(r.get("gen"), int):
            g = r["gen"]
    return g


def _matches(ladder, foe):
    return [m for m in (ladder or [])
            if m.get("t") == "match" and m.get("foe") == foe]


def diagnose_during(recent, ladder, frame_skip=4):
    """돌아가는 중에 봅니다. recent 는 metrics.jsonl 의 최근 줄들,
    ladder 는 ladder.jsonl 의 줄들입니다."""
    out = []
    ups = _upds(recent)
    if not ups:
        return [_item("info", "아직 볼 게 없습니다",
                      "첫 갱신이 나오면 여기에 진단이 뜹니다.", None)]
    last = ups[-5:]                      # 최근 5번 갱신의 평균으로 봅니다
    ep = ups[-1].get("ep", 0)

    # ⑥ 탐험이 굳었는가 (실패기 ⑤ — 엔트로피 0.484 에서 승률이 무너졌다)
    fix_e = {"label": "탐험 계수를 0.02로 올리기", "patch": {"ent": 0.02}}
    for side, key in (("P0", "ent0"), ("P1", "ent1")):
        ent = _avg(last, key, LN12)
        who = SIDE_LABEL[side]
        if ent < ENT_BAD:
            out.append(_item(
                "danger", f"{who}: 탐험이 멈췄습니다",
                f"탐험 정도가 {ent:.2f} 입니다(균등하면 {LN12:.2f}). 그럭저럭 되는 "
                f"전략 하나에 굳어서 더 나은 걸 시도조차 안 합니다. 여기서 그냥 두면 "
                f"성적이 오르다 무너집니다.", fix_e))
        elif ent < ENT_WARN:
            out.append(_item(
                "warn", f"{who}: 탐험이 줄고 있습니다",
                f"탐험 정도 {ent:.2f}. {ENT_WARN:.1f} 밑으로 내려왔습니다. "
                f"{ENT_BAD:.1f} 아래로 가면 붕괴입니다.", fix_e))

    # ⑦ ★ 둘 다 서서 자동교전하는가 (셀프플레이 신규 위험 — 1순위)
    #    행동 0 비율만 보면 안 됩니다. 헛방으로 정지된 유닛은 행동 0 으로 안 세집니다.
    #    그래서 (행동0 + 헛방) 과 스텝당 이동 픽셀을 같이 봅니다.
    thr = MOVE_STOP_PER_FRAME * max(1, int(frame_skip))
    mv0, mv1 = _avg(last, "move0"), _avg(last, "move1")
    st0 = _avg_act0(last, "acts0") + _avg(last, "mf0")
    st1 = _avg_act0(last, "acts1") + _avg(last, "mf1")
    half = max(2, int(frame_skip) // 2)
    fix_s = {"label": f"판단 주기를 {half}프레임으로 줄이기",
             "patch": {"frame-skip": half}}
    if mv0 < thr and mv1 < thr:
        out.append(_item(
            "danger", "양쪽 다 서서 자동교전만 합니다",
            f"스텝당 이동이 왼쪽 {mv0:.1f}px / 오른쪽 {mv1:.1f}px 입니다(서 있음 기준 "
            f"{thr:.1f}px). 제자리 비율은 왼쪽 {st0:.0%} / 오른쪽 {st1:.0%} 입니다. "
            f"둘 다 안 움직이면 판이 자동 교전으로만 끝나서 배울 게 없습니다. "
            f"판단 주기를 줄이거나 시간 벌점을 올려 붙게 만드세요.", fix_s))
    elif mv0 < thr or mv1 < thr:
        who = "왼쪽" if mv0 < thr else "오른쪽"
        mv = mv0 if mv0 < thr else mv1
        stq = st0 if mv0 < thr else st1
        out.append(_item(
            "warn", f"{who}이 서 있습니다",
            f"스텝당 이동 {mv:.1f}px(기준 {thr:.1f}px), 제자리 비율 {stq:.0%} 입니다. "
            f"반대쪽만 움직이는 중이라 한쪽 전략이 굳었을 수 있습니다.", None))
    elif max(st0, st1) > 0.60:
        who = "왼쪽" if st0 > st1 else "오른쪽"
        out.append(_item(
            "warn", f"{who}: 제자리 비율이 높습니다",
            f"행동 0 과 헛방을 합치면 {max(st0, st1):.0%} 입니다. 움직이긴 하는데 "
            f"명령의 절반 이상이 제자리입니다.", None))

    # ⑧ 배선이 잘못됐는가 (실패기 ⑦ 재발 방지 — 15만 판 내내 점수가 한 칸도 안 움직였다)
    #    0세대(학습 전)를 이기지 못하면 학습 신호가 아예 안 들어오고 있는 것입니다.
    zero = _matches(ladder, "past:0")
    if zero:
        m = zero[-1]
        n = max(1, int(m.get("n", 0)))
        wr = int(m.get("win", 0)) / n
        if int(m.get("ep", ep)) > 2000 and abs(wr - 0.5) <= 0.03:
            out.append(_item(
                "danger", "2천 판을 넘겼는데 학습 전과 똑같습니다",
                f"{m.get('ep', ep):,}판을 돌았는데 0세대 상대 승률이 {wr:.1%} "
                f"({m.get('win')}승 / {n}판)입니다. 50% 근처에서 안 움직이면 배선이 "
                f"끊긴 것입니다. 점수가 실제로 들어오는지, 행동이 게임에 닿는지부터 "
                f"보세요. 지난번엔 총알이 안 나가고 있었습니다.", None))
        elif int(m.get("ep", ep)) > 2000:
            out.append(_item(
                "ok", "학습 전보다 나아졌습니다",
                f"0세대 상대 승률 {wr:.1%} ({m.get('win')}승 / {n}판). "
                f"점수가 제대로 들어오고 있습니다.", None))

    # ⑨ 세대가 돌고 있는가 — 지금 세대가 10세대 전을 확실히 이겨야 합니다
    gen = _last_gen(recent)
    if gen is None:
        gen = max([m.get("gen", 0) for m in (ladder or [])
                   if m.get("t") == "match"] or [0])
    if gen >= 10:
        past = _matches(ladder, f"past:{gen - 10}")
        if past:
            m = past[-1]
            n = max(1, int(m.get("n", 0)))
            wr = int(m.get("win", 0)) / n
            if wr < 0.60:
                out.append(_item(
                    "warn", "세대가 제자리걸음입니다",
                    f"지금 {gen}세대가 {gen - 10}세대를 {wr:.1%} 로 이깁니다 "
                    f"({m.get('win')}승 / {n}판). 10세대를 더 돌았는데 60% 도 못 넘으면 "
                    f"실력이 아니라 가위바위보만 돌고 있는 것입니다.", None))
            else:
                out.append(_item(
                    "ok", "세대가 앞으로 갑니다",
                    f"{gen}세대가 {gen - 10}세대를 {wr:.1%} 로 이깁니다.", None))

    if not any(i["level"] in ("danger", "warn") for i in out):
        out.append(_item("ok", "지금은 이상 없습니다",
                         f"{ep:,}판째. 걸린 규칙이 없습니다.", None))
    return out


# ═════════════════════════════════════════════════════════════════════════
# 회귀 시험 — 실패기 문서의 숫자와 맞는지 확인합니다.
# 여기서 죽으면 studio.py 도 안 뜹니다. 그게 목적입니다.
# ═════════════════════════════════════════════════════════════════════════

# 실패기 ③ '전' 열 — 마린 5기 vs 질럿 3기, ppo_micro.py 의 기본 max_steps=120
MARINE_BEFORE = {"r-enemy-hp": 100, "r-ally-hp": 50, "r-kill": 200, "r-death": 200,
                 "r-win": 1000, "r-lose": 1000, "r-timeout": 1000,
                 "r-misfire": 0, "r-step": 1}
# 실패기 ③ '후' 열
MARINE_AFTER = {"r-enemy-hp": 300, "r-ally-hp": 30, "r-kill": 200, "r-death": 100,
                "r-win": 1000, "r-lose": 1000, "r-timeout": 2500,
                "r-misfire": 0, "r-step": 1}
SETUP_5v3 = {"allies": 5, "enemies": 3, "max_steps": 120}
SETUP_8v3 = {"allies": 8, "enemies": 3, "max_steps": 200}


def _near(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: {a!r} 이 {b!r} 과 {tol} 안에서 안 맞습니다"


def _selftest():
    ok = []

    # ── 실패기 ③ '전' : 승리 +19.0 / 도망 −10.1 ───────────────────────────
    t = totals(MARINE_BEFORE, SETUP_5v3)
    _near(t["win"], 19.0, 0.15, "실패기③ 전 승리")     # 닫힌 식 18.88
    _near(t["flee"], -10.1, 0.03, "실패기③ 전 도망")   # 닫힌 식 −10.12
    ok.append(f"실패기③ 전   승리 {t['win']:+.2f} (문서 +19.0) / "
              f"도망 {t['flee']:+.2f} (문서 −10.1)")

    # 문서의 '싸우다 마린 전멸 −19.0' 은 닫힌 식이 아니라 실측 평균(ep 15,886 의 −18.98)
    # 입니다. 전멸(−22.62 ~ −19.62)과 시간초과(−10.12)가 섞인 값이라 그 사이에 있습니다.
    l0 = totals(MARINE_BEFORE, SETUP_5v3, "P0", 0.0)["lose"]
    l1 = totals(MARINE_BEFORE, SETUP_5v3, "P0", 1.0)["lose"]
    assert l0 < -18.98 < t["flee"], "실패기③ 실측 −18.98 이 전멸~도망 사이가 아닙니다"
    assert l0 < -20.58 < t["flee"], "실패기③ 실측 −20.58 이 전멸~도망 사이가 아닙니다"
    ok.append(f"실패기③ 전멸 구간 {l0:+.2f} ~ {l1:+.2f}, "
              f"실측 평균 −18.98 / −20.58 이 도망({t['flee']:+.2f})보다 낮음")

    # 도망이 이득이라는 진단이 실제로 떠야 합니다
    d = diagnose_before(MARINE_BEFORE, SETUP_5v3)
    assert any(i["level"] == "danger" and "도망" in i["title"] for i in d), \
        "실패기③ 전 설정에서 도망 경고가 안 떴습니다"
    # 그리고 제안하는 시간초과 벌점이 문서가 실제로 고른 25(=2500)와 같아야 합니다
    fx = [i["fix"] for i in d if i["level"] == "danger" and "도망" in i["title"]][0]
    assert fx["patch"]["r-timeout"] == 2500, \
        f"제안값이 문서의 2500 이 아닙니다: {fx['patch']}"
    ok.append(f"실패기③ 처방 → {fx['label']} {fx['patch']} (문서가 고른 값과 일치)")

    # ── 실패기 ③ '후' : 승리 +25 / 도망 −25.1 ────────────────────────────
    t = totals(MARINE_AFTER, SETUP_5v3)
    _near(t["win"], 25.0, 0.15, "실패기③ 후 승리")     # 닫힌 식 24.88
    _near(t["flee"], -25.1, 0.03, "실패기③ 후 도망")   # 닫힌 식 −25.12
    ok.append(f"실패기③ 후   승리 {t['win']:+.2f} (문서 +25) / "
              f"도망 {t['flee']:+.2f} (문서 −25.1)")
    d = diagnose_before(MARINE_AFTER, SETUP_5v3)
    assert not any(i["level"] == "danger" and "도망" in i["title"] for i in d), \
        "실패기③ 후 설정에서 도망 경고가 아직 뜹니다"
    ok.append("실패기③ 후   도망 경고 사라짐 (역전 확인)")

    # ── 실패기 ⑦ / ⑨ : 마린 8기 전멸 예상 −20.4 ──────────────────────────
    #    문서의 분해(아군HP −2.4 + 사망 −8 + 패배 −10)는 시간 벌점을 빼고 적은 값이라
    #    닫힌 식에서 Tmax·Rs 를 되돌려 더해 정확히 −20.4 가 나와야 합니다.
    v = _sliders(MARINE_AFTER)
    step = SETUP_8v3["max_steps"] * v["r-step"]
    l = totals(MARINE_AFTER, SETUP_8v3, "P0", 0.0)["lose"]
    _near(l + step, -20.4, 1e-6, "실패기⑦ 전멸 예상")
    ok.append(f"실패기⑦ 8기 전멸 {l:+.2f} (시간 벌점 {step:.2f} 제외하면 "
              f"{l + step:+.1f} = 문서 −20.4)")
    # 실측 −20.3(⑦) / −19.84(⑨) 는 적 HP 를 조금 깎은 만큼 위에 있어야 합니다
    assert l < -20.3 < totals(MARINE_AFTER, SETUP_8v3, "P0", 1.0)["lose"], \
        "실패기⑦ 실측 −20.3 이 전멸 구간 밖입니다"
    assert l < -19.84 < totals(MARINE_AFTER, SETUP_8v3, "P0", 1.0)["lose"], \
        "실패기⑨ 실측 −19.84 가 전멸 구간 밖입니다"
    ok.append("실패기⑦⑨ 실측 −20.3 / −19.84 가 전멸 구간 안에 들어옴")

    # ── 럴커 ① : 0.99^200 = 0.134 ─────────────────────────────────────────
    _near(0.99 ** 200, 0.134, 0.0005, "럴커① 감가율")
    d = diagnose_before({**MARINE_AFTER, "gamma": 0.99}, SETUP_8v3)
    assert any(i["level"] == "danger" and "벌점이 지워" in i["title"] for i in d), \
        "럴커① γ=0.99 에서 경고가 안 떴습니다"
    ok.append(f"럴커① 0.99^200 = {0.99 ** 200:.3f} (문서 0.134) → danger")
    d = diagnose_before({**MARINE_AFTER, "gamma": 0.999}, SETUP_8v3)
    assert not any(i["level"] in ("danger", "warn") and "벌점" in i["title"] and
                   "헛방" not in i["title"] for i in d), "γ=0.999 에서 경고가 남았습니다"
    ok.append(f"럴커① 0.999^200 = {0.999 ** 200:.3f} → 경고 없음")

    # ── 럴커 ③ : 헛방 벌점 누적 (마린 10기 × 200스텝 × 68% × 0.02) ────────
    setup10 = {"allies": 10, "enemies": 2, "max_steps": 200,
               "p_attack": 0.68, "p_out_of_range": 1.0}
    cost = 10 * 200 * 0.68 * 1.0 * 0.02
    _near(cost, 27.2, 1e-9, "럴커③ 헛방 누적")
    d = diagnose_before({**MARINE_AFTER, "r-misfire": 2}, setup10)
    assert any(i["level"] == "danger" and "헛방" in i["title"] for i in d), \
        "럴커③ 헛방 경고가 안 떴습니다"
    ok.append(f"럴커③ 헛방 누적 {cost:.1f}점/판 (문서 본문 20점, 문서 괄호식 27.2) → danger")
    d = diagnose_before(MARINE_AFTER, setup10)
    assert not any("헛방" in i["title"] for i in d), "헛방 벌점 0 인데 경고가 떴습니다"
    ok.append("럴커③ 헛방 벌점 0 이면 경고 없음")

    # ── 럴커 ④ : 키운 점수 스케일 −31 ~ +100 ─────────────────────────────
    big = {**MARINE_AFTER, "r-enemy-hp": 600, "r-win": 4000, "r-lose": 2000}
    d = diagnose_before(big, SETUP_8v3)
    assert any(i["level"] in ("danger", "warn") and "점수 폭" in i["title"] for i in d), \
        "럴커④ 큰 스케일에서 경고가 안 떴습니다"
    t = totals(big, SETUP_8v3, "P0", 1.0)
    ok.append(f"럴커④ 승리 {t['win']:+.1f} / 전멸 "
              f"{totals(big, SETUP_8v3, 'P0', 0.0)['lose']:+.1f} → 스케일 경고")

    # ── 계약서 확정 판(벌처 1 vs 저글링 6, 기본값) 은 깨끗해야 합니다 ─────
    d = diagnose_before({}, {})
    assert not any(i["level"] == "danger" for i in d), \
        "계약서 기본값에서 danger 가 떴습니다: " + \
        str([i["title"] for i in d if i["level"] == "danger"])
    t0 = totals({}, {}, "P0")
    t1 = totals({}, {}, "P1")
    ok.append(f"계약서 기본값 왼쪽 승 {t0['win']:+.1f}/전멸 {t0['lose']:+.1f}/"
              f"도망 {t0['flee']:+.1f} · 오른쪽 승 {t1['win']:+.1f}/"
              f"전멸 {t1['lose']:+.1f}/도망 {t1['flee']:+.1f} → danger 0건")

    # ── 실패기 ⑤ : 엔트로피 0.484 붕괴 ────────────────────────────────────
    rows = [{"t": "upd", "upd": 9, "ep": 26759, "ent0": 0.484, "ent1": 1.9,
             "move0": 30.0, "move1": 45.0, "acts0": [0.1] + [0.0] * 11,
             "acts1": [0.1] + [0.0] * 11, "mf0": 0.0, "mf1": 0.0}]
    r = diagnose_during(rows, [])
    assert any(i["level"] == "danger" and "탐험" in i["title"] for i in r), \
        "실패기⑤ 엔트로피 0.484 에서 경고가 안 떴습니다"
    ok.append("실패기⑤ 탐험 정도 0.484 → danger (균등 2.4849)")

    # ── 셀프플레이 신규 : 둘 다 서 있음 ───────────────────────────────────
    rows = [{"t": "upd", "ep": 40000, "ent0": 1.9, "ent1": 1.9,
             "move0": 1.2, "move1": 0.8, "acts0": [0.30] + [0.0] * 11,
             "acts1": [0.25] + [0.0] * 11, "mf0": 0.32, "mf1": 0.40}]
    r = diagnose_during(rows, [])
    assert any(i["level"] == "danger" and "서서" in i["title"] for i in r), \
        "둘 다 서 있는데 경고가 안 떴습니다"
    ok.append("셀프플레이 move0=1.2 / move1=0.8 (기준 5.0) → danger")
    rows[0]["move0"], rows[0]["move1"] = 31.2, 48.9   # 계약서 예시의 정상값
    r = diagnose_during(rows, [])
    assert not any(i["level"] == "danger" for i in r), "정상 move 에서 danger 가 떴습니다"
    ok.append("계약서 예시 move0=31.2 / move1=48.9 → danger 없음")

    # ── 실패기 ⑦ 재발 방지 : 2천 판 넘게 돌았는데 0세대와 동률 ────────────
    lad = [{"t": "match", "ep": 150000, "gen": 20, "side": "P0", "me": "v020",
            "foe": "past:0", "n": 120, "win": 61, "lose": 59, "to": 0,
            "elo_me": 1001.0}]
    r = diagnose_during([{"t": "upd", "ep": 150000, "ent0": 1.9, "ent1": 1.9,
                          "move0": 30.0, "move1": 40.0}], lad)
    assert any(i["level"] == "danger" and "학습 전과 똑같" in i["title"] for i in r), \
        "실패기⑦ 배선 경고가 안 떴습니다"
    ok.append("배선 검사 0세대 상대 50.8% (2천 판 초과) → danger")

    # 세대 진도 검사
    lad = [{"t": "match", "ep": 90000, "gen": 20, "side": "P0", "me": "v020",
            "foe": "past:0", "n": 120, "win": 96, "lose": 24, "to": 0},
           {"t": "match", "ep": 90000, "gen": 20, "side": "P0", "me": "v020",
            "foe": "past:10", "n": 60, "win": 31, "lose": 29, "to": 0}]
    r = diagnose_during([{"t": "upd", "ep": 90000, "ent0": 1.9, "ent1": 1.9,
                          "move0": 30.0, "move1": 40.0},
                         {"t": "gen", "gen": 20, "ep": 90000}], lad)
    assert any(i["level"] == "warn" and "제자리걸음" in i["title"] for i in r), \
        "세대 제자리걸음 경고가 안 떴습니다"
    ok.append("세대 검사 20세대 vs 10세대 51.7% → warn")

    # ── 출력 형식 ─────────────────────────────────────────────────────────
    for items in (diagnose_before({}, {}), diagnose_during(rows, lad)):
        for i in items:
            assert set(i) == {"level", "title", "body", "fix"}, f"키가 다릅니다: {i}"
            assert i["level"] in ("danger", "warn", "info", "ok"), i["level"]
            assert isinstance(i["title"], str) and isinstance(i["body"], str)
            assert i["fix"] is None or set(i["fix"]) == {"label", "patch"}
    ok.append("출력 형식 {level,title,body,fix} 전부 확인")

    return ok


def _show(title, items):
    print(f"\n── {title} " + "─" * max(0, 62 - len(title)))
    for i in items:
        if i["level"] == "ok":
            continue
        print(f"  [{i['level']:6}] {i['title']}")
        print(f"           {i['body']}")
        if i["fix"]:
            print(f"           ↳ 처방: {i['fix']['label']}  {i['fix']['patch']}")


if __name__ == "__main__":
    print("═" * 70)
    print(" 회귀 시험 — 실패기 문서의 숫자와 맞추기")
    print("═" * 70)
    for line in _selftest():
        print("  OK  " + line)
    print("\n  전부 통과했습니다.")

    print("\n" + "═" * 70)
    print(" 일부러 나쁜 설정 3개")
    print("═" * 70)

    _show("나쁜 설정 1 — 시간만 끌면 벌점을 0 으로 내렸습니다",
          diagnose_before({"r-timeout": 0}, {"allies": 1, "enemies": 6,
                                             "max_steps": 200}))
    _show("나쁜 설정 2 — 나중 챙기기 0.99 + 헛방 벌점 최대 + 판을 400스텝으로",
          diagnose_before({"gamma": 0.99, "r-misfire": 20},
                          {"allies": 1, "enemies": 6, "max_steps": 400}))
    _show("나쁜 설정 3 — 이기면 4000, 지면 0 으로 극단으로 벌렸습니다",
          diagnose_before({"r-win": 4000, "r-lose": 0, "r-death": 0,
                           "r-ally-hp": 0, "r-enemy-hp": 600},
                          {"allies": 1, "enemies": 6, "max_steps": 200}))

    print("\n" + "═" * 70)
    print(" 돌리는 중 — 최악의 metrics 한 줄")
    print("═" * 70)
    _show("탐험 붕괴 + 양쪽 다 서 있음 + 0세대와 동률",
          diagnose_during(
              [{"t": "gen", "gen": 30, "ep": 210000},
               {"t": "upd", "upd": 700, "ep": 210000, "ent0": 0.62, "ent1": 0.71,
                "move0": 0.9, "move1": 1.4, "acts0": [0.41] + [0.0] * 11,
                "acts1": [0.38] + [0.0] * 11, "mf0": 0.22, "mf1": 0.31}],
              [{"t": "match", "ep": 210000, "gen": 30, "side": "P0", "me": "v030",
                "foe": "past:0", "n": 120, "win": 59, "lose": 61, "to": 0},
               {"t": "match", "ep": 210000, "gen": 30, "side": "P0", "me": "v030",
                "foe": "past:20", "n": 60, "win": 30, "lose": 30, "to": 0}]))
    print()
