"""채점기 — 학습한 AI 파일 하나를 손코딩 기준선 4종과 붙여서 성적표를 냅니다.

백준의 채점과 같은 자리입니다. 앱의 [채점] 버튼도, 나중에 서버 채점도 이 파일 하나를 씁니다.

    python train/grade.py runs/sp1/ckpt/v012.pt                  # 벌처 파일 채점 (1001번)
    python train/grade.py runs/sp1/ckpt/z012.pt --side zergling  # 저글링 파일 채점 (1002번)
    python train/grade.py runs/sp1/ckpt/v012.pt --export 제출.npz # 제출용 파일 만들기

채점 규칙
  - 행동은 확률 뽑기가 아니라 가장 높은 것(argmax) — 같은 파일이면 항상 같은 점수가 나옵니다.
  - 상대: 손코딩 4종(정지·돌진·무빙샷·벌리기)과 각각 같은 수의 판.
  - 사람 기준: 같은 편 손코딩 4종 중 가장 잘하는 것(벌처면 보통 무빙샷)의 총 승수.
      맞았습니다  = 사람 기준 이상 이김        → AI 가 사람이 짠 정답을 넘었다
      부분 점수   = 사람 기준의 몇 % 를 이겼나
      틀렸습니다  = 가장 못하는 손코딩보다도 못 이김
  - 공개 채점(앱)은 씨앗이 공개돼 있습니다. 서버 채점은 숨긴 씨앗(SC_GRADE_SEEDS)으로 따로 잽니다.

제출 파일(.npz)에는 신경망 숫자만 들어갑니다. .pt 는 파이썬 pickle 이라 열 때 코드가
실행될 수 있어서, 남의 파일은 .npz 로만 받습니다.
"""
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scripted_check as S                     # noqa: E402  (엔진 붙이기·손코딩 4종)

SHAPES = {
    "body.0.weight": (128, 39), "body.0.bias": (128,),
    "body.2.weight": (128, 128), "body.2.bias": (128,),
    "pi.weight": (12, 128), "pi.bias": (12,),
}
PROBLEMS = {"vulture": ("1001", "벌처 무빙샷"), "zergling": ("1002", "저글링 포위")}
UNITS = ("vulture", 1, "zergling", 6)
FRAME_SKIP = 4

# 공개 씨앗: 상대마다 2개 × 10판 = 20판, 모두 80판. 서버는 SC_GRADE_SEEDS 로 숨긴 씨앗을 쓴다.
PUBLIC_SEEDS = (101, 202)
PUBLIC_EPS = 10


def _seeds():
    s = os.environ.get("SC_GRADE_SEEDS")
    seeds = tuple(int(x) for x in s.split(",")) if s else PUBLIC_SEEDS
    eps = int(os.environ.get("SC_GRADE_EPS", PUBLIC_EPS))
    return seeds, eps, bool(s)


# ── 파일 읽기 ───────────────────────────────────────────────────────
class 형식오류(ValueError):
    pass


def load_weights(path):
    """.npz(제출용) 또는 .pt(앱이 만든 체크포인트) → {이름: float32 배열}."""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            w = {k: np.asarray(z[k], dtype=np.float32) for k in SHAPES if k in z.files}
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    else:
        import torch                             # .pt 는 내 컴퓨터의 내 파일만. 그래도 가중치만 연다.
        d = torch.load(path, map_location="cpu", weights_only=True)
        sd = d.get("model", d) if isinstance(d, dict) else d
        w = {k: sd[k].detach().float().numpy() for k in SHAPES if k in sd}
        meta = {k: d.get(k) for k in ("gen", "ep", "side") if isinstance(d, dict) and k in d}
    for k, shp in SHAPES.items():
        if k not in w:
            raise 형식오류(f"신경망 칸이 빠졌습니다: {k}")
        if tuple(w[k].shape) != shp:
            raise 형식오류(f"{k} 모양이 {tuple(w[k].shape)} 입니다. {shp} 이어야 합니다 (입력 39 → 128 → 128 → 행동 12)")
        if not np.all(np.isfinite(w[k])):
            raise 형식오류(f"{k} 에 숫자가 아닌 값(NaN/무한대)이 있습니다")
    return w, meta


def policy_of(w):
    """가중치 → 정책 함수. torch 없이 numpy 로 계산한다 (서버에 torch 가 없어도 된다)."""
    W0, b0, W2, b2, Wp, bp = (w[k] for k in SHAPES)

    def pol(o, alive):
        h = np.tanh(o @ W0.T + b0)
        h = np.tanh(h @ W2.T + b2)
        return (h @ Wp.T + bp).argmax(-1).tolist()
    return pol


def board_hash():
    """판 버전. 엔진·맵·자리·유닛 중 하나라도 바뀌면 달라진다 → 점수는 이 판 안에서만 비교한다.

    엔진은 '실행 파일'이 아니라 '소스'로 센다. 맥에서 빌드한 것과 리눅스에서 빌드한 것은
    파일이 달라도 판정이 똑같아서(실측 확인), 같은 판으로 봐야 케이스를 옮겨 쓸 수 있다.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "engine", "bwmicro_sp.cpp")
    h = hashlib.sha256()
    for p in ((src if os.path.exists(src) else S.BIN), S.MAP):
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    h.update(json.dumps([S.SPOT, UNITS, FRAME_SKIP]).encode())
    return h.hexdigest()[:12]


# ── 채점 ───────────────────────────────────────────────────────────
def _play(side, pol, foe_pol, seeds, eps):
    """side 쪽에 pol 을 세우고 상대 foe_pol 과 붙인다 → (이김, 짐, 비김)."""
    w = l = t = 0
    for sd in seeds:
        if side == "vulture":
            a, b, c, _ = S.duel(*UNITS, pol, foe_pol, eps, sd, FRAME_SKIP, 1)
        else:
            b, a, c, _ = S.duel(*UNITS, foe_pol, pol, eps, sd, FRAME_SKIP, 1)
        w += a; l += b; t += c
    return w, l, t


def _cache_path():
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "aiarena", "grade_ref.json")


def human_reference(side, seeds, eps, pool):
    """손코딩 4종을 같은 편에 세웠을 때의 성적. 판마다 한 번만 재서 저장해 둔다."""
    key = f"{board_hash()}|{side}|{','.join(map(str, seeds))}|{eps}"
    cp = _cache_path()
    try:
        cache = json.load(open(cp))
    except Exception:
        cache = {}
    if key in cache:
        return cache[key]
    names = list(S.POL)
    jobs = {(me, foe): pool.submit(_play, side, S.POL[me], S.POL[foe], seeds, eps)
            for me in names for foe in names}
    ref = {me: {foe: jobs[(me, foe)].result()[0] for foe in names} for me in names}
    cache[key] = ref
    try:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        json.dump(cache, open(cp, "w"), ensure_ascii=False)
    except Exception:
        pass
    return ref


def grade(path, side="vulture", progress=None):
    """성적표(dict)를 돌려준다. progress(끝난 수, 전체) 콜백이 있으면 진행률을 알린다."""
    t0 = time.time()
    if side not in PROBLEMS:
        raise 형식오류(f"편은 vulture 또는 zergling 입니다: {side}")
    w, meta = load_weights(path)
    pol = policy_of(w)
    seeds, eps, hidden = _seeds()
    names = list(S.POL)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        ref = human_reference(side, seeds, eps, pool)
        futs = {foe: pool.submit(_play, side, pol, S.POL[foe], seeds, eps) for foe in names}
        res = {}
        for i, foe in enumerate(names):
            res[foe] = futs[foe].result()
            if progress:
                progress(i + 1, len(names))

    per = len(seeds) * eps
    rows = [{"상대": foe, "이김": res[foe][0], "짐": res[foe][1], "비김": res[foe][2],
             "판": per} for foe in names]
    totals = {me: sum(ref[me].values()) for me in names}
    human = max(totals, key=totals.get)
    for r in rows:
        r["사람"] = ref[human][r["상대"]]
    mine = sum(r["이김"] for r in rows)
    floor = min(totals.values())
    if mine >= totals[human]:
        verdict, score = "맞았습니다", 100
    elif mine <= floor:
        verdict, score = "틀렸습니다", 0
    else:
        verdict = "부분 점수"
        score = max(1, min(99, round(100 * mine / max(1, totals[human]))))
    pid, pname = PROBLEMS[side]
    return {
        "문제": pid, "문제이름": pname, "편": side, "결과": verdict, "점수": score,
        "이김": mine, "판": per * len(names), "사람": human, "사람_이김": totals[human],
        "바닥": floor, "표": rows, "손코딩표": totals,
        "판버전": board_hash(), "씨앗": "숨김" if hidden else "공개",
        "초": round(time.time() - t0, 1), "파일": meta,
    }


# ── 숨긴 테스트케이스 채점 (서버용) ─────────────────────────────────
# 케이스 파일(SC_GRADE_CASES)과 숨긴 상대 전략(SC_GRADE_POLICIES)은 공개 저장소에 없다.
# 케이스 하나 = 판을 차리는 방식(저글링 대형·거리·투혼의 자리) + 상대 전략 + 숨긴 씨앗.
# 게임 규칙(유닛 체력·속도·공격력)은 절대 안 바꾼다. 바꾸는 건 '어디서 어떻게 시작하나' 뿐이다.
# 통과 = 그 케이스에서 사람 손코딩 4종 중 가장 잘한 것만큼 이상 이김.
def _load_hidden(cases_path=None, policies_path=None):
    import importlib.util
    cases_path = cases_path or os.environ.get("SC_GRADE_CASES")
    policies_path = policies_path or os.environ.get("SC_GRADE_POLICIES")
    if not (cases_path and os.path.exists(cases_path)):
        raise 형식오류("숨긴 케이스 파일이 없습니다 (SC_GRADE_CASES). 숨긴 채점은 서버에서만 합니다.")
    cases = json.load(open(cases_path))
    if cases.get("판버전") != board_hash():
        raise 형식오류(f"케이스가 다른 판 버전({cases.get('판버전')})용입니다. 지금 판은 {board_hash()} — 케이스를 다시 재야 합니다.")
    pols = dict(S.POL)
    if policies_path:
        spec = importlib.util.spec_from_file_location("hidden_policies", policies_path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        pols.update(m.HIDDEN)
    return cases, pols


def grade_hidden(path, side="vulture", reveal=False, progress=None, cases_path=None, policies_path=None):
    t0 = time.time()
    w, meta = load_weights(path)
    pol = policy_of(w)
    cases, pols = _load_hidden(cases_path, policies_path)
    todo = cases["문제"][side]

    def one(c):
        extra = ["--zform", str(c["form"]), "--gap-lo", str(c["gap"][0]), "--gap-hi", str(c["gap"][1])]
        win = lose = tie = 0
        for sd in c["seeds"]:
            if side == "vulture":
                a, b, d, _ = S.duel(*UNITS, pol, pols[c["foe"]], PUBLIC_EPS, sd, FRAME_SKIP, 1, extra, spot=c["spot"])
            else:
                b, a, d, _ = S.duel(*UNITS, pols[c["foe"]], pol, PUBLIC_EPS, sd, FRAME_SKIP, 1, extra, spot=c["spot"])
            win += a; lose += b; tie += d
        return win, lose, tie

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        futs = [(c, pool.submit(one, c)) for c in todo]
        for c, f in futs:
            win, lose, tie = f.result()
            r = {"케이스": c["id"], "통과": win >= c["사람"], "이김": win, "판": win + lose + tie}
            if reveal:
                r.update(이름=c["이름"], 사람=c["사람"], 손코딩=c["손코딩"])
            rows.append(r); done += 1
            if progress:
                progress(done, len(todo))
    k = sum(r["통과"] for r in rows)
    verdict = "맞았습니다" if k == len(rows) else ("틀렸습니다" if k == 0 else "부분 점수")
    pid, pname = PROBLEMS[side]
    return {"문제": pid, "문제이름": pname, "편": side, "결과": verdict,
            "점수": round(100 * k / len(rows)), "통과": k, "케이스수": len(rows), "표": rows,
            "판버전": board_hash(), "씨앗": "숨김", "초": round(time.time() - t0, 1), "파일": meta}


def 숨긴성적표_글(r):
    mark = {"맞았습니다": "🟢", "부분 점수": "🟡", "틀렸습니다": "🔴"}[r["결과"]]
    lines = [f"{r['문제']}번 {r['문제이름']} (숨긴 케이스) — {mark} {r['결과']} ({r['점수']}점)",
             f"  케이스 {r['케이스수']}개 중 {r['통과']}개 통과"]
    for x in r["표"]:
        s = f"  {x['케이스']} {'✓' if x['통과'] else '✗'}  {x['이김']:>2}/{x['판']}"
        if "이름" in x:
            s += f"   사람 {x['사람']:>2}   {x['이름']}"
        lines.append(s)
    lines.append(f"  판 버전 {r['판버전']} · {r['초']}초")
    return "\n".join(lines)


def export_npz(path, out, side, extra=None):
    """제출용 파일: 신경망 숫자 + 설명(meta)만. pickle 이 없어서 받는 쪽이 안전하다."""
    w, meta = load_weights(path)
    info = {"문제": PROBLEMS[side][0], "편": side, "판버전": board_hash(),
            "만든때": time.strftime("%Y-%m-%d %H:%M"), **meta, **(extra or {})}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, meta=np.array(json.dumps(info, ensure_ascii=False)), **w)
    load_weights(out)                              # 다시 열어서 확인
    return out


def 성적표_글(r):
    mark = {"맞았습니다": "🟢", "부분 점수": "🟡", "틀렸습니다": "🔴"}[r["결과"]]
    lines = [f"{r['문제']}번 {r['문제이름']} — {mark} {r['결과']} ({r['점수']}점)",
             f"  {r['판']}판 중 {r['이김']}번 이김 · 사람 기준({r['사람']}) {r['사람_이김']}번",
             f"  {'상대':<8}{'이김':>5}{'짐':>5}{'비김':>5}{'사람':>6}"]
    for x in r["표"]:
        lines.append(f"  {x['상대']:<8}{x['이김']:>5}{x['짐']:>5}{x['비김']:>5}{x['사람']:>6}")
    lines.append(f"  판 버전 {r['판버전']} · 씨앗 {r['씨앗']} · {r['초']}초")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AI 파일 채점기")
    ap.add_argument("file", help=".pt 또는 .npz")
    ap.add_argument("--side", default=None, choices=["vulture", "zergling"],
                    help="안 주면 파일 이름으로 짐작 (v→벌처, z→저글링)")
    ap.add_argument("--export", default=None, help="채점 대신 제출용 .npz 를 만든다")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hidden", action="store_true", help="숨긴 케이스로 채점 (서버 전용: SC_GRADE_CASES 필요)")
    ap.add_argument("--reveal", action="store_true", help="숨긴 케이스의 조건까지 보여준다 (운영자용)")
    a = ap.parse_args()
    side = a.side or ("zergling" if os.path.basename(a.file).startswith("z") else "vulture")
    S.check_files()
    try:
        if a.export:
            print("제출용 파일:", export_npz(a.file, a.export, side))
        elif a.hidden:
            r = grade_hidden(a.file, side, reveal=a.reveal)
            print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else 숨긴성적표_글(r))
        else:
            r = grade(a.file, side)
            print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else 성적표_글(r))
    except 형식오류 as e:
        sys.exit(f"형식 오류: {e}")
