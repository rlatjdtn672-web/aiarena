"""설치 확인 겸 손코딩 대전표.

학습을 돌리기 전에 이걸 먼저 실행하세요. 게임 파일과 빌드가 제대로 됐는지 확인되고,
사람이 손으로 짠 네 가지 전략이 서로 어떻게 이기고 지는지 표가 나옵니다.
이 표가 학습한 AI 를 채점할 기준선입니다.

    python train/scripted_check.py

벌처 무빙샷 vs 저글링 돌진이 12:11 쯤으로 팽팽하게 나오면 정상입니다.
"""
import math, os, struct, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPQ = os.environ.get("SC_MPQ", os.path.join(ROOT, "data", "mpq"))
MAP = os.environ.get("SC_MAP", os.path.join(ROOT, "maps", "Weave_v1.scx"))
BIN = os.environ.get("SC_BIN", os.path.join(ROOT, "engine", "bwmicro_sp"))
OBS = 39
DIRS = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

# 미러 정규화: 뒤집을 관측 인덱스 (자기 절대x, 적3기의 상대x, 아군4기의 상대x)
FLIP_IDX = [3, 5, 11, 17, 23, 27, 31, 35]
def flip_obs(o):
    o = o.copy(); o[:, FLIP_IDX] *= -1.0; return o
def remap_act(a):
    # DIRS 의 x부호를 뒤집으면 이동 인덱스가 되돌려진다. 1(위)·5(아래)는 그대로.
    return a if a in (0, 1, 5, 9, 10, 11) else 10 - a

def dir_action(dx, dy):
    n = math.hypot(dx, dy)
    if n < 1e-6: return 0
    dx, dy = dx/n, dy/n; best, bi = -9, 3
    for i, (ax, ay) in enumerate(DIRS):
        an = math.hypot(ax, ay); d = (dx*ax + dy*ay)/an
        if d > best: best, bi = d, i+1
    return bi

# ── 손코딩 정책 4종 (관측만 보고 행동) ─────────────────────────────
def p_stop(o, alive):   # 가만히 = 제자리 자동공격
    return [0 if alive[i] else 0 for i in range(len(o))]
def p_rush(o, alive):   # 사거리 안이면 쏘고 아니면 붙는다
    a = []
    for i in range(len(o)):
        if not alive[i]: a.append(0); continue
        ex, ey, inr = o[i][5]*256.0, o[i][6]*256.0, o[i][10] > 0.5
        a.append(9 if inr else dir_action(ex, ey))
    return a
def p_kite(o, alive):   # 쿨다운 중이면 물러난다 (무빙샷)
    a = []
    for i in range(len(o)):
        if not alive[i]: a.append(0); continue
        ex, ey, inr = o[i][5]*256.0, o[i][6]*256.0, o[i][10] > 0.5
        cd = o[i][1]
        if inr and cd < 0.1: a.append(9)
        elif inr:            a.append(11)
        else:                a.append(dir_action(ex, ey))
    return a
def p_spread(o, alive):  # 벌려서 접근
    a = []
    for i in range(len(o)):
        if not alive[i]: a.append(0); continue
        ex, ey, inr = o[i][5]*256.0, o[i][6]*256.0, o[i][10] > 0.5
        if inr: a.append(9)
        else:   a.append(dir_action(ex, ey + (60 if i % 2 else -60)))
    return a
POL = {"정지": p_stop, "돌진": p_rush, "무빙샷": p_kite, "벌리기": p_spread}

def duel(ally, n_ally, enemy, n_enemy, pol0, pol1, eps, seed, fs, r_step, extra=()):
    cmd = [BIN, "--data", MPQ, "--map", MAP, "--selfplay", "--strict-fire", "--box-range",
           "--ally", ally, "--enemy", enemy, "--allies", str(n_ally), "--enemies", str(n_enemy),
           "--frame-skip", str(fs), "--max-steps", "200", "--episodes", str(eps),
           "--seed", str(seed), "--r-step", str(r_step), *extra]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, cwd=MPQ)
    def rd(k):
        b = b""
        while len(b) < k:
            c = p.stdout.read(k - len(b))
            if not c: return None
            b += c
        return b
    def robs():
        ep, st, rew, done, nm, nz = struct.unpack("<iifBBB", rd(15))
        o = np.frombuffer(rd(4*nm*OBS), dtype=np.float32).reshape(nm, OBS)
        return done, o.copy(), rd(nm), st
    res, steps = [], []
    while True:
        t = rd(1)
        if t is None: break
        if t == b'E':
            term, dc, nm, mf = struct.unpack("<fBBf", rd(10))
            assert rd(1) == b'F'; rd(10)
            res.append(dc); continue
        done, o0, a0, st = robs()
        assert rd(1) == b'Q'; _, o1, a1, _ = robs()
        if done: steps.append(st); continue
        p.stdin.write(b'A' + bytes(x % 12 for x in pol0(o0, a0)))
        p.stdin.write(b'B' + bytes(x % 12 for x in pol1(o1, a1)))
        p.stdin.flush()
    try: p.stdin.close(); p.wait(timeout=60)
    except Exception: p.kill()
    w0 = res.count(1); w1 = res.count(2); to = res.count(3)
    return w0, w1, to, (sum(steps)/len(steps) if steps else 0)

def check_files():
    missing = [f for f in ("StarDat.mpq", "BrooDat.mpq", "patch_rt.mpq")
               if not os.path.exists(os.path.join(MPQ, f))
               and not os.path.exists(os.path.join(MPQ, f.replace("patch_rt", "Patch_rt")))]
    if missing:
        sys.exit(f"게임 파일이 없습니다: {', '.join(missing)}\n"
                 f"  넣을 곳: {MPQ}\n"
                 f"  받는 법: bash setup/get_mpq.sh  (블리자드 공식 AI 연구용)")
    if not os.path.exists(BIN):
        sys.exit(f"학습 환경이 빌드되지 않았습니다: {BIN}\n"
                 f"  빌드: cd engine && ./build.sh")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ally", default="vulture")
    ap.add_argument("--allies", type=int, default=1)
    ap.add_argument("--enemy", default="zergling")
    ap.add_argument("--enemies", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=12, help="조합마다 몇 판씩")
    ap.add_argument("--frame-skip", type=int, default=4)
    args = ap.parse_args()

    check_files()

    print("=" * 74)
    print(f"{args.ally} {args.allies}기 (P0)  vs  {args.enemy} {args.enemies}마리 (P1)")
    print(f"판단주기 {args.frame_skip}프레임 · 조합마다 {args.episodes * 2}판 (시드 2종)")
    print("=" * 74)

    hdr = "P0 \\ P1"
    print("  " + hdr.ljust(10) + "".join(f"{k:>12}" for k in POL))
    cells = {}
    for n0 in POL:
        row = f"  {n0:<10}"
        for n1 in POL:
            w0 = w1 = to = 0
            for sd in (3001, 4242):
                a, b, c, _ = duel(args.ally, args.allies, args.enemy, args.enemies,
                                  POL[n0], POL[n1], args.episodes, sd, args.frame_skip, 1)
                w0 += a; w1 += b; to += c
            cells[(n0, n1)] = (w0, w1, to)
            row += f"{w0:>4}:{w1:<3}{'*' if to > (w0 + w1 + to) / 2 else ' '}   "
        print(row)
    print("   (* = 시간초과가 절반을 넘은 칸)")

    total = args.episodes * 2
    best = None
    for (n0, n1), (w0, w1, to) in cells.items():
        decided = w0 + w1
        if decided < total * 0.5:
            continue
        gap = abs(w0 / decided - 0.5)
        if best is None or gap < best[0]:
            best = (gap, n0, n1, w0, w1)
    print()
    if best is None:
        print("★ 팽팽한 칸이 없습니다. 유닛 수나 판단주기를 바꿔보세요.")
    else:
        _, n0, n1, w0, w1 = best
        print(f"★ 가장 팽팽한 칸: (P0 {n0}, P1 {n1}) = {w0}:{w1}"
              f"  →  P0 승률 {w0 / (w0 + w1) * 100:.0f}%")
        print("   여기가 실력이 승부를 가르는 자리입니다. 학습은 이 칸을 넘는 걸 목표로 합니다.")
