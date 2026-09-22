"""사다리 — 셀프플레이의 진도를 '절대 좌표'로 재는 부품.

양쪽이 같이 배우면 승률이 50% 로 수렴해서, 승률만 봐서는 늘었는지 알 수가 없습니다.
그래서 여기서는 **움직이지 않는 상대(앵커)** 를 옆에 세워 두고 거기에 붙여 봅니다.
앵커는 실력 점수가 고정이라, 얘가 앵커를 얼마나 이기느냐가 곧 절대 좌표가 됩니다.

쓰는 법
    python3.11 ladder.py --anchors        앵커 4종끼리 전부 붙여서 대전표를 뽑습니다
    python3.11 ladder.py --import-check   지난 편 벌처(runs/vult1)를 불러와 붙여 봅니다
    python3.11 ladder.py --time           60판 한 번에 몇 초 걸리는지 잽니다

파이썬은 반드시 /opt/homebrew/bin/python3.11 (여기에만 torch 가 있습니다).
"""
import argparse, glob, json, math, os, queue, struct, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)          # 매치를 여러 개 동시에 돌리므로 스레드를 늘리지 않습니다

# ── 계약서의 공통 상수 ───────────────────────────────────────────────
BIN  = os.environ.get("SC_BIN", os.path.expanduser("~/scmicro/tools/bwmicro_sp"))
MPQ  = os.environ.get("SC_MPQ", os.path.expanduser("~/scmicro/data/mpq"))
MAP  = os.environ.get("SC_MAP", os.path.expanduser("~/scmicro/maps/Weave_v1.scx"))
RUNS = os.environ.get("SC_RUNS", os.path.expanduser("~/scmicro/runs_sp"))
OBS_DIM, N_ACT = 39, 12
ACT_NAME = ["정지", "이동1", "이동2", "이동3", "이동4", "이동5", "이동6", "이동7", "이동8",
            "최근접공격", "최저HP공격", "후퇴"]

# 항상 켜는 플래그 (UI 에 노출하지 않습니다)
FIXED_FLAGS = ["--selfplay", "--strict-fire", "--box-range"]

# 확정된 판: 벌처 1기(왼쪽·P0) vs 저글링 6마리(오른쪽·P1), 판정주기 4프레임
UNITS   = ("vulture", 1, "zergling", 6)
FRAME_SKIP = 4
MAX_STEPS  = 200

# 시드를 여러 개로 쪼개서 짝으로 잽니다. 한 시드로만 재면 진도가 아니라 노이즈를 봅니다.
DEFAULT_SEEDS = (3001, 4242, 5150, 6300)
DEFAULT_N     = 60

# 앵커 고정 레이팅 (계약서 표)
ANCHOR_ELO = {
    "anchor:stop":   800.0,
    "anchor:rush":  1000.0,
    "anchor:kite":  1200.0,
    "anchor:spread": 1050.0,
    "import:vult1": 1400.0,
}
BASE_ELO = 1000.0                 # 학습 중인 쪽과 gen000 의 출발점

VULT1_CKPT_DIR = os.path.join(os.environ.get("SC_OLD_RUNS", os.path.expanduser("~/scmicro/runs")), "vult1", "ckpt")


# ── 손코딩 정책 4종 (preflight.py 의 POL 을 그대로 복사) ──────────────
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def dir_action(dx, dy):
    """원하는 방향에 가장 가까운 이동 행동(1~8)."""
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 0
    dx, dy = dx / n, dy / n
    best, bi = -9, 3
    for i, (ax, ay) in enumerate(DIRS):
        an = math.hypot(ax, ay)
        d = (dx * ax + dy * ay) / an
        if d > best:
            best, bi = d, i + 1
    return bi


def p_stop(o, alive):        # 가만히 = 제자리 자동공격
    return [0 if alive[i] else 0 for i in range(len(o))]


def p_rush(o, alive):        # 사거리 안이면 쏘고 아니면 붙는다
    a = []
    for i in range(len(o)):
        if not alive[i]:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        a.append(9 if inr else dir_action(ex, ey))
    return a


def p_kite(o, alive):        # 쿨다운 중이면 물러난다 (무빙샷)
    a = []
    for i in range(len(o)):
        if not alive[i]:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        cd = o[i][1]
        if inr and cd < 0.1: a.append(9)
        elif inr:            a.append(11)
        else:                a.append(dir_action(ex, ey))
    return a


def p_spread(o, alive):      # 벌려서 접근
    a = []
    for i in range(len(o)):
        if not alive[i]:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        if inr: a.append(9)
        else:   a.append(dir_action(ex, ey + (60 if i % 2 else -60)))
    return a


POL = {"정지": p_stop, "돌진": p_rush, "무빙샷": p_kite, "벌리기": p_spread}
ANCHOR_POL = {"stop": p_stop, "rush": p_rush, "kite": p_kite, "spread": p_spread}
ANCHOR_KO  = {"stop": "정지", "rush": "돌진", "kite": "무빙샷", "spread": "벌리기"}


# ── 신경망 정책 (ppo_micro.py 와 같은 구조라야 체크포인트가 들어갑니다) ──
class Policy(nn.Module):
    """관측 39 → 은닉 128×2 → (행동 12, 앞으로 딸 점수 예상 1)."""

    def __init__(self, obs=OBS_DIM, hid=128, act=N_ACT):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                  nn.Linear(hid, hid), nn.Tanh())
        self.pi = nn.Linear(hid, act)
        self.v = nn.Linear(hid, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class NetPolicy:
    """체크포인트 하나를 '관측 → 행동' 함수로 감쌉니다.

    greedy=True 면 항상 제일 높은 행동을 고릅니다(재보기 좋게 흔들림이 줄어듭니다).
    기본은 학습 때와 같은 뽑기(sample) 입니다 — 학습 분포 그대로를 재야 하니까요.
    """

    def __init__(self, path, greedy=False, seed=4242, label=None):
        self.path = os.path.expanduser(path)
        self.greedy = greedy
        self.seed = seed
        self.label = label or os.path.basename(self.path)
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        self.meta = {k: v for k, v in ck.items() if k != "model"} if isinstance(ck, dict) else {}
        self.net = Policy()
        self.net.load_state_dict(sd)
        self.net.eval()
        self.gen = torch.Generator().manual_seed(seed)

    def clone(self, seed=None):
        """매치를 여러 개 동시에 돌릴 때 스레드마다 하나씩 씁니다."""
        c = NetPolicy.__new__(NetPolicy)
        c.path, c.greedy, c.label, c.meta = self.path, self.greedy, self.label, self.meta
        c.seed = self.seed if seed is None else seed
        c.net = self.net                      # 추론 전용이라 가중치는 같이 씁니다
        c.gen = torch.Generator().manual_seed(c.seed)
        return c

    def __call__(self, o, alive):
        with torch.no_grad():
            lg, _ = self.net(torch.as_tensor(np.ascontiguousarray(o)))
            if self.greedy:
                act = lg.argmax(-1)
            else:
                pr = torch.softmax(lg, -1)
                act = torch.multinomial(pr, 1, generator=self.gen).squeeze(-1)
        return [int(x) for x in act.tolist()]


def newest_ckpt(d):
    """디렉터리에서 판수가 가장 큰 체크포인트를 고릅니다."""
    c = sorted(glob.glob(os.path.join(os.path.expanduser(d), "*.pt")))
    if not c:
        return None
    def epnum(p):
        b = "".join(ch for ch in os.path.basename(p) if ch.isdigit())
        return int(b) if b else -1
    return max(c, key=epnum)


def load_policy(spec, run_dir=None, side="P0", greedy=False, seed=4242):
    """이름표 하나를 실제로 두는 함수로 바꿉니다.

    쓸 수 있는 이름표
        anchor:stop|rush|kite|spread   손코딩 4종
        import:vult1                   지난 편에서 학습한 벌처 (runs/vult1 마지막 체크포인트)
        past:<세대>  /  gen<세대>       이번 실험의 지난 세대 (run_dir 필요)
        <파일경로>.pt                   체크포인트 파일 직접 지정
    """
    if callable(spec):
        return spec
    if spec.startswith("anchor:"):
        name = spec.split(":", 1)[1]
        if name not in ANCHOR_POL:
            raise ValueError(f"모르는 앵커입니다: {spec}")
        return ANCHOR_POL[name]
    if spec == "import:vult1":
        ck = newest_ckpt(VULT1_CKPT_DIR)
        if ck is None:
            raise FileNotFoundError(f"지난 편 체크포인트가 없습니다: {VULT1_CKPT_DIR}")
        return NetPolicy(ck, greedy=greedy, seed=seed, label="import:vult1")
    if spec.startswith("past:") or spec.startswith("gen"):
        if run_dir is None:
            raise ValueError("past:/gen 이름표는 run_dir 이 있어야 합니다")
        g = int(spec.split(":", 1)[1] if ":" in spec else spec[3:])
        pre = "v" if side == "P0" else "z"
        p = os.path.join(run_dir, "ckpt", f"{pre}{g:03d}.pt")
        return NetPolicy(p, greedy=greedy, seed=seed, label=spec)
    if spec.endswith(".pt"):
        return NetPolicy(spec, greedy=greedy, seed=seed)
    raise ValueError(f"모르는 이름표입니다: {spec}")


# ── 한 판 붙이기 ─────────────────────────────────────────────────────
def _split(n, k):
    """n 판을 시드 k 종에 고르게 나눕니다. 60판·시드4종 → 15,15,15,15."""
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def _run_one(p0, p1, episodes, seed, units, frame_skip, max_steps, extra, timeout):
    """프로세스 하나로 episodes 판을 돌립니다. (내부용)"""
    if episodes <= 0:
        return dict(win0=0, win1=0, timeout=0, n=0, surv0=0.0, surv1=0.0, steps=0.0)
    u0, n0, u1, n1 = units
    cmd = [BIN, "--data", MPQ, "--map", MAP, *FIXED_FLAGS,
           "--ally", u0, "--enemy", u1, "--allies", str(n0), "--enemies", str(n1),
           "--frame-skip", str(frame_skip), "--max-steps", str(max_steps),
           "--episodes", str(episodes), "--seed", str(seed), *extra]
    p = subprocess.Popen(cmd, preexec_fn=lambda: os.nice(6), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, cwd=MPQ)
    # 게임이 멈춰 버리면 stdout.read 에서 영영 기다립니다. 시한이 지나면 강제로 죽입니다.
    watchdog = threading.Timer(timeout, p.kill)
    watchdog.daemon = True
    watchdog.start()

    def rd(k):
        b = b""
        while len(b) < k:
            c = p.stdout.read(k - len(b))
            if not c:
                return None
            b += c
        return b

    def robs():
        ep, st, rew, done, nm, nz = struct.unpack("<iifBBB", rd(15))
        o = np.frombuffer(rd(4 * nm * OBS_DIM), dtype=np.float32).reshape(nm, OBS_DIM)
        return done, o.copy(), rd(nm), st

    res, steps, sv0, sv1 = [], [], [], []
    try:
        while True:
            t = rd(1)
            if t is None:
                break
            if t == b'E':
                # 'E' 다음엔 반드시 'F' 가 옵니다. 둘 다 안 읽으면 C++ 이 즉사합니다.
                term0, dc, s0, mf0 = struct.unpack("<fBBf", rd(10))
                tag = rd(1)
                assert tag == b'F', f"'E' 다음이 'F' 가 아닙니다: {tag}"
                term1, dc1, s1, mf1 = struct.unpack("<fBBf", rd(10))
                res.append(dc); sv0.append(s0); sv1.append(s1)
                continue
            done, o0, a0, st = robs()
            tag = rd(1)
            assert tag == b'Q', f"'O' 다음이 'Q' 가 아닙니다: {tag}"
            _, o1, a1, _ = robs()
            if done:
                steps.append(st)
                continue
            # 행동은 반드시 bytes(int(x) % 12 ...) 로. numpy 를 그냥 넣으면 프로토콜이 깨집니다.
            p.stdin.write(b'A' + bytes(int(x) % N_ACT for x in p0(o0, a0)))
            p.stdin.write(b'B' + bytes(int(x) % N_ACT for x in p1(o1, a1)))
            p.stdin.flush()
    finally:
        watchdog.cancel()
        try:
            p.stdin.close(); p.wait(timeout=30)
        except Exception:
            p.kill()
    return dict(win0=res.count(1), win1=res.count(2), timeout=res.count(3), n=len(res),
                surv0=float(np.mean(sv0)) if sv0 else 0.0,
                surv1=float(np.mean(sv1)) if sv1 else 0.0,
                steps=float(np.mean(steps)) if steps else 0.0)


def match(p0, p1, n=DEFAULT_N, units=UNITS, frame_skip=FRAME_SKIP,
          seeds=DEFAULT_SEEDS, max_steps=MAX_STEPS, extra=(), workers=0, timeout=600):
    """두 정책을 n 판 붙입니다.

    ★ --eval 은 절대 안 씁니다. 스폰이 (lo+hi)/2 로 굳어서 N판이 사실상 1판이 됩니다.
      대신 --seed 를 고정하고 --episodes N 으로 돌립니다 (판마다 시드 = 기준시드+판번호).
    ★ 시드를 여러 종으로 쪼개서 짝으로 잽니다. 한 시드로만 재면 노이즈를 진도로 착각합니다.

    반환: {"win0","win1","timeout","n","surv0","surv1","steps"}
    """
    seeds = tuple(seeds)
    parts = _split(n, len(seeds))
    workers = workers or min(len(seeds), max(1, (os.cpu_count() or 4) - 1))

    def one(i):
        # 스레드마다 자기 신경망 복제본을 씁니다 (손코딩 정책은 상태가 없어 그대로 씁니다).
        q0 = p0.clone(seed=4242 + i) if hasattr(p0, "clone") else p0
        q1 = p1.clone(seed=8484 + i) if hasattr(p1, "clone") else p1
        return _run_one(q0, q1, parts[i], seeds[i], units, frame_skip, max_steps, extra, timeout)

    if workers <= 1:
        outs = [one(i) for i in range(len(seeds))]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(one, range(len(seeds))))

    tot = dict(win0=0, win1=0, timeout=0, n=0)
    for k in ("win0", "win1", "timeout", "n"):
        tot[k] = sum(o[k] for o in outs)
    wsum = sum(o["n"] for o in outs) or 1
    for k in ("surv0", "surv1", "steps"):
        tot[k] = round(sum(o[k] * o["n"] for o in outs) / wsum, 2)
    return tot


# ── 실력 점수(Elo) ──────────────────────────────────────────────────
def expected_score(ra, rb):
    """레이팅 차이로 기대 승률을 냅니다."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def elo_update(ratings, result, k=24):
    """한 매치 결과로 실력 점수를 갱신합니다. 앵커는 고정, 학습 중인 쪽만 움직입니다.

    result 는 ladder.jsonl 한 줄과 같은 모양입니다:
        {"me":"v014","foe":"anchor:kite","n":60,"win":38,"lose":22,"to":0}
    시간초과는 무승부(0.5) 로 셉니다.

    ★ K 는 '한 판'이 아니라 '한 매치(60판)' 에 한 번만 먹입니다.
      그래서 한 번에 최대 ±24 만 움직이고, 이게 그 자체로 지수이동평균이 됩니다.
      판마다 먹이면 60판에 ±1440 이 튀어서 지표가 못 쓰게 됩니다.
    """
    out = dict(ratings)
    me, foe = result["me"], result["foe"]
    n = max(1, int(result.get("n", 0)))
    win = int(result.get("win", 0)); to = int(result.get("to", 0))
    score = (win + 0.5 * to) / n

    r_me = out.get(me, BASE_ELO)
    r_foe = ANCHOR_ELO.get(foe, out.get(foe, BASE_ELO))
    out[foe] = r_foe                       # 앵커·지난 세대는 고정값을 그대로 유지합니다
    if me in ANCHOR_ELO:                   # 앵커는 절대 안 움직입니다
        out[me] = ANCHOR_ELO[me]
        return out
    out[me] = round(r_me + k * (score - expected_score(r_me, r_foe)), 2)
    return out


def fit_elo(results, lo=200.0, hi=2400.0, prior=1.0):
    """앵커 성적 여러 개를 한꺼번에 놓고 실력 점수를 곧바로 풀어냅니다.

    K 를 조금씩 먹이는 방식(elo_update)은 출발점 1000 에서 천천히 기어가서,
    "0승/180판" 같은 명백한 바닥도 몇 판 만에는 못 보여 줍니다.
    앵커는 레이팅이 이미 고정이라, 그냥 방정식을 풀면 됩니다 —
      기대승수의 합 == 실제로 딴 승수의 합
    이 되는 지점을 이분법으로 찾습니다. 한쪽 눈금이 곧 절대 좌표가 됩니다.

    results: [{"foe":"anchor:kite","n":60,"win":44,"to":3}, ...]  (앵커 상대만)
    prior:   전승/전패에서 값이 무한대로 튀지 않게 넣는 가상의 무승부 판수
    """
    rows = [r for r in results if r["foe"] in ANCHOR_ELO]
    if not rows:
        return None
    got = sum(r["win"] + 0.5 * r.get("to", 0) for r in rows) + prior * 0.5 * len(rows)

    def exp_at(x):
        e = sum(r["n"] * expected_score(x, ANCHOR_ELO[r["foe"]]) for r in rows)
        e += sum(prior * expected_score(x, ANCHOR_ELO[r["foe"]]) for r in rows)
        return e

    a, b = lo, hi
    for _ in range(60):
        m = (a + b) / 2
        if exp_at(m) < got: a = m
        else: b = m
    return round((a + b) / 2, 1)


# ── 사다리 워커 ─────────────────────────────────────────────────────
class Ladder:
    """백그라운드에서 사다리를 굴립니다.

    새 세대가 나오면 자동으로
      (1) 그 세대 vs gen000
      (2) 그 세대 vs 앵커 3종 (돌진·무빙샷·벌리기)
      (3) 30세대마다 그 세대 vs 10세대 전 (빙빙 도는지 확인)
    을 큐에 넣고 순서대로 돌려서 ladder.jsonl 에 한 줄씩 붙입니다.

    ladder.jsonl 에 쓰는 놈은 이 클래스 하나뿐입니다. 그래서 락이 없습니다.
    """

    ANCHORS = ("anchor:rush", "anchor:kite", "anchor:spread")
    EMA_A = 0.4                      # 승률에 남은 노이즈를 누르는 지수이동평균 계수
    FIT_EMA_A = 0.5                  # 실력 점수 쪽 지수이동평균 계수

    def __init__(self, run_dir, n=DEFAULT_N, seeds=DEFAULT_SEEDS, units=UNITS,
                 frame_skip=FRAME_SKIP, max_steps=MAX_STEPS, k=24,
                 greedy=False, workers=0, on_row=None, extra=()):
        self.run_dir = os.path.expanduser(run_dir)
        os.makedirs(os.path.join(self.run_dir, "ckpt"), exist_ok=True)
        self.path = os.path.join(self.run_dir, "ladder.jsonl")
        self.n, self.seeds, self.units = n, tuple(seeds), units
        self.frame_skip, self.max_steps, self.k = frame_skip, max_steps, k
        self.greedy, self.workers = greedy, workers
        self.extra = tuple(extra)
        self.on_row = on_row                      # 한 줄 쓸 때마다 부를 콜백 (studio.py 용)

        self.q = queue.Queue()
        self.ratings = {**ANCHOR_ELO, "P0": BASE_ELO, "P1": BASE_ELO}
        self.gen_elo = {"P0": {0: BASE_ELO}, "P1": {0: BASE_ELO}}   # 세대 스냅샷
        self.wr_ema = {}
        self.fit_window = len(self.ANCHORS)   # 앵커 매치 최근 3개 = 그 세대 한 바퀴
        self.recent = {"P0": [], "P1": []}
        self.fit = {"P0": None, "P1": None}
        self.rows = []
        self._stop = threading.Event()
        self._th = None
        self._cache = {}

    # ── 바깥에서 부르는 것들 ───────────────────────────────────────
    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self._loop, name="ladder", daemon=True)
            self._th.start()
        return self

    def stop(self, wait=2.0):
        self._stop.set()
        self.q.put(None)
        if self._th:
            self._th.join(timeout=wait)
        return self

    def submit_gen(self, gen, ep):
        """새 세대가 나왔다고 알려 줍니다. 붙일 상대들을 큐에 채웁니다."""
        for side in ("P0", "P1"):
            self.gen_elo[side][gen] = self.ratings[side]
        jobs = []
        if gen > 0:
            for side in ("P0", "P1"):
                jobs.append((side, gen, ep, "past:0"))
                for a in self.ANCHORS:
                    jobs.append((side, gen, ep, a))
            if gen % 30 == 0 and gen >= 10:
                for side in ("P0", "P1"):
                    jobs.append((side, gen, ep, f"past:{gen - 10}"))
        else:
            for side in ("P0", "P1"):
                for a in self.ANCHORS:
                    jobs.append((side, gen, ep, a))
        for j in jobs:
            self.q.put(j)
        return len(jobs)

    def pending(self):
        return self.q.qsize()

    def elo(self, side="P0"):
        return self.ratings[side]

    # ── 안쪽 ────────────────────────────────────────────────────────
    CACHE_MAX = 24

    def _pol(self, side, gen):
        """이번 실험의 세대 정책. side 에 따라 v###.pt / z###.pt 를 씁니다."""
        key = (side, gen)
        if key not in self._cache:
            pre = "v" if side == "P0" else "z"
            p = os.path.join(self.run_dir, "ckpt", f"{pre}{gen:03d}.pt")
            self._cache[key] = NetPolicy(p, greedy=self.greedy, label=f"{pre}{gen:03d}")
            # gen000 은 늘 붙는 상대라 남기고, 나머지는 오래된 것부터 버립니다.
            while len(self._cache) > self.CACHE_MAX:
                for k in list(self._cache):
                    if k[1] != 0 and k != key:
                        del self._cache[k]; break
                else:
                    break
        return self._cache[key]

    def _foe_pol(self, side, foe):
        """상대 정책. 앵커면 손코딩, past:N 이면 반대쪽 세대 정책입니다."""
        if foe.startswith("anchor:"):
            return ANCHOR_POL[foe.split(":", 1)[1]]
        if foe == "import:vult1":
            return load_policy("import:vult1", greedy=self.greedy)
        if foe.startswith("past:"):
            other = "P1" if side == "P0" else "P0"
            return self._pol(other, int(foe.split(":", 1)[1]))
        raise ValueError(f"모르는 상대입니다: {foe}")

    def _foe_rating(self, side, foe):
        if foe in ANCHOR_ELO:
            return ANCHOR_ELO[foe]
        if foe.startswith("past:"):
            other = "P1" if side == "P0" else "P0"
            return self.gen_elo[other].get(int(foe.split(":", 1)[1]), BASE_ELO)
        return BASE_ELO

    def run_job(self, side, gen, ep, foe):
        """한 매치를 돌리고 ladder.jsonl 에 한 줄 붙입니다."""
        me = self._pol(side, gen)
        fo = self._foe_pol(side, foe)
        # P0(벌처)를 재는 매치면 내가 왼쪽, 상대가 오른쪽. P1(저글링)이면 반대입니다.
        p0, p1 = (me, fo) if side == "P0" else (fo, me)
        r = match(p0, p1, n=self.n, units=self.units, frame_skip=self.frame_skip,
                  seeds=self.seeds, max_steps=self.max_steps, workers=self.workers,
                  extra=self.extra)
        win  = r["win0"] if side == "P0" else r["win1"]
        lose = r["win1"] if side == "P0" else r["win0"]

        row = {"t": "match", "ep": ep, "gen": gen, "side": side,
               "me": ("v" if side == "P0" else "z") + f"{gen:03d}", "foe": foe,
               "n": r["n"], "win": win, "lose": lose, "to": r["timeout"]}
        # 실력 점수의 주인은 '지금 학습 중인 그 진영'입니다. 세대 이름이 아니라 P0/P1 이 키입니다.
        self.ratings[foe] = self._foe_rating(side, foe)
        self.ratings = elo_update(self.ratings, {**row, "me": side}, k=self.k)
        row["elo_me"] = self.ratings[side]

        # 남는 노이즈를 지수이동평균으로 눌러서 같이 적어 둡니다 (그래프용).
        wr = (win + 0.5 * r["timeout"]) / max(1, r["n"])
        key = (side, foe)
        self.wr_ema[key] = wr if key not in self.wr_ema else \
            self.EMA_A * wr + (1 - self.EMA_A) * self.wr_ema[key]
        row["wr"] = round(wr, 4)
        row["wr_ema"] = round(self.wr_ema[key], 4)
        row["surv"] = r["surv0"] if side == "P0" else r["surv1"]
        row["steps"] = r["steps"]

        # 앵커 성적을 모아 실력 점수를 곧바로 풀고, 남는 노이즈만 지수이동평균으로 누릅니다.
        if foe in ANCHOR_ELO:
            self.recent[side].append(row)
            del self.recent[side][:-self.fit_window]
            f = fit_elo(self.recent[side])
            row["elo_raw"] = f                     # 이번 한 바퀴만 보고 푼 날것
            self.fit[side] = f if self.fit[side] is None else \
                round(self.FIT_EMA_A * f + (1 - self.FIT_EMA_A) * self.fit[side], 1)
        if self.fit[side] is not None:
            row["elo_fit"] = self.fit[side]        # ★ 진도 그래프는 이 값을 씁니다

        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)
        if self.on_row:
            try: self.on_row(row)
            except Exception: pass
        return row

    def _loop(self):
        while not self._stop.is_set():
            job = self.q.get()
            if job is None:
                break
            try:
                self.run_job(*job)
            except Exception as e:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"t": "err", "job": list(job), "msg": str(e)},
                                       ensure_ascii=False) + "\n")


# ── 검증용 명령들 ────────────────────────────────────────────────────
def cmd_anchors(args):
    """앵커 4종끼리 전부 붙여서 대전표를 뽑습니다 (계약서 실측표와 맞춰 보는 용도)."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    n = args.n
    print("=" * 78)
    print(f"앵커 대전표 — 벌처{UNITS[1]} vs 저글링{UNITS[3]}, 판정주기 {args.frame_skip}프레임, "
          f"{n}판씩 (시드 {seeds}, 판수 {_split(n, len(seeds))})")
    print("행 = 벌처(P0) 정책, 열 = 저글링(P1) 정책, 칸 = 벌처승:저글링승  (* = 시간초과 절반 초과)")
    print("=" * 78)
    t0 = time.time()
    print("  " + "벌처\\저글링".ljust(12) + "".join(f"{k:>12}" for k in POL))
    grid = {}
    for a in POL:
        row = "  " + a.ljust(12)
        for b in POL:
            r = match(POL[a], POL[b], n=n, frame_skip=args.frame_skip, seeds=seeds,
                      workers=args.workers)
            grid[(a, b)] = r
            star = "*" if r["timeout"] > r["n"] // 2 else " "
            row += f"{r['win0']:>5}:{r['win1']:<4}{star}"
        print(row, flush=True)
    print(f"\n  총 {len(POL)**2 * n}판 · {time.time() - t0:.1f}초")

    # 계약서 실측표와 대조 (24판 기준을 이번 판수로 환산해서 비교)
    ref = {("정지","정지"):(0,0), ("정지","돌진"):(0,24), ("정지","무빙샷"):(0,24), ("정지","벌리기"):(12,12),
           ("돌진","정지"):(0,24), ("돌진","돌진"):(0,24), ("돌진","무빙샷"):(0,24), ("돌진","벌리기"):(7,17),
           ("무빙샷","정지"):(20,1), ("무빙샷","돌진"):(12,11), ("무빙샷","무빙샷"):(17,7), ("무빙샷","벌리기"):(15,9),
           ("벌리기","정지"):(0,24), ("벌리기","돌진"):(0,24), ("벌리기","무빙샷"):(0,24), ("벌리기","벌리기"):(10,14)}
    print("\n  계약서 실측표(24판 기준)와 대조 — 벌처 승률 차이")
    worst = 0.0
    for a in POL:
        for b in POL:
            rw0, rw1 = ref[(a, b)]
            rn = rw0 + rw1
            got = grid[(a, b)]
            gn = got["win0"] + got["win1"]
            if rn == 0 and gn == 0:
                print(f"    {a:<6} vs {b:<6}  둘 다 전부 시간초과 → 일치")
                continue
            pr = rw0 / rn if rn else float("nan")
            pg = got["win0"] / gn if gn else float("nan")
            d = abs(pg - pr) if rn and gn else float("nan")
            worst = max(worst, 0.0 if math.isnan(d) else d)
            flag = "" if (math.isnan(d) or d <= 0.20) else "   ← 차이 큼"
            print(f"    {a:<6} vs {b:<6}  계약서 {pr*100:5.1f}%  이번 {pg*100:5.1f}%  "
                  f"차이 {0 if math.isnan(d) else d*100:4.1f}%p{flag}")
    print(f"\n  최대 차이 {worst*100:.1f}%p → " +
          ("판정: 표가 재현됩니다" if worst <= 0.20 else "판정: 어긋납니다. 원인을 찾아야 합니다"))
    return grid


def cmd_import_check(args):
    """지난 편 벌처(runs/vult1)가 진짜 로드되고 성적이 나오는지 봅니다."""
    print("=" * 78)
    print("import:vult1 수입 검사")
    print("=" * 78)
    ck = newest_ckpt(VULT1_CKPT_DIR)
    print(f"  체크포인트: {ck}")
    if ck is None:
        print("  ✗ 파일이 없습니다"); return
    net = NetPolicy(ck, greedy=args.greedy, label="import:vult1")
    print(f"  메타: {net.meta}")
    nparam = sum(p.numel() for p in net.net.parameters())
    print(f"  가중치 {nparam:,}개 로드 성공 · 행동 뽑기 = {'제일 높은 것' if args.greedy else '확률대로'}")
    o = np.zeros((1, OBS_DIM), dtype=np.float32); o[0, 5] = 0.4; o[0, 7] = 0.4
    a = net(o, bytes([1]))[0]
    print(f"  더미 관측 한 번 넣어 봤을 때 행동: {a} {ACT_NAME[a]}")

    seeds = tuple(int(s) for s in args.seeds.split(","))
    n = args.n
    print(f"\n  [1] 계약서 판 (벌처1 vs 저글링6 · 4프레임 · --box-range · max-steps {MAX_STEPS})")
    print(f"      {n}판씩, 시드 {seeds}")
    for name in ("stop", "rush", "kite", "spread"):
        t0 = time.time()
        r = match(net, ANCHOR_POL[name], n=n, seeds=seeds, workers=args.workers)
        wr = r["win0"] / max(1, r["n"])
        print(f"      vs {ANCHOR_KO[name]:<6} {r['win0']:>3}:{r['win1']:<3} "
              f"(시간초과 {r['timeout']:>2})  벌처 승률 {wr*100:5.1f}%  "
              f"평균 {r['steps']:.0f}스텝  {time.time()-t0:.1f}초", flush=True)
    print("\n  [2] 같은 판에서 손코딩 무빙샷은 얼마나 하는지 (기준선)")
    for name in ("rush", "kite", "spread"):
        r = match(p_kite, ANCHOR_POL[name], n=n, seeds=seeds, workers=args.workers)
        print(f"      손코딩 무빙샷 vs {ANCHOR_KO[name]:<6} {r['win0']:>3}:{r['win1']:<3} "
              f"→ {r['win0']/max(1,r['n'])*100:5.1f}%", flush=True)

    if args.vult1_board:
        print(f"\n  [3] 지난 편이 학습했던 판 그대로 (cx752 cy1024 gap440~560 max-steps 350)")
        extra = ["--cx", "752", "--cy", "1024", "--gap-lo", "440", "--gap-hi", "560"]
        for name in ("rush", "kite", "spread"):
            r = match(net, ANCHOR_POL[name], n=n, seeds=seeds, max_steps=350,
                      extra=extra, workers=args.workers)
            print(f"      vs {ANCHOR_KO[name]:<6} {r['win0']:>3}:{r['win1']:<3} "
                  f"→ {r['win0']/max(1,r['n'])*100:5.1f}%", flush=True)
        for name in ("kite",):
            r = match(p_kite, ANCHOR_POL[name], n=n, seeds=seeds, max_steps=350,
                      extra=extra, workers=args.workers)
            print(f"      손코딩 무빙샷 vs {ANCHOR_KO[name]:<6} {r['win0']:>3}:{r['win1']:<3} "
                  f"→ {r['win0']/max(1,r['n'])*100:5.1f}%", flush=True)


def cmd_time(args):
    """60판 한 번에 몇 초 걸리는지 잽니다."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    print(f"60판 벽시계 시간 측정 (시드 {seeds}, 판수 {_split(args.n, len(seeds))})")
    for label, p0, p1, w in (("무빙샷 vs 돌진 (한 프로세스씩 차례로)", p_kite, p_rush, 1),
                             (f"무빙샷 vs 돌진 (동시 {len(seeds)}개)", p_kite, p_rush, len(seeds))):
        t0 = time.time()
        r = match(p0, p1, n=args.n, seeds=seeds, workers=w)
        dt = time.time() - t0
        print(f"  {label:<38} {dt:6.1f}초  ({r['win0']}:{r['win1']}, "
              f"평균 {r['steps']:.0f}스텝, 판당 {dt/max(1,r['n'])*1000:.0f}ms)")
    ck = newest_ckpt(VULT1_CKPT_DIR)
    if ck:
        net = NetPolicy(ck)
        t0 = time.time()
        r = match(net, p_rush, n=args.n, seeds=seeds, workers=len(seeds))
        dt = time.time() - t0
        print(f"  {'신경망 vs 돌진 (동시)':<38} {dt:6.1f}초  ({r['win0']}:{r['win1']}, "
              f"판당 {dt/max(1,r['n'])*1000:.0f}ms)")
    print("\n  사다리 한 세대 = 8매치(양쪽 × (gen000 + 앵커3종)) = 480판.")


def cmd_elo_demo(args):
    """Elo 갱신이 앵커를 고정한 채 학습 쪽만 움직이는지 눈으로 봅니다."""
    r = {**ANCHOR_ELO, "P0": BASE_ELO}
    print(f"  시작   P0={r['P0']:.1f}  anchor:kite={r['anchor:kite']:.1f}")
    for i in range(8):
        r = elo_update(r, {"me": "P0", "foe": "anchor:kite", "n": 60, "win": 45, "lose": 15, "to": 0})
        print(f"  {i+1:>2}번째 60판 45승  P0={r['P0']:>7.1f}  anchor:kite={r['anchor:kite']:.1f}")
    r2 = {**ANCHOR_ELO, "P0": BASE_ELO}
    for i in range(8):
        r2 = elo_update(r2, {"me": "P0", "foe": "anchor:kite", "n": 60, "win": 15, "lose": 45, "to": 0})
    print(f"  거꾸로 15승만 8번 → P0={r2['P0']:.1f}")


def main():
    ap = argparse.ArgumentParser(description="사다리 — 셀프플레이 진도를 절대 좌표로 재기")
    ap.add_argument("--anchors", action="store_true", help="앵커 4종 대전표")
    ap.add_argument("--import-check", action="store_true", help="지난 편 벌처 수입 검사")
    ap.add_argument("--time", action="store_true", help="60판 벽시계 시간")
    ap.add_argument("--elo-demo", action="store_true", help="Elo 갱신 확인")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="한 매치 판수")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    ap.add_argument("--workers", type=int, default=0, help="동시에 돌릴 프로세스 수 (0=자동)")
    ap.add_argument("--greedy", action="store_true", help="신경망이 제일 높은 행동만 고르게")
    ap.add_argument("--vult1-board", action="store_true", help="지난 편 판 설정으로도 재기")
    args = ap.parse_args()

    if args.anchors:      cmd_anchors(args)
    if args.import_check: cmd_import_check(args)
    if args.time:         cmd_time(args)
    if args.elo_demo:     cmd_elo_demo(args)
    if not any([args.anchors, args.import_check, args.time, args.elo_demo]):
        ap.print_help()


if __name__ == "__main__":
    main()
