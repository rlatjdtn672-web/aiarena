#!/usr/bin/env python3.11
"""스타 AI 스튜디오 — 통합 서버.

브라우저에서 점수를 만지고, 학습을 돌리고, 중간중간 게임을 보고, 그 화면을 녹화한다.
부품(train_sp / diagnose / spectator / ladder)을 붙이는 접착제다.

  ./go.sh          또는          python3.11 studio.py --port 8777 --open

계약: studio/CONTRACT.md
"""
import argparse, http.server, json, os, queue, shutil, signal, socketserver
import subprocess, sys, threading, time, traceback, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HOME = os.path.expanduser("~")
PY311 = os.environ.get("SC_PY", "/opt/homebrew/bin/python3.11")
FFMPEG = os.environ.get("SC_FFMPEG", "/opt/homebrew/bin/ffmpeg")
BIN = os.environ.get("SC_BIN", os.path.join(HOME, "scmicro/tools/bwmicro_sp"))
MPQ = os.environ.get("SC_MPQ", os.path.join(HOME, "scmicro/data/mpq"))
MAP = os.environ.get("SC_MAP", os.path.join(HOME, "scmicro/maps/Weave_v1.scx"))
RUNS = os.environ.get("SC_RUNS", os.path.join(HOME, "scmicro/runs_sp"))
OLD_RUNS = os.environ.get("SC_OLD_RUNS", os.path.join(HOME, "scmicro/runs"))
TRAIN = os.path.join(HERE, "train_sp.py")
UI = os.path.join(HERE, "ui.html")

# ── 판 목록. SC_PRESET 환경변수로 고른다 ────────────────────────────
#    각 판의 상수는 전부 손코딩 대전표를 실제로 재서 정한 값이다.
PRESETS = {
    # 서로 붙이기 (셀프플레이). 투혼 위에서 무빙샷 vs 돌진 32:27 실측.
    "vulture": {
        "이름": "벌처 1기 vs 저글링 6마리 (서로 학습)",
        "bin": BIN,
        "units": ("vulture", 1, "zergling", 6),
        "frame_skip": 4, "max_steps": 200,
        # 투혼 7시 방향 열린 흙바닥. 무빙샷 vs 돌진 32:27 실측 (9/22)
        "extra": ["--cx", "1280", "--cy", "2560"],
        "설명": "양쪽이 같이 배운다. 벌처는 무빙샷, 저글링은 달려드는 법.",
    },
    # 마인 제거. 벌처는 마인만 심고 얼려둔다. AI 조건 = 마인을 직접 찍을 수 있다.
    "dragoon": {
        "이름": "드라군 2기 vs 마인 4개 — AI 조건 (마인 찍기 허용)",
        "bin": os.path.join(HOME, "scmicro/tools/bwdragoon"),
        "units": ("dragoon", 2, "vulture", 1),
        "frame_skip": 4, "max_steps": 300,
        "extra": ["--mines", "4", "--mine-gap", "150", "--win-vulture",
                  "--stall-steps", "100"],
        # ★ 커리큘럼은 뺐다. 마인 1개로 시작하면 무작위 정책이 마인을 아예 못 만나서
        #   25만 판 동안 점수가 −25.30 에 고정됐다(학습 신호 0). 마인 4개면 바로 부딪힌다.
        #   "쉬운 것부터"가 항상 옳은 건 아니다 — 여기선 쉬운 판이 곧 '아무 일도 안 일어나는 판'이었다.
        "curriculum": "",
        "설명": "마인이 올라오는 순간 정확히 찍는 법을 배운다. 사람은 못 하는 조작이다.",
    },
    # ★ 사람 조건. 찍기를 막으면 정지(자동공격)와 이동만 남고, 택견이 유일한 답이 된다.
    #   손코딩 실측(마인 4개): 몸빵 9% / 후퇴4 39% / 후퇴6 53% / 후퇴8 70%.
    #   마인 3개는 몸빵으로도 24%가 나와서 판이 물렀다 — 드라군 체력 180, 폭발 125라
    #   한 방은 버틴다. 4개면 2기가 몸으로 받을 수 있는 게 2개뿐이라 몸빵이 죽는다.
    "dragoon-human": {
        "이름": "드라군 2기 vs 마인 4개 — 사람 조건 (찍기 금지)",
        "bin": os.path.join(HOME, "scmicro/tools/bwdragoon"),
        "units": ("dragoon", 2, "vulture", 1),
        "frame_skip": 4, "max_steps": 300,
        "extra": ["--mines", "4", "--mine-gap", "150", "--no-click", "--win-vulture",
                  "--stall-steps", "100"],
        # ★ 커리큘럼은 뺐다. 마인 1개로 시작하면 무작위 정책이 마인을 아예 못 만나서
        #   25만 판 동안 점수가 −25.30 에 고정됐다(학습 신호 0). 마인 4개면 바로 부딪힌다.
        #   "쉬운 것부터"가 항상 옳은 건 아니다 — 여기선 쉬운 판이 곧 '아무 일도 안 일어나는 판'이었다.
        "curriculum": "",
        "설명": "마인 4개를 뚫고 뒤의 벌처까지 잡아야 이긴다. 마인은 못 찍는다 — "
                "쓸 수 있는 건 사람과 같은 키뿐이다(이동·후퇴·정지·홀드). "
                "실측: 그냥 전진 0% / 뒤로 빼고 정지(홀드 없이) 0% / 뒤 한 발짝→홀드 87%.",
    },
    # ★★ 사람 조건 완전판. 찍기 금지에 더해 '반응속도'와 '손 속도'까지 사람 수준으로 묶는다.
    #   --react-frames 6 : 명령이 6프레임(252ms) 뒤에 나간다. 사람 평균 시각반응 273ms.
    #   --apm 200 --apm-group : 분당 명령 200회, 두 기를 한 부대로 묶어서 센다(사람은 그렇게 한다).
    #   실측(이 조건, 60판씩): 그냥 전진 0% / 마인 보고 뒤로 빼기 3% / 박자 택견 93%.
    #   ★ 반응형 택견이 87%→3% 로 무너진다. 지금까지의 '택견'은 사람이 못 내는 반사신경에
    #     기대고 있었다는 뜻이다. 반대로 '박자 택견'(보지 않고 앞3·뒤1·홀드5 를 반복)은
    #     반응속도가 아예 필요 없어서 93% 로 살아남는다 — 이게 사람이 실제로 쓰는 그 스텝이다.
    "dragoon-real": {
        "이름": "드라군 2기 vs 마인 4개 — 사람 조건 완전판 (반응 252ms · 200 APM)",
        "bin": os.path.join(HOME, "scmicro/tools/bwdragoon"),
        "units": ("dragoon", 2, "vulture", 1),
        "frame_skip": 4, "max_steps": 300,
        "extra": ["--mines", "4", "--mine-gap", "150", "--no-click", "--win-vulture",
                  "--stall-steps", "100", "--nocontact-steps", "150",
                  "--react-frames", "6", "--apm", "200", "--apm-group"],
        "curriculum": "",
        "설명": "찍기도 못 하고, 명령이 0.25초 늦게 나가고, 손도 분당 200번뿐이다. "
                "반사신경으로 되는 게 하나도 없어서 박자(택견)밖에 답이 없다. "
                "실측: 전진 0% / 반응형 뒤로 빼기 3% / 박자 앞3·뒤1·홀드5 93%.",
    },
    # ★★★ 간격 무작위 판. 왜 만들었나:
    #   간격이 150px 로 고정이면 '눈 감고 박자만 세는' 정책 하나로 판이 다 뚫린다.
    #   실측 — 베낀 망에 화면 39개를 전부 0 으로 넣어도 승률 87% → 84% 밖에 안 떨어졌다.
    #   즉 지금까지의 판은 화면을 볼 필요가 없는 판이었다. 지능이 필요 없는 판을 만들어 놓고
    #   "AI 가 못 배운다" 고 한 셈이다. 판마다 간격이 달라지면 박자를 미리 못 세고,
    #   마인이 올라오는 걸 보고 판단해야만 이길 수 있다.
    "dragoon-rand": {
        "이름": "드라군 2기 vs 마인 4개 — 간격 무작위 (눈을 떠야 이기는 판)",
        "bin": os.path.join(HOME, "scmicro/tools/bwdragoon"),
        "units": ("dragoon", 2, "vulture", 1),
        "frame_skip": 4, "max_steps": 300,
        # ★ --mine-gap 150 은 '밭 전체 폭' 을 정한다(맵이 2048px 뿐이라 폭이 늘면 스폰이 통째로 실패).
        #   lo/hi 는 그 폭 안에서 간격을 흩뜨리는 몫이다. 예: 143 / 177 / 153 / 127px
        "extra": ["--mines", "4", "--mine-gap", "150",
                  "--mine-gap-lo", "120", "--mine-gap-hi", "300",
                  "--mine-jitter", "40", "--no-click", "--win-vulture",
                  "--stall-steps", "100", "--nocontact-steps", "150",
                  "--react-frames", "6", "--apm", "200", "--apm-group"],
        "curriculum": "",
        # ★ 실측 결과 — 이 판도 화면을 볼 필요가 없었다(눈 감은 박자 82% > 눈 뜬 박자 77%).
        #   배치를 흔드는 것으로는 '눈이 필요한 판' 이 안 만들어진다. 반응속도가 갈랐다.
        "설명": "마인 간격이 판마다 달라지고 세로로도 ±40px 흔들린다(밭 전체 폭은 고정). "
                "실측: 손코딩 박자 77% · 그 박자를 눈 감고 82% · 반응형 8% · 옛 조건 AI 57%.",
    },
}
PRESET = os.environ.get("SC_PRESET", "vulture")
if PRESET not in PRESETS:
    raise SystemExit(f"모르는 판입니다: {PRESET}. 쓸 수 있는 것: {', '.join(PRESETS)}")
P = PRESETS[PRESET]
BIN = P["bin"]
UNITS = P["units"]
FRAME_SKIP = P["frame_skip"]
MAX_STEPS = P["max_steps"]
EXTRA = P["extra"]
CURRICULUM = P.get("curriculum", "")

SWAP_EVERY = 15.0      # 관전 선수를 갈아끼우는 최소 간격(초)

DEFAULT_SLIDERS = {
    # ★ 내 체력 500. 마인 폭발 한 방(125/180=0.69)이 −3.47 이라 마인 처치(+2.0)보다 비싸다.
    #   30 이었을 때는 한 방이 −0.21 이라 '맞고 지나가기' 가 점수상 완전히 합리적이었다.
    "r-enemy-hp": 300, "r-ally-hp": 500, "r-kill": 200, "r-death": 300,
    "r-win": 1000, "r-lose": 1000, "r-timeout": 2500, "r-misfire": 0, "r-step": 1,
    "ent": 0.01, "gamma": 0.999, "frame-skip": FRAME_SKIP,
}
ENV_KEYS = ["r-enemy-hp", "r-ally-hp", "r-kill", "r-death", "r-win", "r-lose",
            "r-timeout", "r-misfire", "r-step", "frame-skip"]


def log(*a):
    print(*a, flush=True)


# ────────────────────────────────────────────────────────────────────
# 사전 점검 — 하나라도 안 되면 서버를 아예 안 띄운다
# ────────────────────────────────────────────────────────────────────
def 유령정리():
    """★ 지난번 학습기가 살아남아 있으면 죽인다.

    학습기는 start_new_session=True 로 띄운다(멈추기를 프로세스 그룹째 보내려고).
    그래서 스튜디오 서버만 죽이면 학습기는 고아가 되어 계속 돈다 — 환경 6~8개가
    CPU 를 그대로 먹고 있어서, 다음에 켰을 때 화면이 뚝뚝 끊긴다.
    실제로 20분짜리 고아가 나온 적이 있어서 넣었다.

    ★ 내 프로세스 그룹은 절대 건드리지 않는다. killpg 로 나를 죽인 사고가 있었다.
    """
    my_pgid = os.getpgid(0)
    victims = []
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,pgid=,command="], capture_output=True, text=True).stdout
    except Exception:
        return
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, pgid, cmd = parts
        # ★ 앱에서는 '이 앱이 띄운 학습기'만 정리한다. 이름만 보고 죽이면
        #   사용자가 따로 돌리던 train_sp.py 까지 죽는다.
        if TRAIN not in cmd:
            continue
        try:
            pid, pgid = int(pid), int(pgid)
        except ValueError:
            continue
        if pid == os.getpid() or pgid == my_pgid:      # 나와 한 배를 탄 것은 건드리지 않는다
            continue
        victims.append((pid, pgid))
    if not victims:
        return
    log(f"※ 지난번 학습기가 아직 돌고 있습니다 ({len(victims)}개) — 정리합니다")
    for pid, pgid in victims:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try: os.killpg(pgid, sig)
            except Exception:
                try: os.kill(pid, sig)
                except Exception: pass
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except OSError:
                break


def preflight_check():
    bad = []
    if sys.executable != PY311 and "3.11" not in sys.version:
        bad.append(f"파이썬이 {sys.version.split()[0]} 입니다. torch 가 있는 건 python3.11 뿐이에요.\n"
                   f"    이렇게 켜세요:  {PY311} {os.path.join(HERE,'studio.py')}")
    try:
        import torch  # noqa
    except Exception:
        bad.append(f"torch 가 없습니다. {PY311} -c 'import torch' 가 되는지 확인해 주세요.")
    for path, what in ((BIN, "게임 환경 bwmicro_sp"), (MPQ, "게임 데이터 폴더"),
                       (MAP, "맵 파일"), (TRAIN, "학습기 train_sp.py"), (UI, "화면 ui.html")):
        if not os.path.exists(path):
            bad.append(f"{what} 가 없습니다: {path}")
    if not os.environ.get("SC_ENCODER") and not shutil.which(FFMPEG) and not os.path.exists(FFMPEG):
        bad.append(f"ffmpeg 가 없습니다: {FFMPEG}")
    if bad:
        log("\n스튜디오를 켤 수 없습니다.\n")
        for b in bad:
            log("  · " + b)
        log("")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────
# SSE 허브 — 구독자마다 큐 하나
# ────────────────────────────────────────────────────────────────────
class Hub:
    def __init__(self):
        self.subs = []
        self.lock = threading.Lock()
        self.history = []          # 늦게 붙은 구독자에게 최근 것부터 보내준다
        self.HISTORY_MAX = 400

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            for row in self.history[-self.HISTORY_MAX:]:
                try: q.put_nowait(row)
                except queue.Full: break
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def push(self, row):
        with self.lock:
            self.history.append(row)
            if len(self.history) > self.HISTORY_MAX * 3:
                # 곡선은 솎아서 보관한다. 20분이면 만 줄이 넘는다.
                # ★ 단 upd(곡선) 행만 솎는다. 매치·세대·사건은 개수가 적고,
                #   늦게 접속한 브라우저가 대표 숫자를 못 그리면 화면이 통째로 빈다.
                thin = [r for r in self.history if r.get("t") == "upd"][::2][-self.HISTORY_MAX:]
                keep = [r for r in self.history if r.get("t") != "upd"][-self.HISTORY_MAX:]
                self.history = keep + thin
            dead = []
            for q in self.subs:
                try:
                    q.put_nowait(row)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.subs.remove(q)


HUB = Hub()


# ────────────────────────────────────────────────────────────────────
# 학습 프로세스 하나
# ────────────────────────────────────────────────────────────────────
class Runner:
    def __init__(self):
        self.proc = None
        self.run = None
        self.dir = None
        self.sliders = dict(DEFAULT_SLIDERS)
        self.state = "쉬는 중"
        self.started_at = None
        self.target = 0          # 목표 판수. 0 이면 끝없이 돈다
        self._saved = None       # 요약을 이미 남긴 런 (두 번 쓰지 않으려고)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, run_name, sliders, target=30000, init_from=""):
        if self.alive():
            raise RuntimeError("이미 돌고 있습니다. 먼저 멈춰 주세요.")
        self.sliders = dict(DEFAULT_SLIDERS)
        self.sliders.update(sliders or {})
        self.target = int(target or 0)
        self._saved = None
        run_name = "".join(c for c in (run_name or "실험") if c not in "/\\:").strip() or "실험"
        base, i = run_name, 2
        while os.path.exists(os.path.join(RUNS, run_name)):
            run_name = f"{base}_{i}"; i += 1
        self.run, self.dir = run_name, os.path.join(RUNS, run_name)
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)

        ally, allies, enemy, enemies = UNITS
        cmd = [PY311, TRAIN,
               "--ally", ally, "--allies", str(allies),
               "--enemy", enemy, "--enemies", str(enemies),
               "--max-steps", str(MAX_STEPS), "--envs", "6", "--updates", "1000000",
               "--out", self.dir]
        # ★ r-* 를 하나도 빠짐없이 명시적으로 넘긴다. 안 넘기면 C++ 기본값이 몰래 켜진다.
        for k in ENV_KEYS:
            cmd += [f"--{k}", str(self.sliders[k])]
        cmd += ["--ent", str(self.sliders["ent"]), "--gamma", str(self.sliders["gamma"])]
        if CURRICULUM: cmd += ["--curriculum", CURRICULUM]
        # ★ 이어서 배우기. 맨땅에서 승리가 한 번도 안 나오는 판은 여기서 출발선을 옮긴다.
        init_from = init_from or P.get("init_from", "")
        if init_from:
            cmd += ["--init-from", os.path.expanduser(init_from)]

        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump({"run": run_name, "preset": PRESET, "preset_name": P["이름"],
                       "target": self.target, "sliders": self.sliders, "units": UNITS,
                       "frame_skip": self.sliders["frame-skip"], "max_steps": MAX_STEPS,
                       "argv": cmd, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=1)

        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   SC_BIN=BIN, SC_EXTRA=" ".join(EXTRA))
        # ★ 학습기를 한 단계 낮은 우선순위로 돌린다. 안 그러면 CPU 경쟁 때문에
        #   관전 화면이 뚝뚝 끊긴다(관전은 sleep 으로 박자를 맞추기 때문에 밀리면 바로 티가 난다).
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True, env=env,
                                     preexec_fn=lambda: os.nice(6))
        self.state = "돌리는 중"
        self.started_at = time.time()
        log(f"돌리기 시작 — {run_name} (pid {self.proc.pid})")
        return run_name

    def _control(self, payload):
        if not self.dir:
            raise RuntimeError("아직 아무것도 안 돌고 있습니다.")
        tmp = os.path.join(self.dir, ".control.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.dir, "control.json"))

    def pause(self):
        self._control({"cmd": "pause"}); self.state = "멈춤"

    def resume(self):
        self._control({"cmd": "resume"}); self.state = "돌리는 중"

    def tune(self, sliders):
        self.sliders.update(sliders or {})
        self._control({"sliders": {k: self.sliders[k] for k in DEFAULT_SLIDERS}})

    def 요약저장(self):
        """실험이 끝나면 한 장짜리 요약을 남긴다. 나중에 목록에서 이걸 읽는다."""
        if not self.dir or self._saved == self.dir:
            return
        self._saved = self.dir
        m = os.path.join(self.dir, "metrics.jsonl")
        if not os.path.exists(m):
            return
        ups, gens = [], []
        try:
            for line in open(m):
                r = json.loads(line)
                if r.get("t") == "upd":
                    ups.append(r)
                elif r.get("t") == "gen":
                    gens.append(r)
        except Exception:
            pass
        if not ups:
            return
        wr = [u.get("wr0", 0) for u in ups]
        뒤 = ups[-10:]
        def 평균(키, idx=None):
            v = [(u[키][idx] if idx is not None else u[키]) for u in 뒤 if 키 in u]
            return sum(v) / len(v) if v else 0
        물러나기 = 0.0
        for u in 뒤:
            a = u.get("acts0") or []
            if len(a) >= 12:
                물러나기 += a[6] + a[7] + a[8] + a[11]
        물러나기 = 물러나기 / max(1, len(뒤))
        s = {
            "run": self.run, "preset": PRESET, "preset_name": P["이름"],
            "started": self.started_at and time.strftime("%m/%d %H:%M", time.localtime(self.started_at)),
            "elapsed_min": round((time.time() - (self.started_at or time.time())) / 60, 1),
            "ep": ups[-1].get("ep", 0), "target": self.target,
            "gens": len(gens),
            "wr_final": round(sum(wr[-10:]) / max(1, len(wr[-10:])), 3),
            "wr_best": round(max(wr), 3),
            "retreat_final": round(물러나기, 3),
            "ent_final": round(평균("ent0"), 2),
            "sliders": self.sliders,
        }
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        log(f"요약 저장 — {self.run}: {s['ep']:,}판 · 승률 {s['wr_final']*100:.0f}% · 물러나기 {s['retreat_final']*100:.0f}%")

    def stop(self):
        if self.alive():
            try:
                self._control({"cmd": "stop"})
            except Exception:
                pass
            for _ in range(40):
                if not self.alive():
                    break
                time.sleep(0.1)
            if self.alive():
                try: os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                except Exception: pass
        self.state = "쉬는 중"
        # 사다리 평가가 계속 돌면 멈춘 뒤에도 CPU 를 먹는다. 큐를 비운다.
        global LADDER
        if LADDER:
            try:
                LADDER.stop(wait=1.0)
                log("사다리 평가도 멈췄습니다")
            except Exception: pass
            LADDER = None
        try: self.요약저장()
        except Exception: traceback.print_exc()


RUNNER = Runner()


# ────────────────────────────────────────────────────────────────────
# metrics.jsonl 을 오프셋으로 따라가며 SSE 로 흘린다
# ────────────────────────────────────────────────────────────────────
class Tailer(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        self.off = 0
        self.path = None
        self.last = {}
        self.recent = []            # 진단용 최근 upd 행
        self.gens = []              # [(gen, ep)]
        self.last_swap = 0.0
        self.stop_flag = False
        self._stopping = False

    def run(self):
        while not self.stop_flag:
            try:
                p = os.path.join(RUNNER.dir, "metrics.jsonl") if RUNNER.dir else None
                if p != self.path:
                    self.path, self.off = p, 0
                    self.recent, self.gens = [], []
                    self.last_swap = 0.0
                    self._stopping = False
                if p and os.path.exists(p):
                    size = os.path.getsize(p)
                    if size > self.off:
                        with open(p, "r") as f:
                            f.seek(self.off)
                            chunk = f.read()
                            self.off = f.tell()
                        for line in chunk.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            self._handle(row)
            except Exception:
                traceback.print_exc()
            time.sleep(0.25)

    def _handle(self, row):
        t = row.get("t")
        if t == "upd":
            self.last = row
            # 목표 판수에 닿으면 스스로 멈춘다. 안 그러면 끝없이 돈다.
            if (RUNNER.target and row.get("ep", 0) >= RUNNER.target
                    and RUNNER.alive() and not self._stopping):
                self._stopping = True
                log(f"목표 {RUNNER.target:,}판 도달 — 멈춥니다")
                HUB.push({"t": "done", "ep": row.get("ep"), "target": RUNNER.target})

                def 멈추고_최고세대찾기():
                    RUNNER.stop()
                    # ★ 봉우리가 좁아서 마지막 체크포인트가 최고가 아닌 경우가 많다.
                    #   끝나자마자 전 세대를 훑어서 진짜 최고를 찾아둔다(2~3분).
                    _best_worker()
                threading.Thread(target=멈추고_최고세대찾기, daemon=True).start()
            self.recent.append(row)
            if len(self.recent) > 60:
                self.recent = self.recent[-60:]
        elif t == "gen":
            g, ep = row.get("gen"), row.get("ep")
            self.gens.append((g, ep))
            # 새 세대가 나오면 (1) 사다리에 붙이고 (2) 관전 왼쪽을 갈아끼운다.
            # ★ 둘 다 솎아낸다. 세대는 분당 20개 넘게 나오는데 한 세대 평가는 12초 걸린다.
            #   그대로 넣으면 큐가 영원히 안 줄고, 화면은 3초마다 선수가 바뀌어 어지럽다.
            #   체크포인트 자체는 자주 남겨도 된다(96KB짜리고, 나중에 세대 사다리 영상의 재료다).
            if LADDER and LADDER.pending() <= 2:
                try:
                    LADDER.submit_gen(g, ep)
                    row["평가"] = "붙임"
                except Exception:
                    traceback.print_exc()
            else:
                row["평가"] = "건너뜀"
            now = time.time()
            if now - self.last_swap >= SWAP_EVERY:
                self.last_swap = now
                _refresh_spectator()
        elif t == "stop":
            RUNNER.state = "쉬는 중"
        HUB.push(row)


TAILER = Tailer()


# ────────────────────────────────────────────────────────────────────
# 관전 + 사다리
# ────────────────────────────────────────────────────────────────────
SPEC = None
LADDER = None
WATCH = {"left": "current", "right": "gen0", "speed": 1.0}
_spec_lock = threading.Lock()



def _세대번호(파일, 앞글자):
    """v1947.pt → 1947.

    ★ 예전엔 파일 이름의 세 글자만 잘라 읽었다. 그래서 1000세대가 넘어가면
      v1947.pt 를 '194세대'로 읽었다. 3천 세대짜리 실험에서 세대 버튼이
      엉뚱한 체크포인트를 틀고 있었다는 뜻이다.
    """
    if not (파일.startswith(앞글자) and 파일.endswith(".pt")):
        return None
    d = 파일[len(앞글자):-3]
    return int(d) if d.isdigit() else None


def _세대목록(ck, 앞글자="v"):
    if not os.path.isdir(ck):
        return []
    out = [_세대번호(f, 앞글자) for f in os.listdir(ck)]
    return sorted(g for g in out if g is not None)


def _ckpt(side, which):
    """'current' | 'gen0' | 'prev' | 'anchor:*' | 'import:*' → 체크포인트 경로 또는 앵커 이름."""
    if which is None:
        return None
    if which.startswith("anchor:") or which.startswith("import:"):
        return which
    if not RUNNER.dir:
        return "anchor:rush"
    pre = "v" if side == "P0" else "z"
    ck = os.path.join(RUNNER.dir, "ckpt")
    if not os.path.isdir(ck):
        return "anchor:rush"
    gens = _세대목록(ck, pre)
    if not gens:
        return "anchor:rush"
    if which == "best":               # 자동으로 찾아둔 최고 세대
        b = 최고세대()
        g = b["gen"] if b and b.get("gen") in gens else gens[-1]
    elif which == "gen0":
        g = gens[0]
    elif which == "prev":
        g = gens[-2] if len(gens) > 1 else gens[0]
    elif which.startswith("gen:"):        # 특정 세대를 콕 집어서 (세대별 영상 버튼)
        want = int(which.split(":", 1)[1])
        g = min(gens, key=lambda x: abs(x - want))
    else:                       # current
        g = gens[-1]
    return os.path.join(ck, f"{pre}{g:03d}.pt")


def _refresh_spectator():
    """세대가 바뀌었을 때 관전 선수를 최신으로 갈아끼운다."""
    if SPEC is None:
        return
    try:
        with _spec_lock:
            SPEC.set_players(left=_ckpt("P0", WATCH["left"]),
                             right=_ckpt("P1", WATCH["right"]))
    except Exception as e:
        log("관전 선수 교체 실패:", e)


def start_spectator():
    global SPEC
    import spectator as SP
    SP.BIN = BIN
    SPEC = SP.Spectator(left=_ckpt("P0", WATCH["left"]), right=_ckpt("P1", WATCH["right"]),
                        units=UNITS, speed=WATCH["speed"], frame_skip=FRAME_SKIP,
                        max_steps=MAX_STEPS, w=640, h=360, seed=3001,
                        extra=EXTRA).start()
    log("관전 시작")


def start_ladder():
    global LADDER
    import ladder as L
    L.BIN = BIN                      # 판마다 바이너리가 다르다
    # workers 2 로 줄인다. 4 면 관전·학습과 CPU 를 다퉈서 화면이 끊긴다.
    LADDER = L.Ladder(RUNNER.dir, n=60, units=UNITS, frame_skip=FRAME_SKIP,
                      max_steps=MAX_STEPS, on_row=HUB.push, extra=EXTRA, workers=2)
    LADDER.start()
    log("사다리 워커 시작")


# ────────────────────────────────────────────────────────────────────
# 최고 세대 찾기
#   ★ 왜 필요한가: 봉우리가 좁다. 실측으로 19만5천판 세대가 97% 인데 20만7천판은 17%,
#     마지막 25만판은 85% 였다. "마지막 체크포인트 = 제일 잘하는 것" 이 전혀 아니다.
#     그래서 끝나면 자동으로 전 세대를 훑어서 진짜 최고를 찾아 best.json 에 박아둔다.
#   훑는 방법은 두 단계다. 넓게 24판씩 → 상위 3개만 60판씩 다시.
#     한 번에 60판씩 다 재면 3천 세대 × 60판 = 18만 판이라 하루가 걸린다.
# ────────────────────────────────────────────────────────────────────
BEST = {"state": "없음", "run": None, "progress": 0, "total": 0,
        "gen": None, "ep": None, "wr": None, "surv": None, "table": []}
_best_lock = threading.Lock()


def _best_worker(coarse=18, n1=24, n2=60):
    import ladder as L
    L.BIN = BIN
    run_dir = RUNNER.dir
    try:
        ck = os.path.join(run_dir, "ckpt")
        gens = _세대목록(ck)
        ep_of = dict(TAILER.gens)
        if not gens:
            with _best_lock:
                BEST.update(state="체크포인트가 없습니다", progress=0, total=0)
            return
        # 1단계 — 고르게 뽑아서 넓게 훑는다
        step = max(1, len(gens) // coarse)
        골라낸 = gens[::step]
        if gens[-1] not in 골라낸:
            골라낸.append(gens[-1])
        with _best_lock:
            BEST.update(state="넓게 훑는 중", run=RUNNER.run, progress=0,
                        total=len(골라낸) + 3, table=[])
        표 = []
        for i, g in enumerate(골라낸):
            표.append(_잰다(L, run_dir, g, ep_of.get(g), n1))
            with _best_lock:
                BEST.update(progress=i + 1, table=sorted(표, key=lambda r: -r["wr"])[:8])
        # 2단계 — 상위 3개만 제대로
        상위 = [r["gen"] for r in sorted(표, key=lambda r: (-r["wr"], -r["surv"]))[:3]]
        with _best_lock:
            BEST.update(state="상위 3개 다시 재는 중")
        정밀 = []
        for i, g in enumerate(상위):
            정밀.append(_잰다(L, run_dir, g, ep_of.get(g), n2))
            with _best_lock:
                BEST.update(progress=len(골라낸) + i + 1,
                            table=sorted(정밀, key=lambda r: -r["wr"]))
        승자 = max(정밀, key=lambda r: (r["wr"], r["surv"]))
        out = {"gen": 승자["gen"], "ep": 승자["ep"], "wr": 승자["wr"], "surv": 승자["surv"],
               "n": n2, "coarse": sorted(표, key=lambda r: -r["wr"]),
               "fine": sorted(정밀, key=lambda r: -r["wr"]),
               "when": time.strftime("%m/%d %H:%M")}
        with open(os.path.join(run_dir, "best.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        with _best_lock:
            BEST.update(state="끝", gen=승자["gen"], ep=승자["ep"],
                        wr=승자["wr"], surv=승자["surv"], table=out["fine"])
        log(f"최고 세대 = {승자['gen']}세대 ({승자['ep']:,}판) · 승률 {승자['wr']*100:.0f}%"
            f" · 생존 {승자['surv']:.2f}")
        HUB.push({"t": "best", **out})
    except Exception as e:
        traceback.print_exc()
        with _best_lock:
            BEST.update(state=f"실패: {e}")


def _잰다(L, run_dir, gen, ep, n):
    pol = L.load_policy(os.path.join(run_dir, "ckpt", f"v{gen:03d}.pt"))
    r = L.match(pol, L.ANCHOR_POL["stop"], n=n, units=UNITS, frame_skip=FRAME_SKIP,
                max_steps=MAX_STEPS, extra=EXTRA, workers=2)
    tot = max(1, r["n"])
    return {"gen": gen, "ep": ep or 0, "wr": round(r["win0"] / tot, 3),
            "surv": r["surv0"], "n": r["n"]}


def 최고세대(run_dir=None):
    """저장된 best.json 을 읽는다. 없으면 None."""
    d = run_dir or RUNNER.dir
    if not d:
        return None
    p = os.path.join(d, "best.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# 손코딩 대전표 (돌리기 전에 '이 판이 게임이 되는가'를 재준다)
# ────────────────────────────────────────────────────────────────────
PREFLIGHT = {"done": False, "progress": 0, "total": 16, "cells": {}, "frame_skip": FRAME_SKIP}
_pf_lock = threading.Lock()


def _preflight_worker(n=24):
    import ladder as L
    names = ["stop", "rush", "kite", "spread"]
    cells = {}
    prog = 0
    for a in names:
        cells[a] = {}
        for b in names:
            try:
                r = L.match(L.ANCHOR_POL[a], L.ANCHOR_POL[b], n=n, units=UNITS,
                            frame_skip=FRAME_SKIP, max_steps=MAX_STEPS, extra=EXTRA)
                cells[a][b] = {"win": r["win0"], "lose": r["win1"], "to": r["timeout"]}
            except Exception as e:
                cells[a][b] = {"win": 0, "lose": 0, "to": 0, "err": str(e)[:80]}
            prog += 1
            with _pf_lock:
                PREFLIGHT.update(progress=prog, cells=cells)
    with _pf_lock:
        PREFLIGHT.update(done=True, progress=16, cells=cells)
    HUB.push({"t": "preflight", "cells": cells})


# ────────────────────────────────────────────────────────────────────
# 클립
# ────────────────────────────────────────────────────────────────────
def save_clip():
    if not (SPEC and RUNNER.dir):
        raise RuntimeError("관전 중이 아닙니다.")
    info = SPEC.info()
    row = {"t": "clip", "id": int(time.time()), "ep": info.get("ep"),
           "left": info.get("left_name"), "right": info.get("right_name"),
           "step": info.get("step"), "verdict": info.get("verdict"),
           "when": time.strftime("%H:%M:%S")}
    with open(os.path.join(RUNNER.dir, "clips.jsonl"), "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    HUB.push(row)
    return row


# ────────────────────────────────────────────────────────────────────
# HTTP
# ────────────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # ── 도우미 ──
    def _send(self, code, ctype, body, extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode())
        except Exception:
            return {}

    # ── GET ──
    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/watch", "/broadcast"):
                with open(UI, "rb") as f:              # 매 요청 디스크에서 (핫에디트)
                    return self._send(200, "text/html; charset=utf-8", f.read())
            if path == "/live.mjpg":
                return self.mjpg()
            if path == "/api/stream":
                return self.sse()
            if path == "/api/status":
                return self._json(self.status())
            if path == "/api/runs":
                return self._json(self.runs())
            if path == "/api/gens":
                return self._json(self.gens())
            if path == "/api/best":
                with _best_lock:
                    return self._json({**BEST, "saved": 최고세대()})
            if path == "/api/preflight":
                with _pf_lock:
                    return self._json(dict(PREFLIGHT))
            self._send(404, "text/plain; charset=utf-8", "없는 주소입니다")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            try: self._json({"error": "서버 안에서 문제가 생겼습니다"}, 500)
            except Exception: pass

    # ── POST ──
    def do_POST(self):
        path = self.path.split("?")[0]
        b = self._body()
        try:
            if path == "/api/start":
                name = RUNNER.start(b.get("run"), b.get("sliders"), b.get("target", 30000),
                                    b.get("init_from", ""))
                start_ladder()
                _refresh_spectator()
                return self._json({"ok": True, "run": name})
            if path == "/api/pause":
                RUNNER.pause(); return self._json({"ok": True})
            if path == "/api/resume":
                RUNNER.resume(); return self._json({"ok": True})
            if path == "/api/stop":
                RUNNER.stop(); return self._json({"ok": True})
            if path == "/api/tune":
                RUNNER.tune(b.get("sliders")); return self._json({"ok": True})
            if path == "/api/diagnose":
                return self._json(self.diagnose(b))
            if path == "/api/preflight":
                with _pf_lock:
                    if not PREFLIGHT["done"] and PREFLIGHT["progress"] > 0:
                        return self._json(dict(PREFLIGHT))
                    PREFLIGHT.update(done=False, progress=0, cells={})
                threading.Thread(target=_preflight_worker, daemon=True).start()
                return self._json({"done": False, "progress": 0, "total": 16})
            if path == "/api/watch":
                WATCH.update({k: b[k] for k in ("left", "right") if k in b})
                if "speed" in b:
                    WATCH["speed"] = max(0.1, min(8.0, float(b["speed"])))
                    if SPEC: SPEC.set_speed(WATCH["speed"])
                # ★ 사람이 고른 것이므로 판을 처음부터 다시 튼다
                if SPEC:
                    with _spec_lock:
                        SPEC.restart(left=_ckpt("P0", WATCH["left"]),
                                     right=_ckpt("P1", WATCH["right"]))
                return self._json({"ok": True, "watch": WATCH})
            if path == "/api/best":
                with _best_lock:
                    if BEST["state"] in ("넓게 훑는 중", "상위 3개 다시 재는 중"):
                        return self._json(dict(BEST))
                    BEST.update(state="넓게 훑는 중", progress=0, total=0, table=[],
                                gen=None, ep=None, wr=None, surv=None)
                threading.Thread(target=_best_worker, daemon=True).start()
                return self._json(dict(BEST))
            if path == "/api/clip":
                return self._json(save_clip())
            if path == "/api/render_clips":
                return self._json({"ok": False, "msg": "영상 뽑기는 아직 준비 중입니다."})
            self._json({"error": "없는 주소입니다"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e)}, 400)

    # ── 각 엔드포인트 ──
    def status(self):
        st = {
            "state": RUNNER.state, "run": RUNNER.run,
            "ep": TAILER.last.get("ep", 0),
            "gen": (TAILER.gens[-1][0] if TAILER.gens else 0),
            "elo0": (LADDER.fit.get("P0") if LADDER else None),
            "elo1": (LADDER.fit.get("P1") if LADDER else None),
            "pending": (LADDER.pending() if LADDER else 0),
            "speed": WATCH["speed"], "watch": dict(WATCH),
            "sliders": RUNNER.sliders,
            "preset": PRESET, "preset_name": P["이름"], "preset_desc": P["설명"],
            "units": {"ally": UNITS[0], "allies": UNITS[1],
                      "enemy": UNITS[2], "enemies": UNITS[3],
                      "frame_skip": FRAME_SKIP, "max_steps": MAX_STEPS},
            "target": RUNNER.target,
            "elapsed": (time.time() - RUNNER.started_at) if RUNNER.started_at else 0,
        }
        if SPEC:
            try:
                st["watching"] = SPEC.info()
                st["restarts"] = SPEC.restarts
            except Exception:
                st["watching"] = {}
        return st

    def gens(self):
        """저장된 세대 목록. [{gen, ep}] — 세대별로 그때의 플레이를 다시 보려고 쓴다."""
        if not (RUNNER.dir and TAILER.gens):
            return []
        ck = os.path.join(RUNNER.dir, "ckpt")
        have = set(_세대목록(ck))
        b = 최고세대() or {}
        return {"list": [{"gen": g, "ep": e} for g, e in TAILER.gens if g in have],
                "best": {k: b.get(k) for k in ("gen", "ep", "wr", "surv")} if b else None}

    def runs(self):
        out = []
        if os.path.isdir(RUNS):
            for name in sorted(os.listdir(RUNS), reverse=True):
                d = os.path.join(RUNS, name)
                cfg = os.path.join(d, "config.json")
                if not os.path.isfile(cfg):
                    continue
                try:
                    c = json.load(open(cfg))
                except Exception:
                    continue
                ep, gen = 0, 0
                m = os.path.join(d, "metrics.jsonl")
                if os.path.exists(m):
                    try:
                        for line in open(m):
                            r = json.loads(line)
                            ep = r.get("ep", ep)
                            if r.get("t") == "gen":
                                gen = r.get("gen", gen)
                    except Exception:
                        pass
                summ = {}
                sp = os.path.join(d, "summary.json")
                if os.path.isfile(sp):
                    try: summ = json.load(open(sp))
                    except Exception: pass
                out.append({"run": name, "ep": summ.get("ep", ep), "gen": summ.get("gens", gen),
                            "started": c.get("started"), "preset_name": c.get("preset_name"),
                            "target": c.get("target"), "sliders": c.get("sliders"),
                            "wr_final": summ.get("wr_final"), "wr_best": summ.get("wr_best"),
                            "retreat_final": summ.get("retreat_final"),
                            "elapsed_min": summ.get("elapsed_min"),
                            "running": (RUNNER.run == name and RUNNER.alive())})
        return out

    def diagnose(self, b):
        import diagnose as D
        sliders = dict(DEFAULT_SLIDERS); sliders.update(b.get("sliders") or {})
        setup = {"allies": UNITS[1], "enemies": UNITS[3], "max_steps": MAX_STEPS,
                 "ally": UNITS[0], "enemy": UNITS[2]}
        setup.update(b.get("setup") or {})
        out = D.diagnose_before(sliders, setup)
        if b.get("during") and TAILER.recent:
            rows = list(TAILER.recent)
            rows += [{"t": "gen", "gen": g, "ep": e} for g, e in TAILER.gens[-3:]]
            lad = LADDER.rows if LADDER else []
            out += D.diagnose_during(rows, lad, frame_skip=int(sliders["frame-skip"]))
        return {"items": out, "totals": D.totals(sliders, setup)}

    def sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = HUB.subscribe()
        try:
            while True:
                try:
                    row = q.get(timeout=10)
                    payload = "data: " + json.dumps(row, ensure_ascii=False) + "\n\n"
                except queue.Empty:
                    payload = ": keep-alive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    def mjpg(self):
        if SPEC is None:
            return self._send(503, "text/plain; charset=utf-8", "관전이 아직 안 켜졌습니다")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        ver = 0
        try:
            while True:
                ver, jpg = SPEC.wait_frame(ver, timeout=5.0)
                if jpg is None:
                    continue
                self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--no-spectator", action="store_true")
    a = ap.parse_args()

    preflight_check()
    import torch
    import diagnose
    diagnose._selftest()          # 수식이 실패기 문서와 어긋나면 여기서 죽는다

    유령정리()
    os.makedirs(RUNS, exist_ok=True)
    log(f"스타 AI 스튜디오 · python {sys.version.split()[0]} · torch {torch.__version__} · ffmpeg OK")
    log(f"판: {P['이름']}")
    log(f"    {P['설명']}")
    log(f"    판단주기 {FRAME_SKIP}프레임 · 한 판 최대 {MAX_STEPS}스텝 · {os.path.basename(BIN)}")

    TAILER.start()
    if not a.no_spectator:
        try:
            start_spectator()
        except Exception:
            traceback.print_exc()
            log("관전을 못 켰습니다. 나머지는 그대로 돕니다.")

    srv = Server(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    log(f"{url}  ← 브라우저를 엽니다")
    log("(이 창은 닫지 마세요. 끄려면 여기서 Ctrl-C)")
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("\n끕니다…")
    finally:
        TAILER.stop_flag = True
        RUNNER.stop()
        if LADDER:
            try: LADDER.stop()
            except Exception: pass
        if SPEC:
            try: SPEC.close()
            except Exception: pass
        log("끝났습니다.")


if __name__ == "__main__":
    main()
