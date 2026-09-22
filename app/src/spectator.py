"""관전 중계 — 학습이 도는 중에 게임 화면을 브라우저로 실시간으로 흘려보냅니다.

파이프라인 (계약서 그대로):

    bwmicro_sp --selfplay --render-to <절대경로 FIFO> --w 640 --h 360
       │  RGBA 프레임을 FIFO 에 쏟습니다
       ├─ 스레드A: FIFO → ffmpeg.stdin      ★ 없으면 교착합니다
       │           (FIFO 버퍼 64KB < 프레임 921KB)
       ├─ ffmpeg  -f rawvideo -pix_fmt rgba -s 640x360 -i pipe:0 -f mjpeg -q:v 5 pipe:1
       ├─ 스레드B: 연속 JPEG 를 ffd8ff…ffd9 로 잘라 '프레임 슬롯'에 넣습니다
       └─ 스레드C: 프로토콜을 읽고 양쪽 정책을 굴려 A/B 행동을 보냅니다.
                   브루드워 네이티브 23.81fps 로 페이싱합니다 (speed 배속)

프레임 슬롯은 최신 한 장만 들고 있습니다. 브라우저가 느리거나 끊겨도 슬롯이 덮어써질 뿐이라
파이프라인이 막히지 않고, C++ 이 SIGPIPE 로 죽지도 않습니다.

혼자 돌려보기:
    /opt/homebrew/bin/python3.11 spectator.py --demo      # http://127.0.0.1:8899
    /opt/homebrew/bin/python3.11 spectator.py --selftest  # 실측 6종
"""
import glob
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

# ── 공통 상수 (계약서) ────────────────────────────────────────────────
PY311 = os.environ.get("SC_PY", "/opt/homebrew/bin/python3.11")
FFMPEG = os.environ.get("SC_FFMPEG", "/opt/homebrew/bin/ffmpeg")
ENCODER = os.environ.get("SC_ENCODER")   # 앱에서는 ffmpeg 대신 파이썬 JPEG 인코더를 쓴다
BIN = os.environ.get("SC_BIN", os.path.expanduser("~/scmicro/tools/bwmicro_sp"))
MPQ = os.environ.get("SC_MPQ", os.path.expanduser("~/scmicro/data/mpq"))
MAP = os.environ.get("SC_MAP", os.path.expanduser("~/scmicro/maps/Weave_v1.scx"))
OBS_DIM, N_ACT = 39, 12
ALWAYS = ("--selfplay", "--strict-fire", "--box-range")   # UI 에 노출 금지
NATIVE_FPS = 1000.0 / 42.0        # 브루드워 네이티브 23.81fps
UNITS_DEFAULT = ("vulture", 1, "zergling", 6)             # 확정된 판
VERDICT = {0: "", 1: "왼쪽 승", 2: "오른쪽 승", 3: "시간초과"}

_DEBUG = os.environ.get("SPECTATOR_DEBUG") == "1"


# ── 손코딩 정책 4종 (train/preflight.py 의 POL 을 그대로 옮겼습니다) ──
import math

DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def dir_action(dx, dy):
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


def p_stop(o, alive):      # 가만히 = 제자리 자동공격
    return [0 for _ in range(len(o))]


def p_rush(o, alive):      # 사거리 안이면 쏘고 아니면 붙습니다
    a = []
    for i in range(len(o)):
        if not alive[i]:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        a.append(9 if inr else dir_action(ex, ey))
    return a


def p_kite(o, alive):      # 쿨다운 중이면 물러납니다 (무빙샷)
    a = []
    for i in range(len(o)):
        if not alive[i]:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        cd = o[i][1]
        if inr and cd < 0.1:
            a.append(9)
        elif inr:
            a.append(11)
        else:
            a.append(dir_action(ex, ey))
    return a


def p_spread(o, alive):    # 벌려서 접근
    a = []
    for i in range(len(o)):
        if not alive[i]:
            a.append(0); continue
        ex, ey, inr = o[i][5] * 256.0, o[i][6] * 256.0, o[i][10] > 0.5
        if inr:
            a.append(9)
        else:
            a.append(dir_action(ex, ey + (60 if i % 2 else -60)))
    return a


ANCHOR = {"stop": ("정지", p_stop), "rush": ("돌진", p_rush),
          "kite": ("무빙샷", p_kite), "spread": ("벌리기", p_spread)}
for _v in list(ANCHOR.values()):          # "anchor:무빙샷" 처럼 한국어로도 부를 수 있게
    ANCHOR[_v[0]] = _v


# ── 체크포인트 정책 ──────────────────────────────────────────────────
_POLICY_CLS = None


def _policy_class():
    """train/ppo_micro.py 의 Policy 와 같은 그물입니다. 키가 맞아야 불러와집니다."""
    global _POLICY_CLS
    if _POLICY_CLS is None:
        import torch.nn as nn

        class Policy(nn.Module):
            def __init__(self, obs=OBS_DIM, hid=128, act=N_ACT):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                          nn.Linear(hid, hid), nn.Tanh())
                self.pi = nn.Linear(hid, act)
                self.v = nn.Linear(hid, 1)

            def forward(self, x):
                h = self.body(x)
                return self.pi(h), self.v(h).squeeze(-1)

        _POLICY_CLS = Policy
    return _POLICY_CLS


def resolve_player(name):
    """left/right 문자열 하나를 (행동함수, 화면이름) 으로 바꿉니다.

    "anchor:kite" / "anchor:무빙샷"  → 손코딩 정책
    "import:vult1"                   → runs/vult1 의 마지막 세대
    그 밖의 문자열                    → 체크포인트 파일 경로
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"선수 이름이 이상합니다: {name!r}")
    if name.startswith("anchor:"):
        key = name.split(":", 1)[1]
        if key not in ANCHOR:
            raise ValueError(f"모르는 앵커입니다: {name}")
        ko, fn = ANCHOR[key]
        return fn, ko
    path = name
    if name.startswith("import:"):
        run = name.split(":", 1)[1]
        cands = sorted(glob.glob(os.path.join(os.environ.get("SC_OLD_RUNS", os.path.expanduser("~/scmicro/runs")), run, "ckpt", "*.pt")))
        if not cands:
            raise ValueError(f"가져올 체크포인트가 없습니다: {name}")
        path = cands[-1]
    if not os.path.exists(path):
        raise ValueError(f"체크포인트 파일이 없습니다: {path}")

    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net = _policy_class()()
    net.load_state_dict(sd)
    net.eval()
    label = os.path.basename(path).removesuffix(".pt")
    if isinstance(ck, dict) and "gen" in ck:
        label = f"{label}(세대 {ck['gen']})"

    def fn(o, alive):
        with torch.no_grad():
            lg, _ = net(torch.as_tensor(np.asarray(o, dtype=np.float32)))
            a = torch.distributions.Categorical(logits=lg).sample().tolist()
        return [int(a[i]) if alive[i] else 0 for i in range(len(a))]

    return fn, label


# ── 프레임 슬롯 ──────────────────────────────────────────────────────
class _Slot:
    """최신 JPEG 한 장 + 판번호. 이게 이 파일의 심장입니다."""

    def __init__(self):
        self.cv = threading.Condition()
        self.buf = None
        self.ver = 0

    def put(self, b):
        with self.cv:
            self.buf, self.ver = b, self.ver + 1
            self.cv.notify_all()

    def get(self):
        with self.cv:
            return self.buf

    def wait(self, last_ver, timeout=5.0):
        with self.cv:
            if self.ver == last_ver:
                self.cv.wait(timeout)
            return self.ver, self.buf


def _read_exact(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b


def _write_all(fd, data):
    off = 0
    while off < len(data):
        off += os.write(fd, data[off:])


# ── 관전기 ───────────────────────────────────────────────────────────
class Spectator:
    """게임 한 판을 계속 굴리면서 JPEG 를 슬롯에 채웁니다. 판이 끝나면 바로 다음 판."""

    def __init__(self, left="anchor:kite", right="anchor:rush",
                 units=UNITS_DEFAULT, speed=1.0,
                 frame_skip=4, max_steps=200, w=640, h=360, seed=3001, extra=()):
        if isinstance(units, dict):
            units = (units.get("ally", "vulture"), units.get("allies", 1),
                     units.get("enemy", "zergling"), units.get("enemies", 6))
        self.ally, self.allies, self.enemy, self.enemies = units
        self.speed = float(speed)
        self.frame_skip, self.max_steps = int(frame_skip), int(max_steps)
        self.w, self.h, self.seed = int(w), int(h), int(seed)
        self.extra = list(extra)

        self.slot = _Slot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pol0, self._name0 = resolve_player(left)
        self._pol1, self._name1 = resolve_player(right)
        self._info = {"step": 0, "alive0": 0, "alive1": 0, "ep": 0,
                      "left_name": self._name0, "right_name": self._name1, "verdict": ""}
        self.restarts = 0
        self.frames = 0        # 슬롯에 넣은 총 프레임 수
        self.frame_bytes = 0   # 그 총 바이트
        self.last_error = ""
        self._child = None     # (bwmicro, ffmpeg, 임시디렉터리, rfd, keeper, 정지신호, 스레드A)
        self._sup = threading.Thread(target=self._supervise, daemon=True)

    # ── 바깥에서 쓰는 것들 ────────────────────────────────────────
    def start(self):
        self._sup.start()
        return self

    def frame(self):
        """최신 JPEG bytes (아직 없으면 None)."""
        return self.slot.get()

    def wait_frame(self, last_ver=0, timeout=5.0):
        """새 프레임이 올 때까지 막습니다. (판번호, JPEG) 을 돌려줍니다."""
        return self.slot.wait(last_ver, timeout)

    def info(self):
        with self._lock:
            d = dict(self._info)
        d["restarts"] = self.restarts
        d["speed"] = self.speed
        return d

    def set_players(self, left=None, right=None):
        """새 세대가 나오면 갈아끼웁니다. 판 도중에 불러도 안전합니다."""
        new0 = resolve_player(left) if left is not None else None
        new1 = resolve_player(right) if right is not None else None   # 먼저 다 읽고 나서 바꿉니다
        with self._lock:
            if new0:
                self._pol0, self._name0 = new0
                self._info["left_name"] = self._name0
            if new1:
                self._pol1, self._name1 = new1
                self._info["right_name"] = self._name1

    def restart(self, left=None, right=None):
        """선수를 바꾸고 ★판을 처음부터 다시 시작합니다.★

        set_players 는 돌아가던 판 한가운데서 정책만 갈아끼웁니다. 그래서 세대 버튼을
        누르면 '보던 판의 꼬리'만 잠깐 보이고 끝나 버립니다. 처음부터 보려면
        자식 프로세스를 내리고 감시 스레드가 새로 세우게 해야 합니다.
        """
        self.set_players(left, right)
        self._deliberate = True
        self._kill_child()

    def set_speed(self, speed):
        self.speed = max(0.1, min(8.0, float(speed)))

    def close(self):
        self._stop.set()
        self._kill_child()
        if self._sup.is_alive():
            self._sup.join(timeout=8)

    # ── 안쪽 ──────────────────────────────────────────────────────
    def _supervise(self):
        """자식이 죽으면 0.5초 쉬었다 다시 세웁니다."""
        first = True
        while not self._stop.is_set():
            if not first:
                # 내가 일부러 내린 것(세대 갈아끼우기)은 사고가 아니므로 안 센다
                if getattr(self, "_deliberate", False):
                    self._deliberate = False
                else:
                    self.restarts += 1
            first = False
            try:
                self._run_once()
            except Exception as e:              # 어떤 사고든 재기동으로 넘깁니다
                self.last_error = f"{type(e).__name__}: {e}"
                if _DEBUG:
                    import traceback; traceback.print_exc()
            finally:
                self._kill_child()
            self._stop.wait(0.5)

    def _run_once(self):
        tmp = tempfile.mkdtemp(prefix="scspec_")
        fifo = os.path.join(tmp, "frames.rgba")      # ★ 반드시 절대경로
        os.mkfifo(fifo)
        # 읽는 쪽을 먼저 엽니다. O_NONBLOCK 이라 안 막히고, 쓰는 쪽(keeper)을 하나
        # 물고 있어야 bwmicro 가 아직 안 붙었을 때 read 가 EOF 로 오해되지 않습니다.
        rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        keeper = os.open(fifo, os.O_WRONLY)
        child_stop = threading.Event()

        errout = None if _DEBUG else subprocess.DEVNULL
        ff_cmd = ([PY311, ENCODER, str(self.w), str(self.h)] if ENCODER else
                  [FFMPEG, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{self.w}x{self.h}", "-framerate", f"{NATIVE_FPS:.4f}", "-i", "pipe:0",
                   "-an", "-f", "mjpeg", "-q:v", "5", "-fps_mode", "passthrough", "pipe:1"])
        ff = subprocess.Popen(
            ff_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errout, bufsize=0)

        cmd = [BIN, "--data", MPQ, "--map", MAP, *ALWAYS,
               "--ally", self.ally, "--enemy", self.enemy,
               "--allies", str(self.allies), "--enemies", str(self.enemies),
               "--frame-skip", str(self.frame_skip), "--max-steps", str(self.max_steps),
               "--episodes", "1000000", "--seed", str(self.seed),
               "--w", str(self.w), "--h", str(self.h), "--render-to", fifo,
               "--realtime",                      # ★ 프레임을 고르게 흘려보낸다
               *self.extra]
        bw = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=errout, bufsize=0, cwd=MPQ)
        ta = threading.Thread(target=self._pump_fifo, args=(rfd, ff, child_stop), daemon=True)
        tb = threading.Thread(target=self._cut_jpeg, args=(ff, child_stop), daemon=True)
        self._child = (bw, ff, tmp, rfd, keeper, child_stop, ta)
        ta.start(); tb.start()
        try:
            self._play(bw)                        # 스레드C = 이 스레드
        finally:
            child_stop.set()

    def _pump_fifo(self, rfd, ff, child_stop):
        """스레드A — FIFO 를 계속 비웁니다. 이게 없으면 64KB 에서 C++ 이 멈춥니다."""
        try:
            while not child_stop.is_set() and not self._stop.is_set():
                r, _, _ = select.select([rfd], [], [], 0.2)
                if not r:
                    continue
                data = os.read(rfd, 1 << 18)
                if data:
                    _write_all(ff.stdin.fileno(), data)
        except Exception as e:
            self.last_error = f"스레드A: {e}"
        finally:
            try: ff.stdin.close()
            except Exception: pass

    def _cut_jpeg(self, ff, child_stop):
        """스레드B — 붙어서 나오는 JPEG 를 ffd8ff…ffd9 로 잘라 슬롯에 넣습니다."""
        buf = bytearray()
        try:
            while not child_stop.is_set() and not self._stop.is_set():
                chunk = ff.stdout.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                while True:
                    i = buf.find(b"\xff\xd8\xff")
                    if i < 0:
                        del buf[:max(0, len(buf) - 2)]
                        break
                    j = buf.find(b"\xff\xd9", i + 3)
                    if j < 0:
                        del buf[:i]
                        break
                    jpg = bytes(buf[i:j + 2])
                    del buf[:j + 2]
                    self.frames += 1
                    self.frame_bytes += len(jpg)
                    self.slot.put(jpg)
        except Exception as e:
            self.last_error = f"스레드B: {e}"

    def _play(self, bw):
        """스레드C — 프로토콜을 읽고 양쪽 행동을 보냅니다. 네이티브 속도로 페이싱."""
        out, inp = bw.stdout, bw.stdin
        next_t = time.monotonic()
        while not self._stop.is_set():
            tag = out.read(1)
            if not tag:
                raise RuntimeError("bwmicro 가 끊겼습니다")
            if tag == b"E":
                term, done, surv, mf = struct.unpack("<fBBf", _read_exact(out, 10))
                if _read_exact(out, 1) != b"F":     # 'E' 다음엔 반드시 'F'
                    raise RuntimeError("프로토콜: 'F' 가 안 왔습니다")
                _read_exact(out, 10)
                with self._lock:
                    self._info["verdict"] = VERDICT.get(done, "")
                next_t = time.monotonic()           # 다음 판 스폰 시간은 빚으로 안 답니다
                continue
            if tag != b"O":
                raise RuntimeError(f"프로토콜: 모르는 태그 {tag!r}")
            done, o0, al0, ep, st = self._read_obs(out)
            if _read_exact(out, 1) != b"Q":
                raise RuntimeError("프로토콜: 'Q' 가 안 왔습니다")
            _, o1, al1, _, _ = self._read_obs(out)
            with self._lock:
                self._info.update(step=st, ep=ep, alive0=int(sum(al0)), alive1=int(sum(al1)))
            if done:
                continue
            with self._lock:
                pol0, pol1 = self._pol0, self._pol1   # 판 도중 교체를 여기서 흡수합니다
            a0 = bytes(int(x) % N_ACT for x in pol0(o0, al0))
            a1 = bytes(int(x) % N_ACT for x in pol1(o1, al1))

            per_step = self.frame_skip / NATIVE_FPS / max(0.1, self.speed)
            next_t += per_step
            gap = next_t - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            elif gap < -0.5:
                next_t = time.monotonic()             # 너무 밀리면 따라잡기를 포기합니다
            _write_all(inp.fileno(), b"A" + a0)
            _write_all(inp.fileno(), b"B" + a1)

    def _read_obs(self, out):
        ep, st, rew, done, nm, nz = struct.unpack("<iifBBB", _read_exact(out, 15))
        o = np.frombuffer(_read_exact(out, 4 * nm * OBS_DIM), dtype=np.float32)
        return done, o.reshape(nm, OBS_DIM).copy(), _read_exact(out, nm), ep, st

    def _kill_child(self):
        c, self._child = self._child, None
        if not c:
            return
        bw, ff, tmp, rfd, keeper, child_stop, ta = c
        child_stop.set()
        for p in (bw, ff):
            try:
                p.kill(); p.wait(timeout=3)
            except Exception:
                pass
        ta.join(timeout=1.0)          # select 에서 빠져나온 뒤에 fd 를 닫습니다
        for fd in (keeper, rfd):
            try: os.close(fd)
            except Exception: pass
        shutil.rmtree(tmp, ignore_errors=True)


# ── 단독 실행: 표준 라이브러리 http.server 로 :8899 에 내보냅니다 ─────
PAGE = b"""<!doctype html><meta charset=utf-8><title>\xea\xb4\x80\xec\xa0\x84</title>
<body style="margin:0;background:#0b0e13;color:#e6e9ef;font:14px -apple-system,sans-serif">
<div style="padding:10px">
<img src="/live.mjpg" style="width:640px;image-rendering:pixelated;border:1px solid #333">
<pre id=t style="color:#8ab4ff"></pre></div>
<script>setInterval(async()=>{t.textContent=JSON.stringify(await(await fetch('/api/status')).json(),null,1)},500)</script>
"""


def serve(spec, port=8899):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/live.mjpg"):
                return self._mjpg()
            if self.path.startswith("/api/status"):
                b = json.dumps(spec.info(), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)

        def _mjpg(self):
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            ver, misses = 0, 0
            try:
                while misses < 6:
                    ver, jpg = spec.wait_frame(ver, timeout=5.0)
                    if jpg is None:
                        misses += 1; continue
                    misses = 0
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass          # 브라우저가 끊은 것뿐입니다. 슬롯은 그대로 돕니다.

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _demo(args):
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # kill 로 꺼도 뒷정리를 합니다
    spec = Spectator(args.left, args.right, speed=args.speed).start()
    srv = serve(spec, args.port)
    print(f"관전 중계를 켰습니다 → http://127.0.0.1:{args.port}/   (/live.mjpg)")
    print(f"  왼쪽 {spec.info()['left_name']}  vs  오른쪽 {spec.info()['right_name']}"
          f"   배속 {args.speed}")
    try:
        while True:
            time.sleep(2)
            i = spec.info()
            print(f"  판{i['ep']:>4} 스텝{i['step']:>4} "
                  f"왼쪽{i['alive0']} 오른쪽{i['alive1']} "
                  f"프레임 {spec.frames} 재기동 {i['restarts']} {i['verdict']}")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        srv.shutdown(); spec.close()
        print("멈췄습니다.")


def _selftest(port=8899):
    """실제로 돌려서 재는 검사 6종."""
    ok = True

    def say(n, t):
        print(f"\n[{n}] {t}")

    spec = Spectator("anchor:kite", "anchor:rush", speed=1.0).start()
    srv = serve(spec, port)
    try:
        say(1, "첫 프레임이 오는가")
        t0 = time.monotonic()
        v, jpg = spec.wait_frame(0, timeout=30.0)
        print(f"    {time.monotonic()-t0:.2f}초 만에 첫 JPEG {len(jpg) if jpg else 0} 바이트")
        ok &= jpg is not None

        say(2, "초당 프레임 수와 평균 크기 (10초 실측)")
        f0, b0, t0 = spec.frames, spec.frame_bytes, time.monotonic()
        time.sleep(10.0)
        df, db, dt = spec.frames - f0, spec.frame_bytes - b0, time.monotonic() - t0
        print(f"    {df} 프레임 / {dt:.1f}초 = {df/dt:.2f} fps "
              f"(네이티브 {NATIVE_FPS:.2f}), 평균 {db/max(1,df)/1024:.1f} KB")
        ok &= df / dt > 15

        say(3, "브라우저를 두 번 끊었다 붙여도 파이프라인이 사는가")
        for k in range(3):
            r = subprocess.run(["curl", "-s", "--max-time", "3",
                                f"http://127.0.0.1:{port}/live.mjpg"], capture_output=True)
            print(f"    {k+1}회차 curl 3초 → {len(r.stdout)} 바이트, "
                  f"이후 info={spec.info()['step']}스텝")
            ok &= len(r.stdout) > 100000

        say(4, "판 도중 set_players 로 갈아끼워도 안 죽는가")
        before = spec.info()
        spec.set_players(left="anchor:spread")
        time.sleep(2.0)
        after = spec.info()
        print(f"    {before['left_name']} → {after['left_name']}, "
              f"판 {before['ep']}→{after['ep']} 스텝 {before['step']}→{after['step']}")
        v2, _ = spec.wait_frame(spec.slot.ver, timeout=5.0)
        print(f"    교체 뒤에도 프레임이 옵니다 (누적 {spec.frames}장)")
        ok &= after["left_name"] == "벌리기"

        say(5, "자식이 죽으면 재기동하는가")
        r0, f0 = spec.restarts, spec.frames
        while spec._child is None:
            time.sleep(0.1)
        spec._child[0].kill()
        for _ in range(60):
            time.sleep(0.5)
            if spec.restarts > r0 and spec.frames > f0 + 20:
                break
        print(f"    재기동 {r0}→{spec.restarts}, 프레임 {f0}→{spec.frames}")
        ok &= spec.restarts > r0 and spec.frames > f0

        say(6, "판이 끝나면 다음 판으로 넘어가는가")
        e0 = spec.info()["ep"]
        for _ in range(240):
            time.sleep(0.5)
            if spec.info()["ep"] > e0:
                break
        i = spec.info()
        print(f"    판 {e0} → {i['ep']}, 직전 판정 '{i['verdict']}'")
        ok &= i["ep"] > e0
    finally:
        srv.shutdown()
        spec.close()
    print("\n판정:", "★ 전부 통과" if ok else "✗ 실패한 항목이 있습니다")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="관전 중계")
    ap.add_argument("--demo", action="store_true", help="손코딩 무빙샷 vs 돌진으로 중계")
    ap.add_argument("--selftest", action="store_true", help="실측 검사 6종")
    ap.add_argument("--left", default="anchor:kite")
    ap.add_argument("--right", default="anchor:rush")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest(a.port))
    _demo(a)                     # --demo 가 기본입니다
