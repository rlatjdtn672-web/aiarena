"""스타 AI 스튜디오 — 맥 앱 진입점.

하는 일은 셋이다.
  1. 게임 파일이 없으면 첫 화면에서 블리자드 공식 AI 연구용 패키지를 받게 한다 (약관 동의 필수)
  2. 스튜디오 서버를 띄우고, 앱 창 안에 그 화면을 연다
  3. 창을 닫으면 서버와 학습기를 전부 정리한다 (안 그러면 CPU 를 계속 먹는다)
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile

import webview

HERE = os.path.dirname(os.path.abspath(__file__))                 # Resources/app
RES = os.environ.get("SC_HOME") or os.path.dirname(HERE)          # Resources
DATA = os.path.expanduser("~/Library/Application Support/StarAIStudio")
MPQ_DIR = os.path.join(DATA, "mpq")
RUNS = os.path.join(DATA, "runs")
LOGS = os.path.join(DATA, "logs")

PKG_URL = "http://download.blizzard.com/pub/ai/MPQ_AND_EULA.zip"
PKG_PASS = "Iagreetotheeula"          # 블리자드 패키지의 압축 암호 = 약관 동의 문구
NEED = ("StarDat.mpq", "BrooDat.mpq", "patch_rt.mpq")

for d in (DATA, RUNS, LOGS):
    os.makedirs(d, exist_ok=True)


def have_mpq(d):
    if not os.path.isdir(d):
        return False
    names = {n.lower() for n in os.listdir(d)}
    return all(n.lower() in names for n in NEED)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def studio_env():
    env = dict(os.environ)
    env.update({
        "SC_PY": sys.executable,
        "SC_BIN": os.path.join(RES, "engine", "bwmicro_sp"),
        "SC_MPQ": MPQ_DIR,
        "SC_MAP": os.path.join(RES, "maps", "Weave_v1.scx"),
        "SC_RUNS": RUNS,
        "SC_OLD_RUNS": os.path.join(DATA, "old_runs"),
        "SC_ENCODER": os.path.join(HERE, "rgba2mjpeg.py"),
        "SC_PRESET": "vulture",
        # 엔진이 쓰는 SDL 라이브러리는 앱 안에 있다
        "DYLD_LIBRARY_PATH": os.path.join(RES, "engine", "lib"),
        # 소리는 안 쓴다. 맥 오디오 서비스가 삐걱대면 엔진이 여기서 멈춘 적이 있다
        "SDL_AUDIODRIVER": "dummy",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    })
    return env


class Api:
    """첫 화면(onboarding.html)의 버튼들이 부르는 함수들."""

    def __init__(self):
        self.window = None
        self.state = {"stage": "idle", "pct": 0, "msg": "", "eula": ""}
        self._tmp = None
        self.studio = None
        self.url = None

    # ── 상태 ─────────────────────────────────────────────────────
    def status(self):
        return {"have": have_mpq(MPQ_DIR), **self.state}

    # ── 블리자드 공식 패키지 받기 ─────────────────────────────────
    def download(self):
        if self.state["stage"] in ("downloading", "extracting"):
            return self.status()
        self.state.update(stage="downloading", pct=0, msg="블리자드 서버에 연결하는 중")
        threading.Thread(target=self._download, daemon=True).start()
        return self.status()

    def _download(self):
        try:
            self._tmp = tempfile.mkdtemp(prefix="staraistudio_")
            dst = os.path.join(self._tmp, "pkg.zip")
            with urllib.request.urlopen(PKG_URL, timeout=30) as r, open(dst, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if total:
                        self.state["pct"] = int(got * 100 / total)
                        self.state["msg"] = f"{got/1048576:.0f} / {total/1048576:.0f} MB"
            self.state.update(stage="extracting", msg="약관을 꺼내는 중")
            with zipfile.ZipFile(dst) as z:
                z.extractall(self._tmp)
            eula = next((os.path.join(dp, n) for dp, _, fs in os.walk(self._tmp)
                         for n in fs if "EULA" in n and n.endswith(".txt")), None)
            text = open(eula, "rb").read().decode("cp1252", "replace") if eula else "(약관 파일을 찾지 못했습니다)"
            self.state.update(stage="eula", pct=100, eula=text, msg="")
        except Exception as e:
            self.state.update(stage="error", msg=f"받는 중 문제가 생겼습니다: {e}")

    def agree(self):
        """약관에 동의했을 때만 게임 파일을 푼다."""
        if self.state["stage"] != "eula" or not self._tmp:
            return self.status()
        try:
            inner = next(os.path.join(dp, n) for dp, _, fs in os.walk(self._tmp)
                         for n in fs if n == "MPQs.zip")
            os.makedirs(MPQ_DIR, exist_ok=True)
            # ★ AES 암호화라 파이썬 zipfile·기본 unzip 으로는 안 풀린다. 맥 기본 tar(bsdtar)로 푼다.
            subprocess.run(["/usr/bin/tar", "-xf", inner, "--passphrase", PKG_PASS, "-C", MPQ_DIR],
                           check=True, capture_output=True)
            if not have_mpq(MPQ_DIR):
                raise RuntimeError("파일이 다 풀리지 않았습니다")
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
            self.state.update(stage="ready", msg="준비됐습니다")
        except Exception as e:
            self.state.update(stage="error", msg=f"푸는 중 문제가 생겼습니다: {e}")
        return self.status()

    def pick_folder(self):
        """이미 세 파일을 가진 사람용 — 폴더를 고르면 앱 안으로 복사한다."""
        res = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not res:
            return self.status()
        src = res[0] if isinstance(res, (list, tuple)) else res
        if not have_mpq(src):
            self.state.update(stage="error",
                              msg="그 폴더에 StarDat.mpq, BrooDat.mpq, patch_rt.mpq 가 다 있어야 합니다")
            return self.status()
        os.makedirs(MPQ_DIR, exist_ok=True)
        for n in os.listdir(src):
            if n.lower() in {x.lower() for x in NEED}:
                shutil.copy2(os.path.join(src, n), os.path.join(MPQ_DIR, n))
        self.state.update(stage="ready", msg="준비됐습니다")
        return self.status()

    # ── 스튜디오 켜기 ────────────────────────────────────────────
    def start(self):
        if self.url:
            return {"url": self.url}
        port = free_port()
        log = open(os.path.join(LOGS, "studio.log"), "a")
        self.studio = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "studio.py"), "--port", str(port)],
            cwd=HERE, env=studio_env(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        url = f"http://127.0.0.1:{port}/"
        for _ in range(120):                     # 최대 30초 기다린다 (torch 불러오는 시간)
            if self.studio.poll() is not None:
                return {"error": "스튜디오가 켜지지 못했습니다. 기록: " + os.path.join(LOGS, "studio.log")}
            try:
                urllib.request.urlopen(url + "api/status", timeout=1)
                self.url = url
                self.window.load_url(url)
                return {"url": url}
            except Exception:
                time.sleep(0.25)
        return {"error": "스튜디오가 30초 안에 응답하지 않았습니다"}

    # ── 정리 ─────────────────────────────────────────────────────
    def shutdown(self):
        if self.url:
            try:
                req = urllib.request.Request(self.url + "api/stop", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                pass
        if self.studio and self.studio.poll() is None:
            try:
                os.killpg(self.studio.pid, signal.SIGTERM)
            except Exception:
                pass
        # 학습기는 따로 세션을 잡고 뜬다. 이 앱 안에서 뜬 것(명령줄에 앱 경로가 든 것)만 골라 정리한다.
        # 이름으로 고르면 사용자가 따로 돌리던 학습까지 죽는다.
        def mine():
            try:
                out = subprocess.run(["ps", "-eo", "pid=,command="],
                                     capture_output=True, text=True).stdout
            except Exception:
                return []
            pids = []
            for line in out.splitlines():
                pid, _, cmd = line.strip().partition(" ")
                if RES in cmd and "launcher.py" not in cmd and int(pid) != os.getpid():
                    pids.append(int(pid))
            return pids

        for p in mine():
            try:
                os.kill(p, signal.SIGTERM)
            except Exception:
                pass
        for _ in range(30):                      # 3초 기다려 보고
            if not mine():
                break
            time.sleep(0.1)
        for p in mine():                         # 그래도 남으면 강제로
            try:
                os.kill(p, signal.SIGKILL)
            except Exception:
                pass
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)


def main():
    api = Api()
    window = webview.create_window(
        "스타 AI 스튜디오", os.path.join(HERE, "onboarding.html"),
        js_api=api, width=1320, height=900, min_size=(900, 640))
    api.window = window
    window.events.closed += api.shutdown
    webview.start()
    api.shutdown()


if __name__ == "__main__":
    main()
