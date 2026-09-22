"""aiarena 채점 서버 — 백준처럼 '학습은 각자, 채점은 서버에서'.

한 프로세스 안에 셋이 다 들어 있다. 서버 한 대로 운영하려고 일부러 단순하게 만들었다.
  1. 웹      제출 화면 · 채점 현황 · 순위 (web/index.html 한 장)
  2. 대기열  SQLite 표 하나. 제출이 들어오면 '기다리는 중' 으로 쌓인다
  3. 채점기  뒤에서 도는 스레드가 하나씩 꺼내 숨긴 테스트케이스로 채점한다 (train/grade.py)

    python server/app.py                 # http://127.0.0.1:8800
    python server/app.py --rejudge       # 전부 다시 채점 (엔진·케이스를 바꿨을 때)

환경변수 (server/README.md 에 자세히)
  SC_BIN SC_MPQ SC_MAP                   엔진·게임 파일·맵 (없으면 저장소 기본 경로)
  SC_GRADE_CASES SC_GRADE_POLICIES       숨긴 케이스·숨긴 상대 전략 (공개 저장소 밖에 둔다)
  ARENA_DATA                             DB·제출 파일을 둘 폴더 (기본 server/data)
  ARENA_SECRET                           로그인 쿠키 서명 키 (운영에선 반드시 지정)
  GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET  깃허브 로그인
  ARENA_DEV_LOGIN=1                      깃허브 없이 이름만으로 로그인 (내 컴퓨터 시험용. 운영 금지)
  ARENA_DAILY_LIMIT                      문제마다 하루 제출 횟수 (기본 10)
"""
import argparse, hashlib, hmac, http.server, json, os, secrets, socketserver, sqlite3, sys
import threading, time, traceback, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("SC_BIN", os.path.join(ROOT, "engine", "bwmicro_sp"))
os.environ.setdefault("SC_MPQ", os.path.join(ROOT, "data", "mpq"))
os.environ.setdefault("SC_MAP", os.path.join(ROOT, "maps", "Fighting_Spirit_1.3.scx"))
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.join(ROOT, "train"))
import grade as G                                  # noqa: E402

DATA = os.environ.get("ARENA_DATA", os.path.join(HERE, "data"))
SUBS = os.path.join(DATA, "subs")
DB_PATH = os.path.join(DATA, "arena.db")
SECRET = (os.environ.get("ARENA_SECRET") or "").encode()
DEV_LOGIN = os.environ.get("ARENA_DEV_LOGIN") == "1"
GH_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GH_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
DAILY_LIMIT = int(os.environ.get("ARENA_DAILY_LIMIT", "10"))
MAX_BYTES = 200_000                                # 정상 제출은 약 94KB
KST = 9 * 3600

PROBLEMS = {
    "1001": {"side": "vulture", "이름": "벌처 무빙샷",
             "설명": "벌처 1기로 저글링 6마리를 잡으세요. 벌처는 느리게 쏘고 빠르게 달립니다. 저글링은 붙으면 아픕니다.",
             "제출": "벌처 쪽 AI (앱의 v 세대 파일)"},
    "1002": {"side": "zergling", "이름": "저글링 포위",
             "설명": "저글링 6마리로 벌처 1기를 잡으세요. 벌처는 쏘고 도망갑니다. 둘러싸야 잡힙니다.",
             "제출": "저글링 쪽 AI (앱의 z 세대 파일)"},
}

os.makedirs(SUBS, exist_ok=True)
_db_lock = threading.Lock()
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.executescript("""
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, login TEXT UNIQUE, name TEXT, avatar TEXT, via TEXT, created REAL);
CREATE TABLE IF NOT EXISTS subs (
  id INTEGER PRIMARY KEY, user_id INTEGER, problem TEXT, sha TEXT, size INTEGER, created REAL,
  status TEXT, verdict TEXT, score INTEGER, passed INTEGER, total INTEGER, detail TEXT,
  board TEXT, graded REAL, seconds REAL);
CREATE INDEX IF NOT EXISTS subs_user ON subs(user_id, problem, created);
CREATE INDEX IF NOT EXISTS subs_status ON subs(status, id);
""")
_wake = threading.Event()


def q(sql, args=(), one=False):
    with _db_lock:
        cur = _db.execute(sql, args)
        _db.commit()
        rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


# ── 로그인 (서명한 쿠키) ─────────────────────────────────────────────
def _sign(v):
    return hmac.new(SECRET, v.encode(), hashlib.sha256).hexdigest()[:32]


def make_cookie(uid):
    v = f"{uid}.{int(time.time())}"
    return f"arena={v}.{_sign(v)}; Path=/; HttpOnly; SameSite=Lax; Max-Age={60 * 60 * 24 * 30}"


def user_of(headers):
    for part in (headers.get("Cookie") or "").split(";"):
        k, _, val = part.strip().partition("=")
        if k != "arena":
            continue
        try:
            uid, ts, sig = val.split(".")
        except ValueError:
            return None
        if hmac.compare_digest(sig, _sign(f"{uid}.{ts}")) and time.time() - int(ts) < 60 * 60 * 24 * 30:
            return q("SELECT * FROM users WHERE id=?", (int(uid),), one=True)
    return None


def upsert_user(login, name, avatar, via):
    q("INSERT INTO users(login,name,avatar,via,created) VALUES(?,?,?,?,?) "
      "ON CONFLICT(login) DO UPDATE SET name=excluded.name, avatar=excluded.avatar",
      (login, name, avatar, via, time.time()))
    return q("SELECT * FROM users WHERE login=?", (login,), one=True)


# ── 채점기 (뒤에서 하나씩) ───────────────────────────────────────────
def grader_loop():
    q("UPDATE subs SET status='기다리는 중' WHERE status='채점 중'")   # 서버가 죽었다 켜졌을 때
    while True:
        row = q("SELECT * FROM subs WHERE status='기다리는 중' ORDER BY id LIMIT 1", one=True)
        if row is None:
            _wake.wait(5); _wake.clear(); continue
        sid = row["id"]
        q("UPDATE subs SET status='채점 중' WHERE id=?", (sid,))
        try:
            side = PROBLEMS[row["problem"]]["side"]
            r = G.grade_hidden(os.path.join(SUBS, f"{sid}.npz"), side, reveal=False)
            detail = [{"케이스": x["케이스"], "통과": x["통과"]} for x in r["표"]]
            q("UPDATE subs SET status='끝', verdict=?, score=?, passed=?, total=?, detail=?, board=?, "
              "graded=?, seconds=? WHERE id=?",
              (r["결과"], r["점수"], r["통과"], r["케이스수"], json.dumps(detail, ensure_ascii=False),
               r["판버전"], time.time(), r["초"], sid))
        except G.형식오류 as e:
            q("UPDATE subs SET status='끝', verdict='형식 오류', score=0, detail=? WHERE id=?",
              (json.dumps({"오류": str(e)}, ensure_ascii=False), sid))
        except Exception as e:
            traceback.print_exc()
            q("UPDATE subs SET status='끝', verdict='채점 오류', score=0, detail=? WHERE id=?",
              (json.dumps({"오류": "서버 문제로 채점하지 못했습니다. 다시 제출해 주세요."}, ensure_ascii=False), sid))
            print("채점 오류", sid, e, flush=True)


# ── 보여줄 모양 ─────────────────────────────────────────────────────
def sub_json(r, users=None):
    u = users.get(r["user_id"]) if users is not None else q("SELECT * FROM users WHERE id=?", (r["user_id"],), one=True)
    d = {"id": r["id"], "problem": r["problem"], "user": u["login"] if u else "?",
         "name": (u["name"] or u["login"]) if u else "?", "created": r["created"], "status": r["status"],
         "verdict": r["verdict"], "score": r["score"], "passed": r["passed"], "total": r["total"],
         "size": r["size"], "seconds": r["seconds"]}
    if r["detail"]:
        d["detail"] = json.loads(r["detail"])
    return d


def ranking(problem):
    """사람마다 최고 점수 하나. 같은 점수면 그 점수에 먼저 닿은 사람이 위 (백준과 같다)."""
    rows = q("SELECT * FROM subs WHERE problem=? AND status='끝' AND score IS NOT NULL ORDER BY id", (problem,))
    best, tries = {}, {}
    for r in rows:
        tries[r["user_id"]] = tries.get(r["user_id"], 0) + 1
        b = best.get(r["user_id"])
        if b is None or r["score"] > b["score"]:
            best[r["user_id"]] = r
    users = {u["id"]: u for u in q("SELECT * FROM users")}
    out = sorted(best.values(), key=lambda r: (-r["score"], r["created"]))
    return [{"rank": i + 1, "user": users[r["user_id"]]["login"],
             "name": users[r["user_id"]]["name"] or users[r["user_id"]]["login"],
             "score": r["score"], "verdict": r["verdict"], "passed": r["passed"], "total": r["total"],
             "at": r["created"], "sub": r["id"], "tries": tries[r["user_id"]]} for i, r in enumerate(out)]


def today_count(uid, problem):
    now = time.time()
    day0 = now - ((now + KST) % 86400)             # 한국 시간 자정부터
    return q("SELECT COUNT(*) c FROM subs WHERE user_id=? AND problem=? AND created>=?",
             (uid, problem, day0), one=True)["c"]


# ── HTTP ───────────────────────────────────────────────────────────
class H(http.server.BaseHTTPRequestHandler):
    server_version = "aiarena"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, url, cookie=None):
        self.send_response(302)
        self.send_header("Location", url)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def _body(self, limit=MAX_BYTES):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError(f"파일이 너무 큽니다 ({n:,}바이트). 제출 파일은 {limit:,}바이트까지입니다.")
        return self.rfile.read(n)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p, qs = u.path, urllib.parse.parse_qs(u.query)
        me = user_of(self.headers)
        try:
            if p in ("/", "/index.html") or p.startswith("/p/") or p in ("/status", "/ranking", "/me"):
                with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if p == "/api/me":
                return self._send(200, {"user": dict(me) if me else None, "dev_login": DEV_LOGIN,
                                        "github": bool(GH_ID), "daily_limit": DAILY_LIMIT})
            if p == "/api/problems":
                out = []
                for pid, pr in PROBLEMS.items():
                    st = q("SELECT COUNT(*) n, COUNT(DISTINCT user_id) u, "
                           "SUM(verdict='맞았습니다') ok FROM subs WHERE problem=? AND status='끝'", (pid,), one=True)
                    out.append({"id": pid, **pr, "제출수": st["n"], "사람수": st["u"], "맞은수": st["ok"] or 0,
                                "오늘남은": (DAILY_LIMIT - today_count(me["id"], pid)) if me else None})
                return self._send(200, out)
            if p == "/api/status":
                where, args = ["1=1"], []
                if qs.get("problem"):
                    where.append("problem=?"); args.append(qs["problem"][0])
                if qs.get("user"):
                    uu = q("SELECT id FROM users WHERE login=?", (qs["user"][0],), one=True)
                    where.append("user_id=?"); args.append(uu["id"] if uu else -1)
                rows = q(f"SELECT * FROM subs WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT 50", args)
                users = {x["id"]: x for x in q("SELECT * FROM users")}
                waiting = q("SELECT COUNT(*) c FROM subs WHERE status!='끝'", one=True)["c"]
                return self._send(200, {"rows": [sub_json(r, users) for r in rows], "waiting": waiting})
            if p.startswith("/api/sub/"):
                r = q("SELECT * FROM subs WHERE id=?", (int(p.rsplit("/", 1)[1]),), one=True)
                return self._send(200, sub_json(r)) if r else self._send(404, {"error": "없는 제출입니다"})
            if p.startswith("/api/ranking/"):
                return self._send(200, ranking(p.rsplit("/", 1)[1]))
            if p == "/login":
                if not GH_ID:
                    return self._redirect("/me")
                st = secrets.token_urlsafe(16)
                url = "https://github.com/login/oauth/authorize?" + urllib.parse.urlencode(
                    {"client_id": GH_ID, "state": st, "scope": "read:user"})
                self.send_response(302)
                self.send_header("Location", url)
                self.send_header("Set-Cookie", f"arena_st={st}; Path=/; HttpOnly; SameSite=Lax; Max-Age=600")
                return self.end_headers()
            if p == "/auth/github/callback":
                return self.github_callback(qs)
            self._send(404, {"error": "없는 주소입니다"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def github_callback(self, qs):
        st = qs.get("state", [""])[0]
        ck = dict(x.strip().partition("=")[::2] for x in (self.headers.get("Cookie") or "").split(";") if "=" in x)
        if not st or not hmac.compare_digest(st, ck.get("arena_st", "")):
            return self._send(400, {"error": "로그인 확인값이 맞지 않습니다. 다시 시도해 주세요."})
        req = urllib.request.Request("https://github.com/login/oauth/access_token", method="POST",
                                     data=urllib.parse.urlencode({"client_id": GH_ID, "client_secret": GH_SECRET,
                                                                  "code": qs.get("code", [""])[0]}).encode(),
                                     headers={"Accept": "application/json"})
        tok = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("access_token")
        if not tok:
            return self._send(400, {"error": "깃허브 로그인에 실패했습니다"})
        gu = json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/user", headers={"Authorization": f"Bearer {tok}",
                                                    "Accept": "application/vnd.github+json"}), timeout=10).read())
        user = upsert_user(gu["login"], gu.get("name") or gu["login"], gu.get("avatar_url"), "github")
        self._redirect("/me", make_cookie(user["id"]))

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        p, qs = u.path, urllib.parse.parse_qs(u.query)
        me = user_of(self.headers)
        try:
            if p == "/api/dev-login":
                if not DEV_LOGIN:
                    return self._send(403, {"error": "시험용 로그인은 꺼져 있습니다"})
                name = (json.loads(self._body(1000) or b"{}").get("name") or "").strip()[:20]
                if not name:
                    return self._send(400, {"error": "이름을 적어 주세요"})
                user = upsert_user("dev:" + name, name, None, "dev")
                return self._send(200, {"ok": True}, headers={"Set-Cookie": make_cookie(user["id"])})
            if p == "/api/logout":
                return self._send(200, {"ok": True}, headers={"Set-Cookie": "arena=; Path=/; Max-Age=0"})
            if p == "/api/submit":
                return self.submit(me, qs)
            self._send(404, {"error": "없는 주소입니다"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def submit(self, me, qs):
        if not me:
            return self._send(401, {"error": "로그인해야 제출할 수 있습니다"})
        pid = qs.get("problem", [""])[0]
        if pid not in PROBLEMS:
            return self._send(400, {"error": "없는 문제입니다"})
        if today_count(me["id"], pid) >= DAILY_LIMIT:
            return self._send(429, {"error": f"오늘은 이 문제에 {DAILY_LIMIT}번 모두 제출했습니다. 내일 다시 오세요."})
        data = self._body()
        sha = hashlib.sha256(data).hexdigest()
        dup = q("SELECT id FROM subs WHERE user_id=? AND problem=? AND sha=?", (me["id"], pid, sha), one=True)
        if dup:
            return self._send(200, {"id": dup["id"], "dup": True})
        # 채점 줄에 세우기 전에 모양부터 본다 (틀린 파일은 바로 돌려보낸다)
        tmp = os.path.join(SUBS, f"tmp_{secrets.token_hex(8)}.npz")
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            G.load_weights(tmp)
        except Exception as e:
            os.remove(tmp)
            msg = str(e) if isinstance(e, G.형식오류) else "앱의 [제출용 파일 만들기] 로 만든 .npz 파일만 받습니다"
            return self._send(400, {"error": f"형식 오류: {msg}"})
        with _db_lock:
            cur = _db.execute("INSERT INTO subs(user_id,problem,sha,size,created,status) VALUES(?,?,?,?,?,?)",
                              (me["id"], pid, sha, len(data), time.time(), "기다리는 중"))
            _db.commit()
            sid = cur.lastrowid
        os.replace(tmp, os.path.join(SUBS, f"{sid}.npz"))
        _wake.set()
        return self._send(200, {"id": sid})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global SECRET
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--rejudge", action="store_true", help="모든 제출을 다시 채점 줄에 세운다")
    a = ap.parse_args()
    if not SECRET:
        if not DEV_LOGIN:
            sys.exit("ARENA_SECRET 을 지정하세요 (로그인 쿠키 서명 키). 예: export ARENA_SECRET=$(openssl rand -hex 32)")
        SECRET = secrets.token_bytes(32)           # 시험용: 켤 때마다 바뀐다 (다시 로그인하면 됨)
    G._load_hidden()                               # 케이스 파일이 없거나 판 버전이 다르면 켜기 전에 멈춘다
    if a.rejudge:
        n = q("SELECT COUNT(*) c FROM subs", one=True)["c"]
        q("UPDATE subs SET status='기다리는 중', verdict=NULL, score=NULL, passed=NULL, detail=NULL")
        print(f"{n}개를 다시 채점 줄에 세웠습니다")
    threading.Thread(target=grader_loop, daemon=True).start()
    print(f"aiarena 채점 서버 — http://{a.host}:{a.port}  (판 버전 {G.board_hash()}"
          f"{' · 시험용 로그인 켜짐' if DEV_LOGIN else ''})", flush=True)
    Server((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
