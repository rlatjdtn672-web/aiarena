# aiarena 채점 서버

백준처럼 **학습은 각자, 채점은 서버에서** 합니다. 서버 한 대, 프로세스 하나로 돌아갑니다.

- 웹: 문제 · 제출 · 채점 현황 · 순위 (`web/index.html`)
- 대기열: SQLite (`$ARENA_DATA/arena.db`), 제출 파일 (`$ARENA_DATA/subs/`)
- 채점기: 뒤에서 도는 스레드 하나가 제출을 하나씩 꺼내 `train/grade.py` 의 숨긴 케이스로 채점

실측 (맥북): 제출 1건 = 320판, 2~5초. 2코어 서버 하루 약 1만 건.

## 내 컴퓨터에서 시험

```bash
ARENA_DEV_LOGIN=1 \
SC_GRADE_CASES=~/aiarena_private/hidden_cases.json \
SC_GRADE_POLICIES=~/aiarena_private/hidden_policies.py \
python server/app.py            # http://127.0.0.1:8800
```

`ARENA_DEV_LOGIN=1` 이면 깃허브 없이 이름만으로 로그인됩니다. **운영 서버에서는 절대 켜지 마세요.**

## 환경변수

| 이름 | 뜻 |
|---|---|
| `SC_GRADE_CASES` | 숨긴 케이스 파일. **공개 저장소에 두지 않는다** |
| `SC_GRADE_POLICIES` | 숨긴 상대 전략 파이썬 파일. 공개 저장소에 두지 않는다 |
| `SC_BIN` `SC_MPQ` `SC_MAP` | 엔진 · 게임 파일 폴더 · 맵. 없으면 저장소 기본 경로 |
| `ARENA_DATA` | DB·제출 파일 폴더 (기본 `server/data`) |
| `ARENA_SECRET` | 로그인 쿠키 서명 키. 운영에선 필수: `openssl rand -hex 32` |
| `GITHUB_CLIENT_ID` `GITHUB_CLIENT_SECRET` | 깃허브 로그인 (아래) |
| `ARENA_DAILY_LIMIT` | 문제마다 하루 제출 횟수 (기본 10) |

## 규칙

- 제출물은 `.npz` (앱의 [제출용 파일 만들기]). 200KB 넘으면 거절. 모양이 `39→128→128→12` 가 아니면 형식 오류.
  `allow_pickle=False` 로 열어서 제출물이 서버에서 코드를 실행할 수 없다.
- 같은 사람이 같은 파일을 다시 내면 새로 채점하지 않고 예전 결과를 보여준다.
- 결과는 케이스 번호와 통과 여부만 공개. 케이스 조건은 공개하지 않는다.
- 순위: 사람마다 최고 점수 하나, 같은 점수면 먼저 낸 사람이 위.
- 엔진·케이스를 바꾸면 판 버전이 바뀌고 서버가 켜지지 않는다 → 케이스를 다시 재고 `python server/app.py --rejudge`.

## 운영 서버에 올리기 (리눅스 한 대)

아직 리눅스에서 직접 돌려 보지는 않았습니다. 순서만 적어 둡니다.

1. 2코어·4GB 리눅스 서버 (우분투 24.04). 포트 80/443 열기.
2. 엔진 빌드: `git clone --recursive` 후 `cd engine && ./build.sh bwmicro_sp`
   (필요: clang 또는 g++, SDL2 개발 패키지. 맥용 스크립트라 리눅스에선 경로 수정이 필요할 수 있음)
3. 게임 파일: 서버에서 `bash setup/get_mpq.sh` (블리자드 공식 AI 패키지, 약관 동의)
4. `pip install numpy` (채점기는 torch 없이 돈다)
5. 숨긴 케이스 두 파일을 서버로 복사 (scp). **엔진을 서버에서 새로 빌드하면 판 버전이 달라진다**
   → 서버에서 `make_cases.py` · `pick_cases.py` 를 다시 돌려 케이스를 새로 잰다.
6. 깃허브 로그인: github.com → Settings → Developer settings → OAuth Apps → New
   - Homepage URL: `https://도메인`
   - Callback URL: `https://도메인/auth/github/callback`
   - 받은 Client ID / Secret 을 환경변수로
7. 앞단에 HTTPS: Caddy 한 줄 설정 `도메인 { reverse_proxy 127.0.0.1:8800 }`
8. systemd 로 상시 실행 (죽으면 다시 켜짐):

```ini
# /etc/systemd/system/aiarena.service
[Service]
WorkingDirectory=/srv/aiarena
EnvironmentFile=/srv/aiarena.env
ExecStart=/usr/bin/python3 server/app.py --port 8800
Restart=always
User=aiarena
[Install]
WantedBy=multi-user.target
```

## 공개 전 확인

- 블리자드 문의 답변 (게임 파일로 남의 AI 를 채점하는 서비스가 AI 연구 약관에 맞는지)
- `ARENA_DEV_LOGIN` 이 꺼져 있는지, `ARENA_SECRET` 이 지정됐는지
