# 리눅스 서버에 올리기

맥에서 도커로 리눅스 전체 과정을 먼저 돌려 봤습니다 (2026-09-23).

| 확인한 것 | 결과 |
|---|---|
| 엔진 빌드 (우분투 24.04) | 코드 수정 없이 됨 (16초) |
| 손코딩 대전표 | 맥과 **완전히 같음** — 판정이 컴퓨터를 안 탄다 |
| 숨긴 케이스 재현 | 128개 확인, 다른 것 0개 |
| 채점 결과 | 같은 파일 → 맥 19점 / 리눅스 19점 |
| 게임 파일 풀기 | `apt install libarchive-tools` 후 `setup/get_mpq.sh` 로 됨 |
| 속도 (2코어 제한) | 제출 1건 7~24초 (AI 가 오래 버틸수록 길다) → 하루 약 4~10만 건 |

## 서버 준비

2코어·2GB 리눅스면 충분합니다 (우분투 24.04 기준).

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git libarchive-tools
sudo mkdir -p /srv/aiarena/{mpq,secret,data}
git clone --recursive https://github.com/rlatjdtn672-web/aiarena /srv/aiarena/src
```

## 1. 게임 파일 (서버에서 직접 받는다 — 재배포 금지)

```bash
cd /srv/aiarena/src && bash setup/get_mpq.sh        # 약관 동의 후 data/mpq 에 받는다
sudo mv data/mpq/*.mpq /srv/aiarena/mpq/
```

## 2. 숨긴 케이스 (공개 저장소에 없다)

내 컴퓨터에서 서버로 복사합니다.

```bash
scp ~/aiarena_private/hidden_cases.json ~/aiarena_private/hidden_policies.py 서버:/srv/aiarena/secret/
```

판 버전은 **엔진 소스**로 계산하므로 맥에서 만든 케이스를 그대로 씁니다. 옮긴 뒤 한 번 확인하세요.

```bash
docker compose run --rm aiarena python3 /secret/verify_cases.py   # '다른 것 0개' 가 나와야 한다
```

## 3. 깃허브 로그인

github.com → Settings → Developer settings → OAuth Apps → New OAuth App

- Homepage URL: `https://도메인`
- Authorization callback URL: `https://도메인/auth/github/callback`

## 4. 켜기

```bash
cd /srv/aiarena/src/deploy
cp .env.example .env && vi .env          # 도메인·비밀키·깃허브 값 채우기
docker compose up -d --build
docker compose logs -f aiarena
```

도메인의 A 레코드를 서버 IP 로 맞춰 두면 Caddy 가 HTTPS 인증서를 알아서 받습니다.

## 손볼 일

- 엔진이나 케이스를 바꿨다 → 케이스를 다시 재고 `docker compose exec aiarena python3 server/app.py --rejudge`
- 제출·DB 는 `/srv/aiarena/data` 에만 있습니다. 이 폴더만 백업하면 됩니다.
- **공개 전 확인**: `ARENA_DEV_LOGIN` 은 절대 켜지 말 것 (compose 에 없음), `ARENA_SECRET` 이 채워졌는지, 블리자드 답이 왔는지.
