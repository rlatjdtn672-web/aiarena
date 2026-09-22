# aiarena

**진짜 스타크래프트로 유닛 컨트롤 AI를 학습시키는 환경입니다.**

> ### 코드 없이 해보고 싶으면 → [맥 앱 받기 (스타 AI 스튜디오)](https://github.com/rlatjdtn672-web/aiarena/releases/latest)
> 점수 슬라이더만 만지면 학습이 돌고, 진짜 스타 화면으로 AI가 배우는 걸 실시간으로 봅니다.
> 애플 실리콘 맥 전용 · 처음엔 앱을 우클릭 → 열기

벌처 1기로 저글링 6마리를 잡게 만듭니다. 어떻게 싸우라고는 한 줄도 알려주지 않습니다.
점수만 주고, 점수가 높은 행동이 살아남게 둡니다.

```bash
git clone --recursive https://github.com/rlatjdtn672-web/aiarena
cd aiarena

# 1. 게임 파일 받기 (블리자드 공식 AI 연구용 · 약관 동의)
bash setup/get_mpq.sh
# 2. 빌드
cd engine && ./build.sh && cd ..
# 3. 확인
python train/scripted_check.py
# 4. 학습
python train/train.py --ally vulture --allies 1 --enemy zergling --enemies 6 --out 실험1
```

> 이미 클론하셨다면 `git submodule update --init` 를 한 번 돌려주세요.

---

## 뭐가 도는 건가

실제 스타크래프트 엔진입니다. 화면에 뜨는 건 진짜 벌처와 진짜 저글링이고,
체력 80·사거리 160픽셀·쿨다운 30프레임 같은 수치도 게임 데이터에서 그대로 읽어옵니다.

[OpenBW](https://github.com/OpenBW/openbw) 라는 오픈소스 엔진 재구현 위에 올렸습니다.
화면 없이 돌리면 **노트북 한 대로 분당 수백 판**이 나옵니다.

## 판: 벌처 1 vs 저글링 6

한 기가 여섯을 상대합니다. 벌처는 사거리가 10배 길고 조금 빠르지만, 쿨다운이 4배 깁니다.
서서 맞으면 죽고, 도망만 다니면 시간이 끝납니다. **쏘고 빠지는 법**을 배워야 이깁니다.

사람이 손으로 짠 전략 넷을 서로 붙이면 이런 표가 나옵니다 (`train/scripted_check.py`):

```
  P0 \ P1          정지        돌진       무빙샷      벌리기
  정지           0:0 *      0:12       0:12       9:3
  돌진           0:12       0:12       0:12       5:7
  무빙샷        10:0        7:5        7:5        7:5
  벌리기         0:12       0:12       0:12       5:7
```

무빙샷만 이깁니다. 나머지는 전멸합니다. **이 표가 학습한 AI를 채점할 기준선입니다.**

## 학습

```bash
python train/train.py --ally vulture --allies 1 --enemy zergling --enemies 6 --out 실험1
```

양쪽이 같이 배우는 셀프플레이 PPO입니다. 벌처도 배우고 저글링도 배웁니다.

### 점수표가 성격을 만듭니다

여기가 이 판에서 유일하게 창작인 자리입니다.

| 손잡이 | 기본 | 올리면 |
|---|---|---|
| `--r-enemy-hp` | 300 | 적 체력을 깎으러 달려듭니다 |
| `--r-ally-hp` | 30 | 제 몸을 사립니다 |
| `--r-kill` / `--r-death` | 200 / 100 | 마무리를 챙깁니다 / 죽는 걸 무서워합니다 |
| **`--r-timeout`** | **2500** | ★ **낮추면 "안 싸우고 버티기"가 최적해가 됩니다** |
| `--frame-skip` | 4 | 몇 프레임마다 명령할지. 사람으로 치면 손 빠르기입니다 |

값은 1/100 단위 정수입니다 (300 = 3.0). `--r-step` 만 1/1000 입니다.

### 다른 판도 됩니다

`--ally`, `--enemy` 에 유닛 이름을, `--allies`, `--enemies` 에 수를 넣으면 됩니다.

```bash
python train/train.py --ally marine --allies 12 --enemy lurker --enemies 2 --out 마린럴커
```

다만 **판마다 난이도가 천차만별입니다.** 새 조합을 잡으면 먼저
`train/scripted_check.py --ally ... --enemy ...` 로 손코딩 대전표를 뽑아보세요.
모든 칸이 0:12 처럼 일방적이면 실력이 승부를 못 가르는 판이라,
아무리 오래 학습시켜도 의미가 없습니다.

---

## 채점 (백준처럼)

학습한 AI 파일을 사람이 손으로 짠 전략 4가지(정지·돌진·무빙샷·벌리기)와 붙여 성적표를 냅니다.

```bash
python train/grade.py runs/sp1/ckpt/v012.pt                   # 1001번 벌처 무빙샷
python train/grade.py runs/sp1/ckpt/z012.pt --side zergling   # 1002번 저글링 포위
python train/grade.py runs/sp1/ckpt/v012.pt --export 제출.npz  # 제출용 파일
```

- **맞았습니다** = 사람의 정답(손코딩 중 가장 잘하는 것)보다 많이 이김 · **부분 점수** · **틀렸습니다**
- 행동은 가장 높은 확률 하나만 고릅니다 → 같은 파일이면 항상 같은 점수. 채점 한 번에 몇 초.
- 여기 씨앗은 공개(예제 채점)입니다. 서버 채점은 공개하지 않은 판으로 따로 잽니다.
- 제출 파일(`.npz`)에는 신경망 숫자만 들어 있어 주고받아도 안전합니다. 설계: [docs/채점서비스_설계.md](docs/채점서비스_설계.md)

## 구조

```
engine/
  bwmicro_sp.cpp    학습 환경. OpenBW 위에서 판을 만들고 관측·보상을 계산해
                    파이썬으로 넘긴다. 유닛당 관측 39칸, 행동 12가지
  build.sh
train/
  train.py          셀프플레이 PPO
  scripted_check.py 설치 확인 + 손코딩 대전표
maps/
  Fighting_Spirit_1.3.scx  투혼 1.3 (판은 7시 방향 열린 땅에서 선다)
data/mpq/           ← 게임 파일을 넣는 곳 (비어 있음)
external/openbw     OpenBW (서브모듈)
```

### 행동 12가지

| 번호 | 뜻 |
|---|---|
| 0 | 명령 없음 — 알아서 적을 찾아 싸운다 |
| 1~8 | 8방향으로 이동 |
| 9 / 10 | 가장 가까운 적 / 체력이 가장 적은 적을 공격 |
| 11 | 가장 가까운 적의 반대로 물러난다 |

사거리 밖에서 9나 10을 내면 헛방이고, 그 판단은 버려집니다.
이게 없으면 공격 명령이 알아서 적에게 붙어버려서 움직이는 법을 배울 이유가 사라집니다.

---

## 필요한 것

- **스타크래프트 게임 파일** — 블리자드가 AI 연구용으로 공식 배포합니다. `bash setup/get_mpq.sh` ([자세히](setup/게임파일.md))
- **C++17 컴파일러 + SDL2** — `brew install sdl2 sdl2_mixer sdl2_image`
- **Python 3.11+, PyTorch, NumPy**

GPU는 필요 없습니다. 맥북 한 대로 돌아갑니다.

## 라이선스

이 저장소의 코드는 MIT 입니다.

게임 파일(`.mpq`)은 블리자드 소유라 포함하지 않습니다 — 각자 블리자드의 AI 연구용 배포본을 받습니다.
그 약관상 **AI·머신러닝 연구 목적으로만** 쓸 수 있고, 상업적 이용과 재배포는 안 됩니다.
`external/openbw` 는 별도 프로젝트이고 그쪽 조건을 따릅니다.
