# 스타 AI 스튜디오 — 맥 앱

코드 한 줄 없이, 점수 슬라이더만 만져서 벌처 1 vs 저글링 6 AI 를 학습시키는 앱입니다.
화면에 진짜 스타크래프트 그래픽으로 학습 중인 판이 흘러가고, 이긴 비율과 행동 비율이 실시간으로 나옵니다.

## 빌드

```bash
bash app/build_app.sh
```

결과: `app/dist/StarAIStudio.app`, `app/dist/StarAIStudio.dmg`

필요한 것: 애플 실리콘 맥, `uv`, homebrew 의 `sdl2 sdl2_mixer sdl2_image sdl3`

## 앱 안에 든 것

| 경로 | 내용 |
|---|---|
| `Resources/python` | 이동 가능한 파이썬 3.11 + torch·numpy·pillow·pywebview |
| `Resources/engine` | OpenBW 학습 환경 + SDL 라이브러리 40개 (경로를 앱 안으로 바꿔 넣음) |
| `Resources/app` | 스튜디오 코드, 첫 화면 |
| `Resources/maps` | 전투 맵 |

**게임 파일은 들어 있지 않습니다.** 첫 실행 때 블리자드 공식 AI 연구용 패키지를
받고 약관에 동의해야 시작됩니다. 받은 파일과 학습 결과는
`~/Library/Application Support/StarAIStudio/` 에 저장됩니다.

## 알아둘 것

- **애플 실리콘 전용**입니다 (인텔 맥 안 됨)
- 애플 개발자 서명이 없어서, 처음 열 때 "확인되지 않은 개발자" 경고가 뜹니다.
  앱을 **마우스 오른쪽 → 열기** 로 한 번 열면 그다음부터는 그냥 열립니다
- 창을 닫으면 학습도 같이 멈춥니다
