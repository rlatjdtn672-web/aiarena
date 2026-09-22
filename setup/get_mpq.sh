#!/bin/bash
# 블리자드가 AI 연구용으로 공식 배포하는 스타크래프트 게임 파일을 받습니다.
#
#   bash setup/get_mpq.sh
#
# 받는 곳: 블리자드 공식 서버 (download.blizzard.com)
# 약관을 화면에 띄우고, 동의해야 압축을 풉니다.
set -e
cd "$(dirname "$0")/.."
DEST="data/mpq"
URL="http://download.blizzard.com/pub/ai/MPQ_AND_EULA.zip"

if [ -f "$DEST/StarDat.mpq" ] && [ -f "$DEST/BrooDat.mpq" ] && [ -f "$DEST/patch_rt.mpq" ]; then
  echo "이미 받아져 있습니다: $DEST"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "블리자드 공식 서버에서 받는 중 (약 87MB)..."
curl -fL --progress-bar -o "$TMP/pkg.zip" "$URL"
unzip -q "$TMP/pkg.zip" -d "$TMP"
EULA="$(find "$TMP" -name '*EULA*.txt' | head -1)"

echo
echo "════════════════════════════════════════════════════════════"
iconv -f CP1252 -t UTF-8 "$EULA" 2>/dev/null || cat "$EULA"
echo
echo "════════════════════════════════════════════════════════════"
echo
echo "요약: AI 테스트·머신러닝·연구 목적으로만 쓸 수 있고, 상업적 이용은 안 됩니다."
echo "      이 목적의 자동화(봇)는 허용됩니다. 파일을 남에게 다시 배포하면 안 됩니다."
echo
read -r -p "위 약관에 동의하면 'I agree' 라고 입력하세요: " ANS
if [ "$ANS" != "I agree" ]; then
  echo "동의하지 않아 중단합니다."
  exit 1
fi

mkdir -p "$DEST"
INNER="$(find "$TMP" -name 'MPQs.zip' | head -1)"
# 압축 암호는 약관 동의 문구입니다 (블리자드 배포 방식 그대로).
# ★ AES 암호화라 기본 unzip 으로는 안 풀립니다. bsdtar(맥 기본 tar) 또는 7z 를 씁니다.
PASS="Iagreetotheeula"
if tar --version 2>/dev/null | grep -q bsdtar; then
  tar -xf "$INNER" --passphrase "$PASS" -C "$DEST"
elif command -v bsdtar >/dev/null; then
  bsdtar -xf "$INNER" --passphrase "$PASS" -C "$DEST"
elif command -v 7z >/dev/null; then
  7z x -y -p"$PASS" -o"$DEST" "$INNER" >/dev/null
else
  echo "압축을 풀 도구가 없습니다. 하나 설치해주세요:"
  echo "  Ubuntu: sudo apt install libarchive-tools   (또는 p7zip-full)"
  exit 1
fi
echo
echo "완료: $DEST"
ls -la "$DEST"/*.mpq
