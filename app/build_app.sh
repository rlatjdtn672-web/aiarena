#!/bin/bash
# 스타 AI 스튜디오 맥 앱 빌드.
#
#   bash app/build_app.sh
#
# 결과: app/dist/StarAIStudio.app  (+ app/dist/StarAIStudio.dmg)
#
# 앱 안에 들어가는 것
#   Resources/python   이동 가능한 파이썬 3.11 + torch·numpy·pillow·pywebview
#   Resources/engine   OpenBW 학습 환경(bwmicro_sp) + SDL 라이브러리
#   Resources/app      스튜디오 코드와 첫 화면
#   Resources/maps     전투 맵
# 들어가지 않는 것
#   게임 파일(MPQ) — 블리자드 약관상 재배포 금지. 사용자가 첫 실행 때 블리자드에서 직접 받는다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APPDIR="$ROOT/app"
BUILD="$APPDIR/build"
DIST="$APPDIR/dist"
APP="$DIST/StarAIStudio.app"
RES="$APP/Contents/Resources"

step() { printf "\n── %s\n" "$*"; }

step "0. 정리"
rm -rf "$APP" "$DIST/StarAIStudio.dmg"
mkdir -p "$RES" "$APP/Contents/MacOS" "$BUILD"

step "1. 파이썬 (이동 가능한 빌드)"
if ! ls -d "$BUILD"/pycache/cpython-3.11*-macos-aarch64-none >/dev/null 2>&1; then
  uv python install 3.11 --install-dir "$BUILD/pycache"
fi
PYSRC="$(ls -d "$BUILD"/pycache/cpython-3.11.*-macos-aarch64-none | head -1)"
cp -R "$PYSRC" "$RES/python"
rm -f "$RES/python/lib/python3.11/EXTERNALLY-MANAGED"
PY="$RES/python/bin/python3"
"$PY" -m pip install --quiet --no-cache-dir --disable-pip-version-check \
  torch numpy pillow pywebview
# 앱에 필요 없는 것 덜어내기 (테스트·헤더·캐시)
find "$RES/python" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -rf "$RES/python/lib/python3.11/site-packages/torch/include" \
       "$RES/python/lib/python3.11/test" \
       "$RES/python/lib/python3.11/idlelib" \
       "$RES/python/lib/python3.11/tkinter" \
       "$RES/python/share"
"$PY" -c "import torch, numpy, PIL, webview; print('  torch', torch.__version__, '· numpy', numpy.__version__)"

step "2. 엔진 (OpenBW 학습 환경)"
if [ ! -x "$ROOT/engine/bwmicro_sp" ]; then
  (cd "$ROOT/engine" && ./build.sh bwmicro_sp)
fi
mkdir -p "$RES/engine"
cp "$ROOT/engine/bwmicro_sp" "$RES/engine/"
/opt/homebrew/bin/python3.11 "$APPDIR/bundle_dylibs.py" "$RES/engine/bwmicro_sp" "$RES/engine/lib" \
  /opt/homebrew/opt/sdl3/lib/libSDL3.dylib

step "3. 스튜디오 코드·맵"
mkdir -p "$RES/app" "$RES/maps"
cp "$APPDIR"/src/*.py "$APPDIR"/src/*.html "$RES/app/"
cp "$ROOT/maps/Weave_v1.scx" "$RES/maps/"

step "4. 앱 껍데기"
cat > "$APP/Contents/MacOS/StarAIStudio" << 'SH'
#!/bin/bash
R="$(cd "$(dirname "$0")/../Resources" && pwd)"
export SC_HOME="$R" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME/Library/Application Support/StarAIStudio/logs"
exec "$R/python/bin/python3" "$R/app/launcher.py" \
  >> "$HOME/Library/Application Support/StarAIStudio/logs/app.log" 2>&1
SH
chmod +x "$APP/Contents/MacOS/StarAIStudio"

cat > "$APP/Contents/Info.plist" << 'PL'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>StarAIStudio</string>
  <key>CFBundleDisplayName</key><string>스타 AI 스튜디오</string>
  <key>CFBundleIdentifier</key><string>kr.aiarena.studio</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>StarAIStudio</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key>
  <dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict>
</plist>
PL

step "5. 아이콘"
"$PY" "$APPDIR/make_icon.py" "$BUILD/icon.iconset"
iconutil -c icns "$BUILD/icon.iconset" -o "$RES/icon.icns"

step "6. 서명 (애드혹)"
codesign --force --deep -s - "$APP" 2>&1 | tail -2 || true
codesign --verify --deep "$APP" && echo "  서명 확인 OK"

step "7. 디스크 이미지"
STAGE="$BUILD/dmg"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -quiet -volname "스타 AI 스튜디오" -srcfolder "$STAGE" -ov -format UDZO "$DIST/StarAIStudio.dmg"

step "끝"
du -sh "$APP" "$DIST/StarAIStudio.dmg"
