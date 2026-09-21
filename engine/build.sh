#!/bin/bash
# 학습 환경 빌드.  사용법: ./build.sh bwmicro_sp
# 필요한 것: c++ (C++17), SDL2, SDL2_mixer, SDL2_image
set -e
T="${1:-bwmicro_sp}"
cd "$(dirname "$0")"
c++ -std=c++17 -O2 -DNDEBUG \
  -I../external/openbw -I../external/openbw/ui \
  $(sdl2-config --cflags) \
  "$T.cpp" ../external/openbw/ui/sdl2.cpp -o "$T" \
  $(sdl2-config --libs) -lSDL2_mixer -lSDL2_image
echo "빌드 완료: engine/$T"
