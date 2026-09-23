# aiarena 채점 서버 (리눅스). 게임 파일(MPQ)은 이미지에 넣지 않는다 — 블리자드 약관상 재배포 금지.
#   서버에서 setup/get_mpq.sh 로 받아 /data/mpq 에 두고, 실행할 때 붙인다.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ make ca-certificates \
      libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev \
      python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/aiarena
COPY external ./external
COPY engine ./engine
RUN cd engine && ./build.sh bwmicro_sp
COPY maps ./maps
COPY train ./train
COPY server ./server
ENV SC_BIN=/srv/aiarena/engine/bwmicro_sp \
    SC_MPQ=/data/mpq \
    SC_MAP=/srv/aiarena/maps/Fighting_Spirit_1.3.scx \
    SC_GRADE_CASES=/secret/hidden_cases.json \
    SC_GRADE_POLICIES=/secret/hidden_policies.py \
    ARENA_DATA=/data/arena \
    SDL_AUDIODRIVER=dummy \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 8800
CMD ["python3", "server/app.py", "--host", "0.0.0.0", "--port", "8800"]
