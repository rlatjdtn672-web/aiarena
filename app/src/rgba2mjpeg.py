"""ffmpeg 대신 쓰는 작은 JPEG 인코더.

    python rgba2mjpeg.py <가로> <세로>

표준입력으로 RGBA 원본 프레임이 줄줄이 들어오면, 한 장씩 JPEG 로 바꿔 표준출력에
이어 붙여 내보낸다. 실시간 관전 화면이 ffmpeg 에 기대던 일을 그대로 대신한다.

앱에 ffmpeg 를 통째로 넣으면 딸린 라이브러리만 수십 개라, 이 역할 하나만 떼어냈다.
"""
import io
import sys

from PIL import Image


def main():
    w, h = int(sys.argv[1]), int(sys.argv[2])
    frame = w * h * 4
    src = sys.stdin.buffer
    out = sys.stdout.buffer
    buf = bytearray()
    while True:
        chunk = src.read(frame - len(buf) if len(buf) < frame else frame)
        if not chunk:
            break
        buf += chunk
        while len(buf) >= frame:
            raw = bytes(buf[:frame])
            del buf[:frame]
            img = Image.frombuffer("RGBA", (w, h), raw, "raw", "RGBA", 0, 1).convert("RGB")
            b = io.BytesIO()
            img.save(b, "JPEG", quality=72)
            try:
                out.write(b.getvalue())
                out.flush()
            except BrokenPipeError:
                return


if __name__ == "__main__":
    main()
