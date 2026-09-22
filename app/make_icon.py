"""앱 아이콘 — 한 기(파랑)와 사거리, 여섯(빨강). 이 판 그대로다.

    python make_icon.py <출력 .iconset 폴더>
"""
import math
import os
import sys

from PIL import Image, ImageDraw


def draw(size):
    s = size / 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 90 * s
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=190 * s, fill=(21, 24, 31, 255))

    cx, cy = 430 * s, 520 * s
    r = 250 * s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(108, 184, 245, 40), outline=(111, 220, 140, 255),
              width=max(1, int(14 * s)))
    b = 58 * s
    d.ellipse([cx - b, cy - b, cx + b, cy + b], fill=(108, 184, 245, 255))

    for i in range(6):
        a = math.radians(-50 + i * 20)
        x = cx + 420 * s * math.cos(a)
        y = cy + 420 * s * math.sin(a)
        z = 34 * s
        d.ellipse([x - z, y - z, x + z, y + z], fill=(255, 122, 122, 255))
    return img


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    base = draw(1024)
    for pt in (16, 32, 128, 256, 512):
        base.resize((pt, pt), Image.LANCZOS).save(os.path.join(out, f"icon_{pt}x{pt}.png"))
        base.resize((pt * 2, pt * 2), Image.LANCZOS).save(os.path.join(out, f"icon_{pt}x{pt}@2x.png"))


if __name__ == "__main__":
    main()
