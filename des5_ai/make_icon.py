# -*- coding: utf-8 -*-
"""把一张 PNG 生成 Windows exe 用的多分辨率 .ico。

用法：
  python make_icon.py                          # 用默认源图，输出 des5_ai.ico
  python make_icon.py 图片.png -o out.ico      # 指定源图与输出
  python make_icon.py --sizes 16,32,48,64,128,256

要点：
- Windows 图标必须正方形。源图若不是正方形，会居中补透明边（不拉伸变形）。
- 一个 .ico 里写入多档尺寸（16~256），Windows 会按显示环境自动挑合适的，
  避免大图缩小后糊、小图放大后毛边。
- 帧用 32 位 BGRA DIB（bitmap_format="bmp"），兼容性最好。
- 需要 Pillow： python -m pip install pillow（仅打包时需要，运行时不需要）
"""
import argparse
import os
import sys

from PIL import Image

PY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = r"D:\新建文件夹\image.png"
DEFAULT_OUT = os.path.join(PY, "des5_ai.ico")
DEFAULT_SIZES = [16, 24, 32, 48, 64, 128, 256]
MAX_SIZE = 256          # ICO 允许的最大边长


def square(im):
    """补透明边把图变成正方形（原样缩放，不拉伸）。"""
    w, h = im.size
    if w == h:
        return im
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2), im)
    return canvas


def build(src: str, out: str, sizes) -> int:
    if not os.path.isfile(src):
        raise SystemExit("找不到源图: %s" % src)

    with Image.open(src) as raw:
        im = raw.convert("RGBA")
    src_size = im.size

    im = square(im)
    master = im.resize((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)

    # 尺寸必须不超过主图，且去重升序
    use = sorted({s for s in sizes if 0 < s <= MAX_SIZE})
    master.save(out, format="ICO", sizes=[(s, s) for s in use],
                bitmap_format="bmp")

    # 回读校验：确认每一档都真的写进去了
    with Image.open(out) as chk:
        got = sorted(chk.info.get("sizes", []))
    ok = all((s, s) in got for s in use)

    print("源图    : %s  %dx%d" % (src, src_size[0], src_size[1]))
    print("输出    : %s  (%.1f KB)" % (out, os.path.getsize(out) / 1024.0))
    print("包含尺寸: %s" % ", ".join("%dx%d" % (w, h) for w, h in got))
    if not ok:
        print("[警告] 有尺寸未成功写入", file=sys.stderr)
        return 1
    print("[OK] 多分辨率图标已生成")
    return 0


def main():
    ap = argparse.ArgumentParser(description="生成 exe 用的多分辨率 .ico")
    ap.add_argument("src", nargs="?", default=DEFAULT_SRC, help="源图（PNG）")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="输出 .ico 路径")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="要写入的边长，逗号分隔")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sizes = [int(x) for x in args.sizes.replace(" ", "").split(",") if x.isdigit()]
    return build(args.src, os.path.abspath(args.out), sizes or DEFAULT_SIZES)


if __name__ == "__main__":
    sys.exit(main())
