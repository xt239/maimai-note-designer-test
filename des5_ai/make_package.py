# -*- coding: utf-8 -*-
"""
把本目录打包成一个自包含的 zip（代码 + 模型 + 界面 + 数据 + 文档）。

用法：
  python make_package.py                      # 输出到上级目录 des5_ai.zip
  python make_package.py --out D:\\tmp\\x.zip
  python make_package.py --no-data             # 不含 data/（体积更小，训练时需重建）
"""
import argparse
import os
import sys
import zipfile

PY = os.path.dirname(os.path.abspath(__file__))

# 白名单：代码 / 模型 / 界面 / 文档 / 数据
INCLUDE_FILES = [
    "README.md",
    "report.md",
    "pyrightconfig.json",
    "start.bat",
    "extract_dataset.py",
    "features.py",
    "build_features.py",
    "train.py",
    "train_cnn.py",
    "cnn.py",
    "test_cnn_grad.py",
    "eval_ensemble.py",
    "predict.py",
    "server.py",
    "paths.py",
    "des5_app.py",
    "build_exe.py",
    "make_package.py",
    "make_icon.py",
    "des5_ai.ico",
    "model_bag.npz",
    "model_cnn.npz",
    "metrics_bag.json",
    "metrics_cnn.json",
    "static/index.html",
    "static/bg.jpg",
    "data/summary.json",
]
INCLUDE_DATA = ["data/dataset.jsonl", "data/features.npz"]
EXCLUDE_DIRS = {"__pycache__", ".git", ".idea", ".vscode"}
EXCLUDE_EXT = {".log", ".pyc", ".tmp", ".bak"}
ROOT_NAME = "des5_ai"


def collect(with_data: bool):
    items = []
    for rel in INCLUDE_FILES:
        fp = os.path.join(PY, rel.replace("/", os.sep))
        if os.path.isfile(fp):
            items.append((fp, "%s/%s" % (ROOT_NAME, rel)))
        else:
            print("  跳过（缺失）: %s" % rel, file=sys.stderr)
    if with_data:
        for rel in INCLUDE_DATA:
            fp = os.path.join(PY, rel.replace("/", os.sep))
            if os.path.isfile(fp):
                items.append((fp, "%s/%s" % (ROOT_NAME, rel)))
            else:
                print("  跳过（缺失）: %s" % rel, file=sys.stderr)
    return items


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(PY), ROOT_NAME + ".zip"))
    ap.add_argument("--no-data", action="store_true", help="不打包 data/ 下的数据集与特征缓存")
    args = ap.parse_args()

    items = collect(not args.no_data)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)

    total_raw = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for fp, arc in items:
            z.write(fp, arc)
            total_raw += os.path.getsize(fp)

    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        names = z.namelist()

    print("打包完成: %s" % out)
    print("  文件数 %d   原始 %.1f MB   压缩后 %.1f MB   完整性校验 %s"
          % (len(names), total_raw / 1048576.0, os.path.getsize(out) / 1048576.0,
             "OK" if bad is None else "损坏:%s" % bad))
    for n in names:
        print("   ", n)


if __name__ == "__main__":
    main()
