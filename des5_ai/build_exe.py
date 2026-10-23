# -*- coding: utf-8 -*-
"""把本地界面打包成 Windows 可执行文件（PyInstaller）。

用法：
  python build_exe.py                 # 单文件 exe：_build\\dist\\des5_ai.exe，双击即用
  python build_exe.py --onedir        # 文件夹版：_build\\dist\\des5_ai\\des5_ai.exe，启动更快
  python build_exe.py --windowed      # 无黑窗口（但只能靠任务管理器结束进程）
  python build_exe.py --clean# 先删掉构建产物再打包

前置： python -m pip install pyinstaller

注意：默认产物是 --console（保留黑窗口），这样既能看到加载日志，
      也能用 Ctrl+C 正常停止服务。若要 --windowed，请清楚它无法用
      Ctrl+C 退出，且程序出错时看不到任何提示。
"""
import argparse
import os
import shutil
import subprocess
import sys

PY = os.path.dirname(os.path.abspath(__file__))
NAME = "des5_ai"
ENTRY = os.path.join(PY, "des5_app.py")
BUILD_DIR = os.path.join(PY, "_build")
ICON = os.path.join(PY, "des5_ai.ico")

# PyInstaller 用 os.pathsep 之外的字符分隔「源;目标」，Windows 上固定用分号
ADD_SEP = ";"

# 逐个文件打进包（static/ 下的内容在 expand_data 里动态枚举）
ROOT_DATA = [("model_bag.npz", "."), ("model_cnn.npz", "."), ("README.md", ".")]

# 这些模块本项目用不到，排除掉可以明显减小体积
EXCLUDE = [
    "tkinter", "unittest", "pydoc", "doctest", "distutils", "setuptools",
    "pip", "sqlite3", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "matplotlib", "pandas", "scipy", "PIL", "pytest", "IPython", "notebook",
]

# 项目内部模块：显式声明，避免 PyInstaller 静态分析漏掉
HIDDEN = ["numpy", "features", "cnn", "extract_dataset", "predict", "paths"]


def expand_data():
    """枚举要打进去的只读资源，返回 [(绝对源路径, 包内相对目录), ...]。"""
    static = os.path.join(PY, "static")
    if not os.path.isdir(static):
        raise SystemExit("缺少 static/ 目录，无法打包")
    items = [(os.path.join(static, n), "static")
             for n in sorted(os.listdir(static)) if not n.startswith(".")]
    items += [(os.path.join(PY, n), d) for n, d in ROOT_DATA]
    return items


def check_inputs():
    missing = [rel for rel in ("des5_app.py", "model_bag.npz", "model_cnn.npz", "README.md")
               if not os.path.isfile(os.path.join(PY, rel))]
    if not os.path.isdir(os.path.join(PY, "static")):
        missing.append("static/")
    if missing:
        raise SystemExit("缺少打包所需文件: %s\n"
                         "模型请先执行train.py / train_cnn.py 生成。" % ", ".join(missing))


def build(args) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit("没有装 PyInstaller，请先执行:  python -m pip install pyinstaller")

    check_inputs()
    if args.clean:
        for d in (os.path.join(BUILD_DIR, "build"), os.path.join(BUILD_DIR, "dist")):
            if os.path.isdir(d):
                print("清理 %s" % d)
                shutil.rmtree(d, ignore_errors=True)
        spec = os.path.join(BUILD_DIR, NAME + ".spec")
        if os.path.isfile(spec):
            os.remove(spec)

    os.makedirs(BUILD_DIR, exist_ok=True)
    dist = os.path.join(BUILD_DIR, "dist")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--log-level", "WARN",
        "--name", NAME,
        "--distpath", dist,
        "--workpath", os.path.join(BUILD_DIR, "build"),
        "--specpath", BUILD_DIR,
        "--onedir" if args.onedir else "--onefile",
        "--windowed" if args.windowed else "--console",
    ]
    # 图标：Windows 上 exe 的图标必须是 .ico（由 make_icon.py 从 PNG 生成）
    if not args.no_icon:
        if os.path.isfile(ICON):
            cmd += ["--icon", ICON]
        else:
            print("[提示] 没找到 %s，本次不带图标。"
                  "可用 python make_icon.py 生成。" % os.path.basename(ICON))
    for mod in HIDDEN:
        cmd += ["--hidden-import", mod]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    for src, dest in expand_data():
        cmd += ["--add-data", src + ADD_SEP + dest]
    cmd.append(ENTRY)

    print("打包模式：%s / 控制台%s / 数据文件 %d 个 / 图标%s"
          % ("文件夹" if args.onedir else "单文件",
             "无" if args.windowed else "有", len(expand_data()),
             "无" if (args.no_icon or not os.path.isfile(ICON))
             else os.path.basename(ICON)))
    print("")
    proc = subprocess.run(cmd, cwd=PY)
    if proc.returncode != 0:
        print("\n打包失败，退出码 %d" % proc.returncode)
        return proc.returncode

    # ---------- 结果汇报 ----------
    target = (os.path.join(dist, NAME, NAME + ".exe") if args.onedir
              else os.path.join(dist, NAME + ".exe"))
    if not os.path.isfile(target):
        print("\n没有找到产物: %s" % target)
        return 1

    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _d, fs in os.walk(os.path.dirname(target)) for f in fs)
    exe_mb = os.path.getsize(target) / 1048576.0
    print("")
    print("打包完成: %s" % target)
    print("  exe %.1f MB   合计 %.1f MB   模式 %s   控制台 %s"
          % (exe_mb, size / 1048576.0,
             "文件夹" if args.onedir else "单文件", "无" if args.windowed else "有"))
    if args.windowed:
        print("")
        print("  使用：双击该 exe 即可；结束进程请用任务管理器（无法 Ctrl+C）。")
    else:
        print("")
        print("  使用：双击该 exe，浏览器会自动打开界面；")
        print("        关闭黑窗口或按 Ctrl+C 即停止服务。")
    print("")
    if not args.onedir:
        print("  提示：单文件模式每次启动都要先解压到 %%TEMP%%，约需 1~3 秒；")
        print("        若嫌慢可以用 --onedir，体积更大但启动接近瞬时。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="把 des5_ai 打包成 Windows exe")
    ap.add_argument("--onedir", action="store_true", help="文件夹版而非单文件（启动更快）")
    ap.add_argument("--windowed", action="store_true", help="不显示黑窗口（无法 Ctrl+C 停止）")
    ap.add_argument("--no-icon", action="store_true", help="不设置 exe 图标")
    ap.add_argument("--clean", action="store_true", help="打包前清理构建产物")
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())