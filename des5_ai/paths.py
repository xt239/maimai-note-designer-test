# -*- coding: utf-8 -*-
"""资源定位：源码运行时用脚本目录，打成 exe 后用 PyInstaller 的解包目录。

模型（*.npz）与网页界面（static/）都是「只读资源」。源码形态与打包形态下
它们的根目录不同：源码时是本文件所在目录；PyInstaller 单文件模式下会被
解压到 %TEMP%\\_MEIxxxxx，而用户双击 exe 时的工作目录是任意的（可能是桌面、
可能来自开始菜单），所以不能靠 CWD 或 `__file__` 碰运气，统一在这里封装。
"""
import os
import sys

# 可选：把资源目录指到别处（例如外置自己重训的模型）
ENV_HOME = "DES5_HOME"


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包出来的可执行文件里。"""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> str:
    """只读资源（模型 / 界面）的根目录。

    优先级：环境变量 DES5_HOME > PyInstaller 解包目录 > 本文件所在目录。
    """
    home = os.environ.get(ENV_HOME, "").strip()
    if home:
        return os.path.abspath(os.path.expanduser(home))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass
    return os.path.dirname(os.path.abspath(__file__))


def resolve(path: str) -> str:
    """把「可能是相对路径」的 path 解析成绝对路径。

    - 空串原样返回
    - 先展开 %VAR% 与 ~
    - 绝对路径原样返回（用户显式指定的模型不会被改写）
    - 相对路径按 bundle_dir() 解析，因此 exe 从任意目录启动都能找到内置模型
    """
    if not path:
        return path
    p = os.path.expandvars(os.path.expanduser(path.strip()))
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(bundle_dir(), p))