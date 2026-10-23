# -*- coding: utf-8 -*-
"""桌面版入口：双击即用 —— 自动选端口、加载模型、拉起本地服务并打开浏览器。

这是 build_exe.py 打包成 des5_ai.exe 时使用的入口脚本。与 server.py 的区别：
server.py 面向命令行（自己管参数、自己报错），本脚本面向「双击」这种场景，
把 start.bat 里用批处理做的四件事（找端口 / 检查模型 / 等服务起来 / 开浏览器）
收进Python，并且同时兼容源码运行与 PyInstaller 冻结运行。

用法：
  python des5_app.py                  # 正常启动（浏览器自动打开）
  python des5_app.py --no-browser     # 不自动开浏览器
  python des5_app.py --port 9000      # 指定起始端口
  python des5_app.py --models "model_bag.npz:1,model_cnn.npz:0.2"
"""
import argparse
import ctypes
import json
import os
import sys
import threading
import time
import urllib.request

DEFAULT_PORT = 8765
BANNER_W = 64


# ------------------------------------------------------------------ 控制台
def enable_utf8_console() -> None:
    """把控制台代码页切到 UTF-8，否则中文与日文谱师名会显示成乱码。

    打包成 --windowed 时没有真实控制台，直接跳过。
    """
    if os.name == "nt" and sys.stdout is not None:
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def say(msg: str = "") -> None:
    """打印；--windowed 打包下 stdout 为None，静默跳过。"""
    if sys.stdout is not None:
        try:
            print(msg, flush=True)
        except Exception:
            pass


def die(msg: str, code: int = 1) -> None:
    """出错退出前停一下，避免双击运行时窗口一闪而过、看不到原因。"""
    say("")
    say("  [错误] " + msg)
    say("")
    say("  按回车键关闭窗口……")
    try:
        input()
    except Exception:
        time.sleep(10)
    sys.exit(code)


# ------------------------------------------------------------------ 网络
def service_alive(url: str, timeout: float = 1.5) -> bool:
    """该地址上是否已经有本服务在跑（避免重复启动开两个窗口）。"""
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("ok") is True
    except Exception:
        return False


def open_browser_later(url: str, delay: float = 0.6) -> None:
    """等服务真正开始响应之后再弹浏览器，否则可能打开一个空白页/连接被拒。"""

    def _worker():
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if service_alive(url, timeout=0.8):
                break
            time.sleep(0.2)
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            try:
                os.startfile(url)  # type: ignore[attr-defined]
            except Exception:
                say("  [提示] 未能自动打开浏览器，请手动访问上面的地址。")

    threading.Thread(target=_worker, daemon=True).start()


def wait_forever(host: str, preferred: int, tries: int = 40):
    """绑定端口；被占用就自动往后找一个空闲的。"""
    from http.server import ThreadingHTTPServer
    from server import Handler

    last: Exception | None = None
    for i in range(tries):
        port = preferred + i
        if port > 65535:
            break
        try:
            return ThreadingHTTPServer((host, port), Handler), port
        except OSError as exc:      # 端口被占 / 权限不足
            last = exc
            continue
    raise OSError("找不到可用端口（从 %d 起试了 %d 个）: %s" % (preferred, tries, last))


# ------------------------------------------------------------------ 主流程
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="des5_ai",
        description="maimai 谱面谱师识别（inote_N 区间 -> des_N 区间）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="起始端口，被占用时自动顺延")
    ap.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    ap.add_argument("--models", default=None, help='模型清单，如 "model_bag.npz:1,model_cnn.npz:0.2"')
    ap.add_argument("--alpha", type=float, default=None, help="温度平滑alpha")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    enable_utf8_console()
    args = parse_args(argv)

    # 延迟导入：让 --help 之类的轻量调用不必付出加载 numpy 的代价
    import paths
    import server as S

    specs = args.models or S.P.DEFAULT_MODELS
    S.STATE["specs"] = specs
    S.STATE["alpha"] = args.alpha
    home = paths.bundle_dir()

    say("=" * BANNER_W)
    say("   maimai 谱面谱师识别   inote_N 区间  ->  des_N 区间")
    say("=" * BANNER_W)
    say("")
    say("  资源目录: %s" % home)

    # ---------- 模型 ----------
    missing = [p for p, _w in S.P.parse_specs(specs) if not os.path.exists(p)]
    if missing:
        die("缺少模型文件:\n         %s" % "\n         ".join(missing))
    say("  [1/3] 正在加载模型……")
    try:
        pred = S.get_predictor(specs, args.alpha)
    except Exception as exc:
        die("模型加载失败: %s: %s" % (type(exc).__name__, exc))
    say("  [2/3] 模型就绪：架构 %s · 谱师标签 %d 个 · alpha %s"
        % (pred.arches, len(pred.labels), pred.alpha))

    # ---------- 服务 ----------
    url_probe = "http://%s:%d" % (args.host, args.port)
    if service_alive(url_probe):
        # 上一次的服务还在跑：直接把它的页面打开就行，不必再起一个
        say("  [3/3] 服务已在运行（端口 %d），直接打开浏览器。" % args.port)
        if not args.no_browser:
            import webbrowser
            webbrowser.open(url_probe)
        return 0

    say("  [3/3] 正在启动服务……")
    try:
        srv, port = wait_forever(args.host, args.port)
    except OSError as exc:
        die(str(exc))
    url = "http://%s:%d" % (args.host, port)
    if port != args.port:
        say("  [提示] 原端口被占用，已改用端口 %d。" % port)

    say("")
    say("  地址   : %s" % url)
    say("  停止   : 关闭本窗口，或按 Ctrl+C")
    say("")
    if not args.no_browser:
        open_browser_later(url)
        say("  浏览器将在服务就绪后自动打开……")
    say("")

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        say("")
        say("  已停止。")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())