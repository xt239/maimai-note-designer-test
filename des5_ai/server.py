# -*- coding: utf-8 -*-
"""
本地交互界面：输入 zip（或 maidata.txt）路径 → 返回区间定位与谱师预测。

启动： python server.py [--port 8000] [--host 127.0.0.1]
依赖：仅标准库（http.server）+ numpy；模型文件需已训练好。
"""
import argparse
import base64
import binascii
import json
import os
import posixpath
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import predict as P  # noqa: E402
import paths  # noqa: E402

STATIC = os.path.join(paths.bundle_dir(), "static")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

STATE = {"predictor": None, "specs": P.DEFAULT_MODELS, "alpha": None}

# 上传限制：单个文件 64 MB，单次请求 256 MB（zip 谱包通常 1~15 MB）
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BODY_BYTES = 256 * 1024 * 1024
MAX_UPLOAD_FILES = 40


def get_predictor(specs, alpha):
    key = (specs, alpha)
    if STATE["predictor"] is None or STATE["_key"] != key:
        missing = [p for p, _w in P.parse_specs(specs) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("缺少模型文件: %s" % ", ".join(missing))
        STATE["predictor"] = P.Predictor(P.parse_specs(specs), alpha=alpha)
        STATE["_key"] = key
    return STATE["predictor"]


def _diffs_of(req):
    """请求里的 diffs 字段 → Optional[List[int]]。"""
    diffs = req.get("diffs")
    if isinstance(diffs, str):
        diffs = [int(x) for x in diffs.replace(" ", "").split(",") if x.isdigit()] or None
    elif isinstance(diffs, list):
        diffs = [int(x) for x in diffs] or None
    return diffs or None


def _fuse_args(req):
    return (bool(req.get("fuse", True)),
            float(req.get("fuseQ") if req.get("fuseQ") not in (None, "") else 0.0),
            int(req.get("fuseTop") or 8))


class Handler(BaseHTTPRequestHandler):
    server_version = "des5ai/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ---------- 工具 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        rel = path.lstrip("/") or "index.html"
        rel = posixpath.normpath(rel).replace("\\", "/")
        if rel.startswith(".."):
            return self._send(403, {"error": "forbidden"})
        fp = os.path.join(STATIC, rel.replace("/", os.sep))
        if not os.path.isfile(fp):
            return self._send(404, {"error": "not found"})
        ctype = MIME.get(os.path.splitext(fp)[1].lower(), "application/octet-stream")
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    # ---------- 路由 ----------
    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        if n > MAX_BODY_BYTES:
            raise ValueError("请求体过大（%.1f MB），单个文件请控制在 %d MB 以内"
                             % (n / 1048576.0, MAX_FILE_BYTES // 1048576))
        return self.rfile.read(n)

    def _upload(self):
        """网页「浏览」上传：{"files":[{"name":"x.zip","b64":"..."}], "diffs":"5", ...}

        浏览器出于安全限制拿不到本地绝对路径，所以由前端读字节传上来，
        后端在内存里解析 zip / txt，无需服务器能访问该路径。
        """
        try:
            req = json.loads(self._read_body().decode("utf-8") or "{}")
        except ValueError:
            return self._send(400, {"error": "请求体不是合法 JSON"})
        files = req.get("files") or []
        if not files:
            return self._send(400, {"error": "没有收到文件"})
        if len(files) > MAX_UPLOAD_FILES:
            return self._send(400, {"error": "一次最多 %d 个文件（收到 %d）"
                                          % (MAX_UPLOAD_FILES, len(files))})
        specs = req.get("models") or STATE["specs"]
        alpha = req.get("alpha")
        alpha = float(alpha) if alpha not in (None, "") else None
        pred = get_predictor(specs, alpha)
        diffs = _diffs_of(req)
        top = int(req.get("top") or 6)
        show_text = bool(req.get("showText", True))
        fuse, fq, ftop = _fuse_args(req)

        sources, errors = [], []
        for item in files:
            name = str(item.get("name") or "(未命名)")
            try:
                data = base64.b64decode(item.get("b64") or "", validate=False)
            except (binascii.Error, ValueError):
                errors.append({"file": name, "error": "文件数据损坏（base64 解码失败）"})
                continue
            if len(data) > MAX_FILE_BYTES:
                errors.append({"file": name, "error": "文件过大（%.1f MB）"
                                                    % (len(data) / 1048576.0)})
                continue
            sources.append((name, "bytes", data))

        res = P.analyze_many(sources, pred, diffs, top,
                             fuse=fuse, fuse_q=fq, fuse_top=ftop, want_text=show_text)
        if not res.get("ok"):
            all_err = (res.get("errors") or []) + errors
            msg = all_err[0]["error"] if all_err else "没有可解析的谱面"
            return self._send(400, {"error": msg, "errors": all_err})

        payload = {"ok": True, "batch": True, "count": len(res["members"]),
                   "members": res["members"], "fused": res["fused"],
                   "errors": errors + (res.get("errors") or []),
                   "arch": res.get("arch"), "alpha": res.get("alpha"),
                   "source": "upload"}
        return self._send(200, payload)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._static("index.html")
        if u.path == "/api/health":
            p = STATE["predictor"]
            return self._send(200, {
                "ok": True,
                "models": STATE["specs"],
                "arch": p.arches if p else None,
                "alpha": p.alpha if p else None,
                "labels": len(p.labels) if p else 0,
            })
        if u.path.startswith("/static/"):
            return self._static(u.path[len("/static/"):])
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/upload":
            return self._upload()
        if u.path != "/api/predict":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            path = (req.get("path") or "").strip().strip('"')
            if not path:
                return self._send(400, {"error": "请填写 zip 或 maidata.txt 的完整路径"})
            path = os.path.expandvars(os.path.expanduser(path))
            if not os.path.exists(path):
                return self._send(400, {"error": "路径不存在: %s" % path})
            specs = req.get("models") or STATE["specs"]
            alpha = req.get("alpha")
            alpha = float(alpha) if alpha not in (None, "") else None
            pred = get_predictor(specs, alpha)
            diffs = _diffs_of(req)
            top = int(req.get("top") or 6)
            limit = int(req.get("limit") or 20)
            fuse, fq, ftop = _fuse_args(req)
            files, is_batch = P.expand_targets(path, limit)
            res = P.analyze_many([(fp, "path", fp) for fp in files], pred, diffs, top,
                                 fuse=fuse, fuse_q=fq, fuse_top=ftop,
                                 want_text=bool(req.get("showText", True)))
            errs = res.get("errors") or []
            if not res.get("ok"):
                msg = errs[0]["error"] if errs else "该路径下没有可用的谱面（缺少 &inote_N= → E 区间）"
                return self._send(400, {"error": msg, "errors": errs})
            payload = {"ok": True, "batch": is_batch, "count": len(res["members"]),
                       "members": res["members"], "fused": res["fused"],
                       "errors": errs, "arch": res.get("arch"),
                       "alpha": res.get("alpha"), "source": "path"}
            if not is_batch and res["members"]:
                payload["result"] = res["members"][0]   # 兼容单文件旧字段
            return self._send(200, payload)
        except FileNotFoundError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:  # 把后端异常原样带回前端，便于排查
            traceback.print_exc()
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", default=P.DEFAULT_MODELS)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    STATE["specs"] = args.models
    STATE["alpha"] = args.alpha
    try:
        get_predictor(args.models, args.alpha)
        print("模型已加载: %s" % args.models)
    except Exception as exc:
        print("警告：%s（界面仍会启动，但预测会报错）" % exc)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("界面已启动: http://%s:%d" % (args.host, args.port))
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
