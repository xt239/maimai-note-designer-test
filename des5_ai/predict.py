# -*- coding: utf-8 -*-
"""
推理：给定 maidata.txt / 谱面包（zip），只看 &inote_N= → E 的音符区间，
识别 &des_N= 区间，并预测其中的谱师（仅在「有署名」标签空间内）。

支持两种架构与任意加权集成：
  model_bag.npz  词袋（音符格 + 字符 n-gram + 结构化特征 → MLP）
  model_cnn.npz  1D-CNN 时序（音符格序列 → 三层卷积 + 掩码池化）

用法：
  python predict.py <maidata.txt | xxx.zip | 目录> [--diff 5] [--top 5]
  python predict.py <path> --models "model_bag.npz:1,model_cnn.npz:0.2" --alpha 0.5

输出示例：
  [inote_5 区间] 字符 [329,12410)  第 329~519 行
  [des_5   区间] 字符 [322,333)  值区间 [329,333)  实际值 = Jack
  预测 top5     Jack 0.42 / サファ太 0.17 / ...
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import features as F
import paths
from cnn import CnnNet
from extract_dataset import decode, parse_maidata, EMPTY_LABEL

NAMELESS = {EMPTY_LABEL, "-"}


# ------------------------------------------------------------------ 模型
def _pad(ids_list, cap):
    L = max(1, min(cap, max((len(v) for v in ids_list), default=1)))
    ids = np.zeros((len(ids_list), L), dtype=np.int64)
    w = np.zeros((len(ids_list), L), dtype=np.float32)
    for r, v in enumerate(ids_list):
        n = min(len(v), L)
        if n:
            ids[r, :n] = v[:n]
            w[r, :n] = 1.0 / len(v)
    return ids, w


class BaseModel:
    arch = "?"

    def __init__(self, path):
        self.path = path
        self.m = np.load(path, allow_pickle=True)
        self.labels = [str(x) for x in self.m["labels"]]
        self.num_mean = self.m["num_mean"]
        self.num_std = self.m["num_std"]
        self.prior = self.m["prior"]
        self.alpha = float(self.m["alpha"]) if "alpha" in self.m else 1.0

    def _num(self, nums):
        return (np.stack(nums).astype(np.float32) - self.num_mean) / self.num_std


class BagModel(BaseModel):
    arch = "bag"

    def __init__(self, path):
        super().__init__(path)
        m = self.m
        self.E1, self.E2 = m["E1"], m["E2"]
        self.W1, self.b1, self.W2, self.b2 = m["W1"], m["b1"], m["W2"], m["b2"]
        self.cap = int(m["cap_cell"]) if "cap_cell" in m else F.CELL_LEN + 1200

    def score(self, feats):
        bag_ids, chg_ids, nums = zip(*feats)
        i1, w1 = _pad(bag_ids, self.cap)
        i2, w2 = _pad(chg_ids, F.CHG_LEN)
        X1 = (self.E1[i1] * w1[:, :, None]).sum(axis=1)
        X2 = (self.E2[i2] * w2[:, :, None]).sum(axis=1)
        h = np.concatenate([X1, X2, self._num(nums)], axis=1)
        return np.maximum(h @ self.W1 + self.b1, 0.0) @ self.W2 + self.b2


class CnnModel(BaseModel):
    arch = "cnn"

    def __init__(self, path):
        super().__init__(path)
        m = self.m
        self.c, self.o = int(m["c"]), int(m["o"])
        self.dh, self.d_chg = int(m["dh"]), int(m["d_chg"])
        self.cap = int(m["cap"])
        self.chg_cap = int(m["chg_cap"]) if "chg_cap" in m else F.CHG_LEN
        self.net = CnnNet(len(self.labels), d_chg=self.d_chg, dn=self.num_mean.shape[0],
                          dh=self.dh, c=self.c, o=self.o, seq_len=self.cap, seed=1)
        for k in ("E", "E2", "W1", "b1", "W2", "b2", "W3", "b3", "Wf", "bf", "Wo", "bo"):
            setattr(self.net, k, m[k])

    def score(self, feats):
        seq_ids, chg_ids, nums = zip(*feats)
        ids, mask = _pad_mask(seq_ids, self.cap)
        cids, cw = _pad(chg_ids, self.chg_cap)
        logits, _ = self.net.forward(ids, mask, cids, cw, self._num(nums))
        return logits


def _pad_mask(ids_list, cap):
    L = max(1, min(cap, max((len(v) for v in ids_list), default=1)))
    ids = np.zeros((len(ids_list), L), dtype=np.int64)
    mask = np.zeros((len(ids_list), L), dtype=np.float32)
    for r, v in enumerate(ids_list):
        n = min(len(v), L)
        if n:
            ids[r, :n] = v[:n]
            mask[r, :n] = 1.0
    return ids, mask


def load_model(path):
    m = np.load(path, allow_pickle=True)
    arch = str(m["arch"]) if "arch" in m else "bag"
    m.close()
    return CnnModel(path) if arch == "cnn" else BagModel(path)


class Predictor:
    def __init__(self, specs, alpha=None):
        """specs: [(path, weight), ...]"""
        self.models = [load_model(p) for p, _w in specs]
        w = np.asarray([x for _p, x in specs], dtype=np.float32)
        self.weights = w / w.sum()
        self.alpha = alpha if alpha is not None else self.models[0].alpha
        self.prior = self.models[-1].prior
        self.labels = self.models[0].labels
        self.arches = "+".join(m.arch for m in self.models)

    def proba(self, notes_list):
        """返回完整概率矩阵 [N, C]（融合分析用，保留全部类别）。"""
        if not notes_list:
            return np.zeros((0, len(self.labels)), dtype=np.float64)
        feats = [featurize_all(s) for s in notes_list]
        z = None
        for mdl, wt in zip(self.models, self.weights):
            lg = mdl.score(feats) * wt
            z = lg if z is None else z + lg
        z = z + self.alpha * self.prior
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def predict(self, notes_list, top=5):
        p = self.proba(notes_list)
        order = np.argsort(-p, axis=1)[:, :top]
        return [[(self.labels[j], float(p[i, j])) for j in order[i]] for i in range(p.shape[0])]


# ------------------------------------------------------------------ 多谱面融合
def fuse_probs(P, q: float = 0.0) -> np.ndarray:
    """把 N 个谱面的概率矩阵 [N, C] 融合成一个综合分布 [C]。

    用幂平均（power mean）家族，q 控制「一致性」的权重：
      q = 1   算术平均 —— 每个谱面等权，某个谱面的一票不会消失
      q = 0   几何平均（对数几率池）—— **默认，实测最优**
      q = 1.5 折中
      q → ∞   只看"是否所有谱面都指向同一人"

    留出集实测（35 位有 ≥2 张谱的谱师，判对几位）：
      vote(多数票) 31.4%   mean(q=1) 31.4%   geo(q=0) 37.1%   pow(q=1.5) 31.4%
    几何平均胜出，说明"所有谱面共同支持"比"总概率高"更可靠，
    因此默认 q=0。
    """
    if P.size == 0:
        return np.zeros(0, dtype=np.float64)
    A = np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1.0)
    if abs(q) < 1e-6:                      # 几何平均
        w = np.exp(np.log(A).mean(axis=0))
    else:
        w = (A ** q).mean(axis=0) ** (1.0 / q)
    total = w.sum()
    return w / total if total > 0 else w


# 融合可靠性：留出集实测「谱面数 → 融合命中率」
#   2 张 0/9 · 3~4 张 2/7 · 5~8 张 3/8 · 9~15 张 3/5 · 16+ 张 6/6
FUSE_RELIABILITY = (
    (3, "very-low", "谱面太少（<3），融合基本等于噪声，请直接看单张判断"),
    (5, "low", "谱面较少（3~4 张），实测融合命中率约 29%，仅供参考"),
    (9, "mid-low", "谱面偏少（5~8 张），实测融合命中率约 25~38%"),
    (16, "mid", "谱面较多（9~15 张），实测融合命中率约 40~60%"),
    (10 ** 9, "good", "谱面充足（≥16 张），实测融合命中率约 83~100%，结论较可靠"),
)


def fuse_reliability(n_charts: int) -> Dict[str, str]:
    for cap, level, note in FUSE_RELIABILITY:
        if n_charts < cap:
            return {"level": level, "note": note, "n_charts": n_charts}
    return {"level": "good", "note": "", "n_charts": n_charts}



def entropy_ratio(p) -> float:
    """归一化熵 ∈ [0,1]：0 = 全部集中在一人，1 = 完全均匀。"""
    v = np.asarray(p, dtype=np.float64)
    v = v[v > 0]
    if v.size <= 1:
        return 0.0
    h = float(-(v * np.log(v)).sum())
    return h / float(np.log(len(p)))



def featurize_all(notes):
    """(音符格序列, 字符 n-gram, 结构化数值特征)"""
    bag_ids, chg_ids, num = F.featurize(notes)
    return F.sequence_ids(notes), chg_ids, num


# ------------------------------------------------------------------ 文件解析
def read_maidata(path: str) -> Optional[Tuple[str, str]]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if os.path.basename(n).lower() == "maidata.txt"]
            if not names:
                return None
            raw = zf.read(names[0])
    else:
        with open(path, "rb") as f:
            raw = f.read()
    return decode(raw)


def read_maidata_bytes(data: bytes, name: str = "") -> Optional[Tuple[str, str]]:
    """从内存字节解析（网页上传用）：按文件名后缀判断 zip / 纯文本。"""
    if name.lower().endswith(".zip") or data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = [n for n in zf.namelist()
                         if os.path.basename(n).lower() == "maidata.txt"]
                if not names:
                    return None
                raw = zf.read(names[0])
        except (zipfile.BadZipFile, OSError):
            return None
        return decode(raw)
    return decode(data)


def line_of(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1


def locate_des_region(text: str, diff: int, inote_start: int) -> Optional[Dict[str, Any]]:
    """&des_N= 恒位于 &inote_N= 上一行；据此从音符区间反推 des 区间（取值可含空格）。"""
    tag_len = len("&des_%d=" % diff)
    best = None
    for m in re.finditer(r"(?m)^&des_%d=(.*)$" % diff, text[:inote_start]):
        best = m
    if best is None:
        return None
    value = best.group(1).rstrip("\r")
    vs = best.start() + tag_len
    return {"tag_span": [best.start(), vs + len(value)],
            "value_span": [vs, vs + len(value)], "value": value}


def expand_targets(path: str, limit: int = 20) -> Tuple[List[str], bool]:
    """目录 → 其下的 *.zip 列表（最多 limit 个）；单文件 → ([path], False)。"""
    if not os.path.isdir(path):
        return [path], False
    files: List[str] = []
    for dp, _dirs, fs in os.walk(path):
        files.extend(os.path.join(dp, f) for f in fs if f.lower().endswith(".zip"))
    files.sort()
    return files[:limit], True


def analyze_text(text: str, enc: str, label: str, predictor: "Predictor",
                 diffs: Optional[List[int]], top: int,
                 want_text: bool = True, text_limit: int = 600) -> Optional[Dict[str, Any]]:
    """核心分析：给定 maidata.txt 全文，产出区间定位 + 谱师候选。

    label 只是用于展示的来源标识（磁盘路径或上传文件名）。
    """
    meta, samples = parse_maidata(text)
    by_diff = {s["diff"]: s for s in samples}
    todo = [d for d in (diffs or sorted(by_diff)) if d in by_diff]
    if not todo:
        return None
    preds = predictor.predict([by_diff[d]["notes"] for d in todo], top=top)
    out: Dict[str, Any] = {"file": label, "encoding": enc, "meta": meta,
                           "arch": predictor.arches, "alpha": predictor.alpha, "results": []}
    for d, cand in zip(todo, preds):
        s = by_diff[d]
        a, b = s["inote_span"]
        region = locate_des_region(text, d, a)
        true_val = s["des_raw"]
        lab = true_val if true_val else EMPTY_LABEL
        in_label_space = lab not in NAMELESS
        rec: Dict[str, Any] = {
            "diff": d,
            "lv": s["lv"],
            "inote_span": [a, b],
            "inote_lines": [line_of(text, a), line_of(text, b)],
            "inote_chars": b - a,
            "notes_lines": s["notes_lines"],
            "has_e": s["has_e"],
            "des_region": region,
            "des_value": lab,
            "des_in_label_space": in_label_space,
            "top": [{"des": c, "p": round(p, 4)} for c, p in cand],
            "hit": bool(cand) and cand[0][0] == lab,
        }
        if want_text:
            rec["notes_preview"] = by_diff[d]["notes"][:text_limit]
        out["results"].append(rec)
    return out


def analyze_file(path: str, predictor: "Predictor", diffs: Optional[List[int]], top: int,
                 want_text: bool = True, text_limit: int = 600) -> Optional[Dict[str, Any]]:
    got = read_maidata(path)
    if got is None:
        return None
    text, enc = got
    return analyze_text(text, enc, path, predictor, diffs, top, want_text, text_limit)


def analyze_bytes(data: bytes, name: str, predictor: "Predictor",
                  diffs: Optional[List[int]], top: int,
                  want_text: bool = True, text_limit: int = 600) -> Optional[Dict[str, Any]]:
    """网页上传入口：直接吃内存字节，不需要服务器能访问该路径。"""
    got = read_maidata_bytes(data, name)
    if got is None:
        return None
    text, enc = got
    return analyze_text(text, enc, name, predictor, diffs, top, want_text, text_limit)


# ------------------------------------------------------------------ 批量 + 融合
def _member_record(label: str, diff: int, lv: str, text: str,
                   inote_span: Tuple[int, int], notes_lines: int, has_e: bool,
                   region: Optional[Dict[str, Any]], true_val: str,
                   notes: str, cand: List[Tuple[str, float]],
                   want_text: bool, text_limit: int,
                   meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    lab = true_val if true_val else EMPTY_LABEL
    rec: Dict[str, Any] = {
        "file": label,
        "meta": meta or {},
        "diff": diff,
        "lv": lv,
        "inote_span": [inote_span[0], inote_span[1]],
        "inote_lines": [line_of(text, inote_span[0]), line_of(text, inote_span[1])],
        "inote_chars": inote_span[1] - inote_span[0],
        "notes_lines": notes_lines,
        "has_e": has_e,
        "des_region": region,
        "des_value": lab,
        "des_in_label_space": lab not in NAMELESS,
        "top": [{"des": c, "p": round(p, 4)} for c, p in cand],
        "hit": bool(cand) and cand[0][0] == lab,
    }
    if want_text:
        rec["notes_preview"] = notes[:text_limit]
    return rec


def analyze_many(sources: Sequence[Tuple[str, str, Any]], predictor: "Predictor",
                 diffs: Optional[List[int]], top: int,
                 fuse: bool = True, fuse_q: float = 0.0, fuse_top: int = 8,
                 want_text: bool = True, text_limit: int = 600) -> Dict[str, Any]:
    """批量分析多个谱包，并把所有谱面的证据融合成一个综合「谱师印象」。

    sources: [(来源名, 来源标识, 内容), ...]
        来源标识为 'path'   → 内容是磁盘路径
        来源标识为 'bytes'  → 内容是原始字节（网页上传）

    所有谱面一次性前向传播（比逐个调用快），再从同一份概率矩阵里
    既取每个谱面的 Top-K，又做跨谱面融合。

    融合结果里的 top1_match 表示：若这批谱面本身带真值署名，
    综合结论是否与真值一致（无真值时为 null）。
    """
    members: List[Dict[str, Any]] = []
    plans: List[Tuple[str, int, Dict[str, Any]]] = []   # (label, diff, sample)
    texts: Dict[str, str] = {}
    metas: Dict[str, Dict[str, str]] = {}
    errors: List[Dict[str, str]] = []

    for label, kind, payload in sources:
        if kind == "path":
            got = read_maidata(payload)
        else:
            got = read_maidata_bytes(payload, label)
        if got is None:
            errors.append({"file": label, "error": "未找到 maidata.txt 或无法作为 zip 解析"})
            continue
        text, enc = got
        texts[label] = text
        meta, samples = parse_maidata(text)
        metas[label] = meta
        by_diff = {s["diff"]: s for s in samples}
        todo = [d for d in (diffs or sorted(by_diff)) if d in by_diff]
        if not todo:
            errors.append({"file": label,
                           "error": "该包内没有可用难度（%s）"
                                    % ("/".join(str(d) for d in (diffs or sorted(by_diff))) or "-")})
            continue
        for d in todo:
            plans.append((label, d, by_diff[d]))

    if not plans:
        return {"ok": False, "members": [], "fused": None, "errors": errors}

    # 一次性推理：所有谱面 [M] → 概率 [M, C]
    P = predictor.proba([s["notes"] for _l, _d, s in plans])

    for i, (label, d, s) in enumerate(plans):
        text = texts[label]
        a, b = s["inote_span"]
        region = locate_des_region(text, d, a)
        order = np.argsort(-P[i])[:top]
        cand = [(predictor.labels[j], float(P[i, j])) for j in order]
        members.append(_member_record(label, d, s["lv"], text, (a, b), s["notes_lines"],
                                      s["has_e"], region, s["des_raw"], s["notes"],
                                      cand, want_text, text_limit,
                                      meta=metas.get(label)))

    out: Dict[str, Any] = {"ok": True, "members": members, "fused": None, "errors": errors,
                           "arch": predictor.arches, "alpha": predictor.alpha}

    if not (fuse and len(members) > 1):
        return out

    # ---- 融合 ----
    fused = fuse_probs(P, q=fuse_q)
    order = np.argsort(-fused)[:fuse_top]
    names = [predictor.labels[j] for j in order]
    probs = [float(fused[j]) for j in order]

    # 一致性诊断：各谱面 Top-1 的票数
    votes: Dict[str, int] = {}
    for m in members:
        if m["top"]:
            n = m["top"][0]["des"]
            votes[n] = votes.get(n, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    top_votes = ranked[0][1] if ranked else 0
    total = len(members)

    # 已知署名（真值）：单独列出，不混进融合，避免"用答案证明答案"
    known = [m for m in members if m["des_in_label_space"]]
    known_names = sorted({m["des_value"] for m in known})
    truth_match: Optional[bool] = None
    if len(known_names) == 1 and known_names[0] in names:
        truth_match = (names[0] == known_names[0])

    out["fused"] = {
        "q": fuse_q,
        "n_charts": total,
        "n_files": len({m["file"] for m in members}),
        "by_diff": {str(k): v for k, v in sorted(
            Counter(m["diff"] for m in members).items())},
        "top": [{"des": n, "p": round(p, 4)} for n, p in zip(names, probs)],
        "top1": names[0] if names else None,
        "top1_p": round(probs[0], 4) if probs else 0.0,
        "margin": round(probs[0] - probs[1], 4) if len(probs) > 1 else 1.0,
        "entropy": round(entropy_ratio(fused), 4),
        "votes": ranked[:8],
        "agreement": round(top_votes / total, 4) if total else 0.0,
        "top_votes": top_votes,
        "known_signed": len(known),
        "known_names": known_names[:12],
        "truth_match": truth_match,
        "reliability": fuse_reliability(total),
    }
    return out



# ------------------------------------------------------------------ CLI
def parse_specs(arg):
    """解析 "model_bag.npz:1,model_cnn.npz:0.2" → [(绝对路径, 权重), ...]

    相对路径按 paths.resolve 解析（打包成 exe 后指向内置模型），
    因此从任意目录启动都不会因为 CWD 不同而找不到模型。
    """
    specs = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            p, w = part.rsplit(":", 1)
            specs.append((paths.resolve(p), float(w)))
        else:
            specs.append((paths.resolve(part), 1.0))
    return specs


DEFAULT_MODELS = "model_bag.npz:1,model_cnn.npz:0.2"


def _utf8_stdout() -> None:
    """让 Windows 控制台也能打印日文谱师名。"""
    reconf = getattr(sys.stdout, "reconfigure", None)
    if reconf is not None:
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="maidata.txt / zip / 目录（目录则批量 *.zip）")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="如 model_bag.npz:1,model_cnn.npz:0.2")
    ap.add_argument("--diff", default="5", help="难度：5 / 2,5 / all")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    diffs = None if args.diff == "all" else [int(x) for x in args.diff.split(",")]
    predictor = Predictor(parse_specs(args.models), alpha=args.alpha)
    print("模型: %s   alpha=%.2f   标签数=%d" % (predictor.arches, predictor.alpha, len(predictor.labels)))

    target = args.target
    if os.path.isdir(target):
        files = []
        for dp, _d, fs in os.walk(target):
            files.extend(os.path.join(dp, f) for f in fs if f.lower().endswith(".zip"))
        files.sort()
        files = files[: args.limit]
    else:
        files = [target]

    out: List[Dict[str, Any]] = []
    for f in files:
        print("=" * 78)
        print("文件: %s" % f)
        rec = analyze_file(f, predictor, diffs, args.top)
        if not rec:
            print("  跳过（无可用谱面）")
            continue
        for r in rec["results"]:
            print("  &inote_%d 区间: 字符 [%d,%d)  第 %d~%d 行  (%d 字符 / %d 行音符, E 收尾=%s)"
                  % (r["diff"], r["inote_span"][0], r["inote_span"][1],
                     r["inote_lines"][0], r["inote_lines"][1], r["inote_chars"],
                     r["notes_lines"], "是" if r["has_e"] else "否"))
            reg = r["des_region"]
            if reg:
                print("  &des_%d   区间: 字符 [%d,%d)  值区间 [%d,%d)  → 实际值 %s"
                      % (r["diff"], reg["tag_span"][0], reg["tag_span"][1],
                         reg["value_span"][0], reg["value_span"][1], repr(r["des_value"])))
            else:
                print("  &des_%d   区间: 文件中缺失" % r["diff"])
            if not r["des_in_label_space"]:
                print("  提示: 该谱无署名（%s），不在模型标签空间内，以下为「谁可能打这首歌」的候选"
                      % r["des_value"])
            print("  预测 top%d: %s" % (args.top, "  /  ".join("%s %.3f" % (c["des"], c["p"]) for c in r["top"])))
            print("  命中: %s" % ("✔" if r["hit"] else "✘"))
        m = rec["meta"]
        print("  曲名: %s  /  作曲: %s  /  版本: %s"
              % (m.get("title", ""), m.get("artist", ""), m.get("version", "")))
        out.append(rec)

    hits = [r["hit"] for rec in out for r in rec["results"] if r["des_in_label_space"]]
    if hits:
        print("=" * 78)
        print("top1 命中率（仅统计有署名的谱）: %d/%d = %.1f%%"
              % (sum(hits), len(hits), 100.0 * sum(hits) / len(hits)))
    if args.json and out:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("json -> %s" % args.json)


if __name__ == "__main__":
    main()
