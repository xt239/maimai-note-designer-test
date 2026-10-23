# -*- coding: utf-8 -*-
"""
特征工程：输入 = &inote_N= → E 之间的谱面音符原文
输出 = 三路特征
  1) cell token 嵌入（按 , 与换行切分的音符格，含相邻二元组）
  2) 字符 n-gram 嵌入（2/3-gram 哈希）
  3) 结构化数值特征（音符类型密度、小节/时值分布、同按数量等）
"""
import re
import zlib

import numpy as np

CELL_BUCKETS = 4096
CHG_BUCKETS = 8192
CELL_LEN = 3000        # 每首歌保留的音符格上限（词袋分支）
SEQ_LEN = 1024         # 1D-CNN 分支的音符序列长度
CHG_LEN = 1500         # 每首歌取前 N 个字符做 n-gram

EMPTY_LABEL = "\u2205"

COUNT_CHARS = "0123456789hVbpqwxyz*$@%/-><!C"
TICK_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
BRACKET_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
BPM_VALUES = [60, 80, 100, 120, 140, 150, 160, 170, 180, 200, 220, 240, 280, 300, 350]

FEATURE_NAMES = (
    ["log_lines", "log_cells", "log_chars", "empty_ratio", "uniq_cell_ratio",
     "cells_per_line", "bpm_log", "has_Ch", "has_Cf", "has_star", "has_pp",
     "has_dollar", "has_at", "has_slide_gtr", "has_slide_lt", "has_e_marker"]
    + ["c_%s" % c for c in COUNT_CHARS]
    + ["r_%s" % c for c in COUNT_CHARS]
    + ["tick_%d" % v for v in TICK_VALUES]
    + ["tick_m_%d" % v for v in TICK_VALUES]
    + ["brk_%d" % v for v in BRACKET_VALUES]
    + ["sim_%d" % i for i in range(5)]
    + ["tick_mean", "tick_max", "tick_std", "tick_n",
       "cell_maxlen_mean", "cell_maxlen_max", "lane_mean", "lane_std", "hold_ratio", "star_ratio"]
)

NUM_DIM = len(FEATURE_NAMES)

SPLIT_RE = re.compile(r"[,\n]")
TICK_RE = re.compile(r"\{(\d+)\}")
BRK_RE = re.compile(r"\[(\d+):(\d+)\]")
BPM_RE = re.compile(r"^\((\d+)\)")


def _hash(tok, buckets):
    return zlib.crc32(tok.encode("utf-8")) % buckets


def cell_tokens(notes):
    return [t for t in SPLIT_RE.split(notes) if t]


def char_ngrams(notes):
    s = notes[:CHG_LEN]
    ids = []
    for n in (2, 3):
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            ids.append(_hash(s[i:i + n], CHG_BUCKETS))
    return ids[:CHG_LEN]


def numeric_features(notes):
    cells = cell_tokens(notes)
    n_cells = max(1, len(cells))
    n_lines = notes.count("\n") + 1
    f = []

    uniq = len(set(cells)) / n_cells
    bpm = 0
    m = BPM_RE.search(notes)
    if m:
        try:
            bpm = float(m.group(1))
        except ValueError:
            bpm = 0.0
    bpm_log = float(np.log(bpm / 100.0)) if bpm > 0 else 0.0

    ticks = [int(t) for t in TICK_RE.findall(notes)]
    brk_a = [int(a) for a, _b in BRK_RE.findall(notes)]

    counts = [notes.count(c) for c in COUNT_CHARS]
    empty = sum(1 for c in cells if not c.strip())
    empty_ratio = empty / n_cells

    sim = [0] * 5
    maxlens = []
    lane_vals = []
    holds = 0
    stars = 0
    for c in cells:
        core = c.replace(" ", "")
        if not core:
            continue
        n_slash = core.count("/") + core.count(">") + core.count("<") + core.count("-")
        k = min(4, n_slash)
        sim[k] += 1
        maxlens.append(len(core))
        if "h" in core:
            holds += 1
        if "*" in core:
            stars += 1
        for ch in core:
            if ch.isdigit():
                lane_vals.append(ord(ch) - 48)

    tick_hist = [0.0] * len(TICK_VALUES)
    for t in ticks:
        if t in TICK_VALUES:
            tick_hist[TICK_VALUES.index(t)] += 1.0
    tick_hist = [h / max(1, len(ticks)) for h in tick_hist]
    tick_mean = float(np.mean(ticks)) if ticks else 0.0
    tick_max = float(max(ticks)) if ticks else 0.0
    tick_std = float(np.std(ticks)) if len(ticks) > 1 else 0.0

    f.extend([
        float(np.log1p(n_lines)),
        float(np.log1p(n_cells)),
        float(np.log1p(len(notes))),
        empty_ratio,
        uniq,
        n_cells / max(1.0, float(n_lines)),
        bpm_log,
        1.0 if "Ch" in notes else 0.0,
        1.0 if "Cf" in notes else 0.0,
        1.0 if "*" in notes else 0.0,
        1.0 if ("p" in notes or "q" in notes) else 0.0,
        1.0 if "$" in notes else 0.0,
        1.0 if "@" in notes else 0.0,
        1.0 if ">" in notes else 0.0,
        1.0 if "<" in notes else 0.0,
        1.0 if "\nE" in "\n" + notes else 0.0,
    ])
    f.extend(float(c) for c in counts)
    f.extend(c / n_cells for c in counts)
    f.extend(tick_hist)
    f.extend(h / max(1.0, float(len(brk_a))) for h in
             [sum(1.0 for a in brk_a if a == v) for v in TICK_VALUES])
    f.extend([brk_a.count(v) / max(1.0, float(len(brk_a))) for v in BRACKET_VALUES])
    f.extend([v / n_cells for v in sim])
    f.extend([
        tick_mean / 32.0,
        tick_max / 128.0,
        tick_std / 32.0,
        float(np.log1p(len(ticks))),
        float(np.mean(maxlens)) / 16.0 if maxlens else 0.0,
        float(max(maxlens)) / 32.0 if maxlens else 0.0,
        float(np.mean(lane_vals)) / 9.0 if lane_vals else 0.0,
        float(np.std(lane_vals)) / 9.0 if len(lane_vals) > 1 else 0.0,
        holds / n_cells,
        stars / n_cells,
    ])
    x = np.asarray(f, dtype=np.float32)
    if x.shape[0] != NUM_DIM:  # 安全兜底
        x = np.resize(x, NUM_DIM)
    return x


def featurize(notes):
    """词袋分支用：音符格 id（含二元组）、字符 n-gram、结构化特征。"""
    cells = cell_tokens(notes)
    cell_ids = [_hash(c, CELL_BUCKETS) for c in cells[:CELL_LEN]]
    for i in range(min(len(cell_ids) - 1, 1200)):
        cell_ids.append(_hash(cells[i] + "\x1f" + cells[i + 1], CELL_BUCKETS))
    return cell_ids, char_ngrams(notes), numeric_features(notes)


def sequence_ids(notes, cap=SEQ_LEN):
    """1D-CNN 分支用：按时序排列的音符格 id（保留顺序信息）。"""
    return [_hash(c, CELL_BUCKETS) for c in cell_tokens(notes)[:cap]]
