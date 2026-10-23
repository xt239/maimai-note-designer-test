# -*- coding: utf-8 -*-
"""
dataset.jsonl → features.npz（扁平 ragged 存储，训练与推理共用）

默认剔除无署名样本（&des_N= 为空 ∅ 或占位 "-"），只保留真实谱师。

输出：
  cell_flat/cell_off       音符格 token（格 + 相邻二元组）→ 词袋分支
  cellseq_flat/cellseq_off 纯音符格序列（按时序）        → 1D-CNN 分支
  chg_flat/chg_off         字符 2/3-gram id
  num                      结构化数值特征 [N, NUM_DIM]
  y / group / diff / lv / labels / des / des_span / inote_span / ...

用法：
  python build_features.py                      # 默认：有署名样本
  python build_features.py --named-only 0       # 保留 ∅ 与 -
  python build_features.py --min-count 2        # 剪掉只出现 1 次的长尾谱师
"""
import argparse
import json
import os
import sys

import numpy as np

import features as F

DROP_LABELS = (F.EMPTY_LABEL, "-")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _flatten(parts):
    """[np.array, ...] → (flat int32, off int64)"""
    lens = np.asarray([len(p) for p in parts], dtype=np.int64)
    off = np.zeros(len(parts) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    flat = np.empty(int(off[-1]), dtype=np.int32)
    for i, p in enumerate(parts):
        if len(p):
            flat[off[i]:off[i + 1]] = p
    return flat, off


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "data", "dataset.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "data", "features.npz"))
    ap.add_argument("--named-only", type=int, default=1,
                    help="1=剔除 ∅ 与 -（无署名/占位）；0=保留全部")
    ap.add_argument("--min-count", type=int, default=1,
                    help="标签出现次数下限，低于则整类丢弃（剪长尾）")
    ap.add_argument("--seq-len", type=int, default=F.SEQ_LEN, help="CNN 分支序列长度上限")
    args = ap.parse_args()

    seq_parts, bag_parts, chg_parts, nums = [], [], [], []
    labs, diffs, lvs, vers, titles, artists, zips, dess = [], [], [], [], [], [], [], []
    des_spans, inote_spans, has_e = [], [], []
    n_in = n_drop = 0
    counter = {}

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lab = rec["des"]
            n_in += 1
            if args.named_only and lab in DROP_LABELS:
                n_drop += 1
                continue
            counter[lab] = counter.get(lab, 0) + 1
            cells = F.cell_tokens(rec["notes"])
            cids = [F._hash(c, F.CELL_BUCKETS) for c in cells[: F.CELL_LEN]]
            for i in range(min(len(cells) - 1, 1200)):
                cids.append(F._hash(cells[i] + "\x1f" + cells[i + 1], F.CELL_BUCKETS))
            seq_parts.append(np.asarray([F._hash(c, F.CELL_BUCKETS) for c in cells[: args.seq_len]],
                                        dtype=np.int32))
            bag_parts.append(np.asarray(cids, dtype=np.int32))
            chg_parts.append(np.asarray(F.char_ngrams(rec["notes"]), dtype=np.int32))
            nums.append(F.numeric_features(rec["notes"]))
            labs.append(lab)
            diffs.append(rec["diff"])
            lvs.append(_to_float(rec["lv"]))
            vers.append(rec.get("version", ""))
            titles.append(rec.get("title", ""))
            artists.append(rec.get("artist", ""))
            zips.append(rec["zip"])
            dess.append(rec.get("des_raw", ""))
            des_spans.append(rec["des_span"])
            inote_spans.append(rec["inote_span"])
            has_e.append(bool(rec["has_e"]))
            if len(nums) % 1000 == 0:
                print("  featurized %d" % len(nums), flush=True)

    keep = {k for k, v in counter.items() if v >= max(1, args.min_count)}
    if len(keep) < len(counter):
        print("长尾剪枝：丢弃 %d 个出现次数 < %d 的标签" % (len(counter) - len(keep), args.min_count))
    mask = np.asarray([l in keep for l in labs], dtype=bool)
    n_pruned = int((~mask).sum())

    sel = np.where(mask)[0]
    labels = sorted(keep)
    label_index = {l: i for i, l in enumerate(labels)}
    y = np.asarray([label_index[labs[i]] for i in sel], dtype=np.int32)

    zips_sel = [zips[i] for i in sel]
    group_index = {}
    group = np.empty(len(sel), dtype=np.int32)
    for k, zk in enumerate(zips_sel):
        if zk not in group_index:
            group_index[zk] = len(group_index)
        group[k] = group_index[zk]

    seq_flat, seq_off = _flatten([seq_parts[i] for i in sel])
    bag_flat, bag_off = _flatten([bag_parts[i] for i in sel])
    chg_flat, chg_off = _flatten([chg_parts[i] for i in sel])
    num_mat = np.asarray(nums, dtype=np.float32)[sel]

    np.savez_compressed(
        args.out,
        cell_flat=bag_flat, cell_off=bag_off,
        cellseq_flat=seq_flat, cellseq_off=seq_off,
        chg_flat=chg_flat, chg_off=chg_off,
        num=num_mat,
        y=y, group=group,
        diff=np.asarray(diffs, dtype=np.int32)[sel],
        lv=np.asarray(lvs, dtype=np.float32)[sel],
        labels=np.asarray(labels, dtype=object),
        des=np.asarray([dess[i] for i in sel], dtype=object),
        des_span=np.asarray(des_spans, dtype=np.int64)[sel],
        inote_span=np.asarray(inote_spans, dtype=np.int64)[sel],
        has_e=np.asarray(has_e, dtype=bool)[sel],
        version=np.asarray([vers[i] for i in sel], dtype=object),
        title=np.asarray([titles[i] for i in sel], dtype=object),
        artist=np.asarray([artists[i] for i in sel], dtype=object),
        zip=np.asarray(zips_sel, dtype=object),
    )
    dcount = np.asarray(diffs, dtype=np.int32)[sel]
    print("读入 %d 条 → 剔除无署名/占位 %d 条 → 长尾剪枝 %d 条 → 入模 %d 条，标签 %d 个，歌曲 %d 首"
          % (n_in, n_drop, n_pruned, len(sel), len(labels), len(group_index)))
    print("难度分布: %s" % {int(k): int(v) for k, v in zip(*np.unique(dcount, return_counts=True))})
    top = sorted(counter.items(), key=lambda kv: -kv[1])[:10]
    print("标签 Top10: %s" % top)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
