# -*- coding: utf-8 -*-
"""
在清洗后（仅真名）的留出集上比较：词袋 / 1D-CNN / 集成。
用法：python eval_ensemble.py
"""
import os
import sys

import numpy as np

import features as F
from cnn import CnnNet
from train import SEED, load, infer as infer_bag, tune_alpha, topk_metrics, macro_f1
from train_cnn import infer as infer_cnn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = os.path.dirname(os.path.abspath(__file__))


def load_cnn(path, num_dim):
    m = np.load(path, allow_pickle=True)
    net = CnnNet(int(m["c"]), d_chg=int(m["d_chg"]), dn=num_dim, dh=int(m["dh"]),
                 o=int(m["o"]), seq_len=int(m["cap"]), seed=1)
    for k in ("E", "E2", "W1", "b1", "W2", "b2", "W3", "b3", "Wf", "bf", "Wo", "bo"):
        setattr(net, k, m[k])
    return net, m


class BagNet:
    def __init__(self, path):
        m = np.load(path, allow_pickle=True)
        self.E1, self.E2 = m["E1"], m["E2"]
        self.W1, self.b1, self.W2, self.b2 = m["W1"], m["b1"], m["W2"], m["b2"]
        self.cap_cell = int(m["cap_cell"])

    def forward(self, X1, X2, Xn, training=False, rng=None):
        h = np.concatenate([X1, X2, Xn], axis=1)
        z1 = h @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        return a1 @ self.W2 + self.b2, None


def acc(z, y, k):
    return float((np.argsort(-z, axis=1)[:, :k] == y[:, None]).any(axis=1).mean())


def main():
    d = load(os.path.join(PY, "data", "features.npz"))
    y = d["y"].astype(np.int64)
    group, diff = d["group"], d["diff"]
    labels = [str(x) for x in d["labels"]]
    rng = np.random.default_rng(SEED)
    groups = np.unique(group)
    rng.shuffle(groups)
    val_set = set(groups[: int(len(groups) * 0.12)].tolist())
    va = np.where(np.array([g in val_set for g in group]))[0]

    bag = BagNet(os.path.join(PY, "model_bag.npz"))
    mb = np.load(os.path.join(PY, "model_bag.npz"), allow_pickle=True)
    cnn, mc = load_cnn(os.path.join(PY, "model_cnn.npz"), d["num"].shape[1])

    prior = mc["prior"]
    lb = infer_bag(bag, d, va, mb["num_mean"], mb["num_std"], prior)
    lc = infer_cnn(cnn, d, va, mc["num_mean"], mc["num_std"], prior, int(mc["cap"]),
                   chg_cap=int(mc["chg_cap"]))
    yv = y[va]

    print("留出集 n=%d  类别=%d" % (len(va), len(labels)))
    for tag, lg in (("词袋", lb), ("1D-CNN", lc)):
        a, _ = tune_alpha(lg, yv, prior)
        z = lg + a * prior
        print("  %-7s alpha=%.2f top1=%.4f top3=%.4f top5=%.4f macroF1=%.4f"
              % (tag, a, acc(z, yv, 1), acc(z, yv, 3), acc(z, yv, 5),
                 macro_f1(lg, yv, len(labels), prior * a)))

    best = (None, -1)
    for w in np.arange(0.0, 1.01, 0.1):
        mix = (1 - w) * lb + w * lc
        a, t1 = tune_alpha(mix, yv, prior)
        z = mix + a * prior
        t3, t5 = acc(z, yv, 3), acc(z, yv, 5)
        print("  集成 w_cnn=%.1f alpha=%.2f top1=%.4f top3=%.4f top5=%.4f" % (w, a, t1, t3, t5))
        if t1 > best[1]:
            best = ((float(w), float(a)), t1)
    print("最佳集成: w_cnn=%.1f alpha=%.2f top1=%.4f" % (best[0][0], best[0][1], best[1]))

    w, a = best[0]
    z = ((1 - w) * lb + w * lc) + a * prior
    print("\n分难度 top1（最佳集成）:")
    for dd in range(2, 8):
        sel = va[diff[va] == dd]
        if not len(sel):
            continue
        idx = np.searchsorted(va, sel)
        zz = z[idx]
        print("  diff %d  n=%4d  top1=%.3f  top5=%.3f"
              % (dd, len(sel), (np.argmax(zz, axis=1) == y[sel]).mean(), acc(zz, y[sel], 5)))


if __name__ == "__main__":
    main()
