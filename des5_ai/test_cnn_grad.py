# -*- coding: utf-8 -*-
"""有限差分梯度自检：验证 cnn.py 手写反向传播的正确性。"""
import os
import sys

import numpy as np

import features as F
from cnn import CnnNet

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    rng = np.random.default_rng(7)
    B, L, nc = 5, 40, 4
    net = CnnNet(nc, d_chg=6, dn=8, dh=12, c=5, o=6, seed=3)
    # 让所有 ReLU 恒激活（权重正、偏置大）：避开 ReLU 拐点，
    # 这样有限差分才是精确导数，可严格校验反向传播。
    net.E = (net.E * 6).astype(np.float32)
    net.E2 = (net.E2 * 6).astype(np.float32)
    for W, b in ((net.W1, net.b1), (net.W2, net.b2), (net.W3, net.b3), (net.Wf, net.bf)):
        W[...] = 0.2
        b[...] = 1.0
    net.Wo[...] = 0.2
    net.bo[...] = 0.0
    ids = rng.integers(0, 60, size=(B, L)).astype(np.int64)
    lens = rng.integers(10, L + 1, size=B)
    mask = (np.arange(L)[None, :] < lens[:, None]).astype(np.float32)
    chg_ids = rng.integers(0, 50, size=(B, 9)).astype(np.int64)
    chg_w = np.full((B, 9), 1.0 / 9, dtype=np.float32)
    Xn = rng.standard_normal((B, 8)).astype(np.float32)

    logits, cache = net.forward(ids, mask, chg_ids, chg_w, Xn)
    for k in ("rm1", "rm3", "rm5"):
        print("  激活 %s 非零比例 %.2f" % (k, float((cache[k] > 0).mean())))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    dlogits = (p * rng.standard_normal((B, nc))).astype(np.float32)
    grads = net.backward(dlogits, cache)

    eps = 1e-2
    FLOOR = 1e-3      # float32 噪声地板
    ABS_TOL = 3e-3    # 绝对容差（梯度量级小时用）
    REL_TOL = 5e-2    # 相对容差
    names = ["E", "E2", "W1", "b1", "W2", "b2", "W3", "b3", "Wf", "bf", "Wo", "bo"]
    ok = True
    for name, g in zip(names, grads):
        P = getattr(net, name)
        idxs = [tuple(rng.integers(0, s) for s in P.shape) for _ in range(4)]
        worst_rel = 0.0
        worst_abs = 0.0
        bad = False
        for ix in idxs:
            old = float(P[ix])
            P[ix] = old + eps
            lp, _ = net.forward(ids, mask, chg_ids, chg_w, Xn)
            P[ix] = old - eps
            lm, _ = net.forward(ids, mask, chg_ids, chg_w, Xn)
            P[ix] = old
            num = float(((lp - lm) * dlogits).sum()) / (2 * eps)
            ana = float(g[ix])
            diff = abs(num - ana)
            rel = diff / max(abs(num), abs(ana), FLOOR)
            worst_rel = max(worst_rel, rel)
            worst_abs = max(worst_abs, diff)
            if diff >= ABS_TOL and rel >= REL_TOL:
                bad = True
        if bad:
            ok = False
        print("  %-4s %-3s 相对误差 %.2e  绝对误差 %.2e"
              % ("BAD" if bad else "OK ", name, worst_rel, worst_abs))
    print("梯度自检: %s" % ("全部通过" if ok else "存在偏差"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
