# -*- coding: utf-8 -*-
"""
1D-CNN 时序编码器（纯 numpy，手写前向/反向）

相对词袋基线的改进：保留音符格的**先后顺序**，用多尺度卷积捕捉
「起手动机 / 中段铺面 / 特殊记号段落」等谱面局部风格。

结构：
  音符格 id [B,L]
      → Embedding(C=32) → conv(k3,s2)+ReLU → maxpool2
                       → conv(k3,s2)+ReLU → maxpool2
                       → conv(k3,s2)+ReLU → 掩码全局 max/mean 池化   ⇒ 128 维
  字符 n-gram EmbeddingBag(48) ┐
  结构化数值特征(123)         ├→ concat(299) → FC256+ReLU+Dropout → FC(C)
"""
import numpy as np

import features as F


# ------------------------------------------------------------------ 基础算子
def relu(x):
    return np.maximum(x, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def sample_mask(mask, pos):
    """按位置采样掩码，越界视为 0。mask: [B,L] float32"""
    width = max(int(pos.max()) + 1, mask.shape[1])
    pad = np.zeros((mask.shape[0], width), dtype=np.float32)
    pad[:, : mask.shape[1]] = mask
    return pad[:, pos]


def ensure_len(x, mask, minlen):
    """保证序列长度 >= minlen（补零 + 掩码补 0），用于极短谱面。"""
    Lx, Lm = x.shape[-1], mask.shape[-1]
    if Lx < minlen:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, minlen - Lx)])
    if Lm < minlen:
        mask = np.pad(mask, [(0, 0), (0, minlen - Lm)])
    return x, mask


def conv_forward(x, W, b, k, stride, pos):
    """x:[B,C,L] → y:[B,Lout,O]；W:[O,C*k]；同时返回 im2col 结果 P 与位置 pos"""
    B, C, L = x.shape
    O = W.shape[0]
    Lout = len(pos)
    P = np.empty((B, Lout, C * k), dtype=np.float32)
    for i in range(k):
        P[:, :, i * C:(i + 1) * C] = x[:, :, pos + i].transpose(0, 2, 1)
    return P @ W.T + b, P


def conv_backward(dy, W, P, pos, x_len, C, k):
    B, Lout, O = dy.shape
    dP = (dy @ W).astype(np.float32)
    dW = (dy.reshape(-1, O).T @ P.reshape(-1, C * k)).astype(np.float32)
    db = dy.sum(axis=(0, 1)).astype(np.float32)
    dx = np.zeros((B, C, x_len), dtype=np.float32)
    for i in range(k):
        dx[:, :, pos + i] += dP[:, :, i * C:(i + 1) * C].transpose(0, 2, 1)
    return dx, dW.astype(np.float32), db


def maxpool2_forward(x):
    B, C, L = x.shape
    L2 = L // 2
    xr = x[:, :, : L2 * 2].reshape(B, C, L2, 2)
    arg = xr.argmax(axis=-1)
    val = np.take_along_axis(xr, arg[..., None], axis=-1)[..., 0]
    return val, arg


def maxpool2_backward(dy, arg, x_len):
    """dy: [B,C,L2d]；arg: [B,C,L2] 池化取到的下标(0/1)。

    arg 可能比 dy 短（末端补零产生的多余位置没有 arg），这些位置梯度直接丢弃。
    """
    B, C, _ = dy.shape
    L2 = arg.shape[2]
    dx = np.zeros((B, C, x_len), dtype=np.float32)
    l_idx = 2 * np.arange(L2)[None, None, :] + arg
    dx[np.arange(B)[:, None, None], np.arange(C)[None, :, None], l_idx] = dy[:, :, :L2]
    return dx


def embed_backward(nb, d, ids, mask, dX):
    """Embedding 的 mean-free 查表梯度：dE[ids[b,l]] += mask[b,l]*dX[b,l,:]"""
    B, L = ids.shape
    contrib = (mask.reshape(-1, 1) * dX.transpose(0, 2, 1).reshape(B * L, d)).astype(np.float32).reshape(-1)
    col = np.arange(d, dtype=np.int64)
    idx = (ids.reshape(-1, 1) * d + col[None, :]).reshape(-1)
    return np.bincount(idx, weights=contrib, minlength=nb * d).astype(np.float32).reshape(nb, d)


def embedbag_forward(emb, ids, w):
    """ids:[B,L]  w:[B,L] 已含 1/len 权重"""
    return (emb[ids] * w[:, :, None]).sum(axis=1)


def embedbag_backward(nb, d, ids, w, dX):
    B, L = ids.shape
    rows = np.repeat(np.arange(B), L)
    contrib = (w.reshape(-1, 1) * dX[rows]).astype(np.float32).reshape(-1)
    col = np.arange(d, dtype=np.int64)
    idx = (ids.reshape(-1, 1) * d + col[None, :]).reshape(-1)
    return np.bincount(idx, weights=contrib, minlength=nb * d).astype(np.float32).reshape(nb, d)


# ------------------------------------------------------------------ 网络
class CnnNet:
    def __init__(self, nc, d_chg=48, dn=F.NUM_DIM, dh=256, c=32, o=64, d1=None, d2=None,
                 seq_len=F.SEQ_LEN, seed=20261017, drop=0.25):
        rng = np.random.default_rng(seed)
        self.drop = float(drop)
        self.nc, self.c, self.o = nc, c, o
        self.d_chg, self.dn, self.dh = d_chg, dn, dh
        self.seq_len = seq_len
        self.E = (rng.standard_normal((F.CELL_BUCKETS, c)) * 0.05).astype(np.float32)
        self.E2 = (rng.standard_normal((F.CHG_BUCKETS, d_chg)) * 0.05).astype(np.float32)
        self.W1 = (rng.standard_normal((o, c * 3)) * np.sqrt(2.0 / (c * 3))).astype(np.float32)
        self.b1 = np.zeros(o, dtype=np.float32)
        self.W2 = (rng.standard_normal((o, o * 3)) * np.sqrt(2.0 / (o * 3))).astype(np.float32)
        self.b2 = np.zeros(o, dtype=np.float32)
        self.W3 = (rng.standard_normal((o, o * 3)) * np.sqrt(2.0 / (o * 3))).astype(np.float32)
        self.b3 = np.zeros(o, dtype=np.float32)
        din = o * 2 + d_chg + dn
        self.Wf = (rng.standard_normal((din, dh)) * np.sqrt(2.0 / din)).astype(np.float32)
        self.bf = np.zeros(dh, dtype=np.float32)
        self.Wo = (rng.standard_normal((dh, nc)) * np.sqrt(2.0 / dh)).astype(np.float32)
        self.bo = np.zeros(nc, dtype=np.float32)

    def params(self):
        return [self.E, self.E2, self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.Wf, self.bf, self.Wo, self.bo]

    # ---- 前向 ----
    def forward(self, ids, mask, chg_ids, chg_w, Xn, training=False, rng=None):
        L = ids.shape[1]
        x = (self.E[ids] * mask[:, :, None]).transpose(0, 2, 1)          # [B,c,L]

        p1 = np.arange(0, L - 2, 2)
        m1 = sample_mask(mask, p1)
        y1, P1 = conv_forward(x, self.W1, self.b1, 3, 2, p1)
        rm1 = (y1 > 0) * m1[:, :, None]
        a1 = y1 * rm1
        q1, arg1 = maxpool2_forward(a1.transpose(0, 2, 1))
        m2 = m1[:, ::2]
        if m2.shape[1] < q1.shape[1]:
            m2 = np.pad(m2, ((0, 0), (0, q1.shape[1] - m2.shape[1])))

        x2, m2 = ensure_len(q1, m2, 4)
        p2 = np.arange(0, x2.shape[2] - 2, 2)
        m3 = sample_mask(m2, p2)
        y2, P2 = conv_forward(x2, self.W2, self.b2, 3, 2, p2)
        rm3 = (y2 > 0) * m3[:, :, None]
        a2 = y2 * rm3
        q2, arg2 = maxpool2_forward(a2.transpose(0, 2, 1))
        m4 = m3[:, ::2]
        if m4.shape[1] < q2.shape[1]:
            m4 = np.pad(m4, ((0, 0), (0, q2.shape[1] - m4.shape[1])))

        x3, m4 = ensure_len(q2, m4, 4)
        p3 = np.arange(0, x3.shape[2] - 2, 2)
        m5 = sample_mask(m4, p3)
        y3, P3 = conv_forward(x3, self.W3, self.b3, 3, 2, p3)
        rm5 = (y3 > 0) * m5[:, :, None]
        a3 = y3 * rm5

        # 掩码全局池化
        a3t = a3.transpose(0, 2, 1)                                     # [B,O,L3]
        neg = np.where(m5[:, None, :] > 0, a3t, -1e9)
        am = neg.argmax(axis=-1)                                        # [B,O]
        Bi = np.arange(a3t.shape[0])[:, None]
        Oi = np.arange(self.o)[None, :]
        gmax = neg[Bi, Oi, am]
        wsum = np.maximum(m5.sum(axis=1), 1.0)
        gmean = (a3t * m5[:, None, :]).sum(axis=2) / wsum[:, None]

        X2 = embedbag_forward(self.E2, chg_ids, chg_w)
        h = np.concatenate([gmax, gmean, X2, Xn], axis=1)
        z = h @ self.Wf + self.bf
        af = relu(z)
        dmask = None
        if training and rng is not None and self.drop > 0:
            keep = 1.0 - self.drop
            dmask = (rng.random(af.shape) >= self.drop).astype(np.float32) / keep
            af = af * dmask
        logits = af @ self.Wo + self.bo
        cache = dict(ids=ids, mask=mask, chg_ids=chg_ids, chg_w=chg_w, p1=p1, m1=m1,
                     P1=P1, a1=a1, arg1=arg1, p2=p2, m3=m3, P2=P2, a2=a2, arg2=arg2,
                     p3=p3, m5=m5, P3=P3, a3=a3, am=am, gmax=gmax, wsum=wsum,
                     h=h, z=z, af=af, dmask=dmask, rm1=rm1, rm3=rm3, rm5=rm5,
                     len1=int(x.shape[2]), len2=int(x2.shape[2]), len3=int(x3.shape[2]))
        return logits, cache

    # ---- 反向 ----
    def backward(self, dlogits, cache):
        dWo = cache["af"].T @ dlogits
        dbo = dlogits.sum(axis=0)
        d_af = dlogits @ self.Wo.T
        if cache["dmask"] is not None:
            d_af = d_af * cache["dmask"]
        d_z = d_af * (cache["z"] > 0)
        d_h = d_z @ self.Wf.T
        dWf = cache["h"].T @ d_z
        dbf = d_z.sum(axis=0)
        o = self.o
        d_gmax = d_h[:, :o]
        d_gmean = d_h[:, o:2 * o]
        d_X2 = d_h[:, 2 * o:2 * o + self.d_chg]
        # 池化反传
        m5 = cache["m5"]
        a3t = cache["a3"].transpose(0, 2, 1)
        d_a3t = np.zeros_like(a3t)
        b_i = np.arange(d_gmax.shape[0])
        am = cache["am"]
        valid = np.take_along_axis(m5, am, axis=1) > 0        # [B,O]
        d_a3t[b_i[:, None], np.arange(o)[None, :], am] = np.where(valid, d_gmax, 0.0)
        d_a3t += (m5[:, None, :] * d_gmean[:, :, None]) / cache["wsum"][:, None, None]
        d_y3 = d_a3t.transpose(0, 2, 1) * cache["rm5"]
        d_x3, dW3, db3 = conv_backward(d_y3, self.W3, cache["P3"], cache["p3"],
                                       cache["len3"], o, 3)
        d_a2 = maxpool2_backward(d_x3, cache["arg2"], cache["a2"].shape[1])
        d_y2 = d_a2.transpose(0, 2, 1) * cache["rm3"]
        d_x2, dW2, db2 = conv_backward(d_y2, self.W2, cache["P2"], cache["p2"],
                                       cache["len2"], o, 3)
        d_a1 = maxpool2_backward(d_x2, cache["arg1"], cache["a1"].shape[1])
        d_y1 = d_a1.transpose(0, 2, 1) * cache["rm1"]
        d_x, dW1, db1 = conv_backward(d_y1, self.W1, cache["P1"], cache["p1"],
                                      cache["ids"].shape[1], self.c, 3)
        dE = embed_backward(F.CELL_BUCKETS, self.c, cache["ids"], cache["mask"], d_x)
        dE2 = embedbag_backward(F.CHG_BUCKETS, self.d_chg, cache["chg_ids"], cache["chg_w"], d_X2)
        return [dE, dE2, dW1, db1, dW2, db2, dW3, db3, dWf, dbf, dWo, dbo]
