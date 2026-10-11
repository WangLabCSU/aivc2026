"""BizardNet: Context-Conditioned Perturbation Response Transformer (v10_full_redesign §3.1).

三 token 序列设计——每条序列 = (上下文 c, 靶基因 g) 的 N 个响应基因:
  [CTX]  NTC basal 上下文指纹 (6 模块分数 + 全局统计 + 分位数 + top-var 基因 PCA, 19 维)
  [TRG]  靶基因 = 共享基因身份嵌入 + 预训练基因嵌入(SE600M+Tahoe 256) + 靶级标量(9)
  [GEN]×N 响应基因 = 同上嵌入 + fusion v3.5 逐行标量(20: 跨源签名统计/basal_j/模块/交互/has_sig)
  → Transformer Encoder (6L, 8H, d512, d_ff2048, pre-LN) → linear(512→1) 每基因 logFC
基因 token 无序(集合语义) → 用 token-type embedding 替代位置编码; PAD 位 key-padding mask。

数据 = l15_fusion v3.5 缓存 feat_train.npz (30.78M 行 X/y/ctx/gpos/tpos/w, 全部在册索引),
LOCO-HepG2 = 排除 ctx==hepg2 行训练。显存常驻标量张量 (~3GB) + (S,128) 序列表 gather。

子命令: build / train --arm {loco,full} / predict --src {bench_hepg2,submit_A..} --arm ...
        eval-loco  # LOCO 模型对 hepg2 训练行的 proxy 指标 (spearman/recall/sign)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
CFG_PATH = ROOT / "configs/bizard/bizardnet.yaml"
GEN_COLS = {"cross_mean", "cross_std", "cross_maxabs", "basal_j", "basal_j_zero",
            "x_sign_mean_basal", "x_absmean_basal", "has_sig"}
TRG_X_COLS = {"basal_g", "basal_g_zero"}


def load_cfg() -> dict:
    import os
    p = os.environ.get("BIZNET_CFG") or "configs/bizard/bizardnet.yaml"
    with open(ROOT / p) as f:
        return yaml.safe_load(f)


def wd(cfg: dict) -> Path:
    d = ROOT / cfg["workdir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def ctx_names(cfg: dict) -> list[str]:
    return list(cfg["train_contexts"]) + list(cfg["extra_contexts"])


def load_vocab_and_modules(cfg: dict):
    vocab = np.array(json.load(open(ROOT / cfg["vocab_json"])))
    ms = yaml.safe_load(open(ROOT / "configs/l15_fusion/v35.yaml"))["features"]["module_sets"]
    flags = np.zeros((len(ms), len(vocab)), np.float32)
    for i, m in enumerate(ms):
        spec = ms[m]
        if isinstance(spec, list):
            spec = {"genes": spec}
        hit = np.zeros(len(vocab), bool)
        for pre in spec.get("prefix", []):
            hit |= np.char.startswith(vocab, pre)
        for g in spec.get("genes", []):
            hit |= (vocab == g)
        flags[i] = hit
    return vocab, flags


def pre_emb(cfg: dict) -> np.ndarray:
    """预训练基因嵌入 (V+1, 256): SE600M ⊕ Tahoe, 标准化, 行 0 = UNK。"""
    v35 = ROOT / cfg["fusion_v35_dir"]
    e = np.concatenate([np.load(v35 / "emb_se.npy"), np.load(v35 / "emb_tahoe.npy")], 1).astype(np.float32)
    G = np.nan_to_num((e - e.mean(0)) / np.maximum(e.std(0), 1e-6))
    return np.vstack([np.zeros((1, G.shape[1]), np.float32), G])


# ------------------------------------------------------------------ build
def build_fingerprints(cfg: dict, mflags: np.ndarray) -> np.ndarray:
    """11 上下文 × 19 维指纹: [mod×6(NaN→CRISPRi 中位), mean, sd, nz, q5..q95, pca×5]."""
    v35 = ROOT / cfg["fusion_v35_dir"]
    raw = {n: np.load(v35 / f"basal_{n}.npy").astype(np.float32) for n in ctx_names(cfg)}
    crispr = cfg["crispr_contexts"]
    F = {}
    for name, b in raw.items():
        scores = []
        for i in range(mflags.shape[0]):
            sel = mflags[i] > 0
            s = float(b[sel].mean()) if sel.any() else 0.0
            scores.append(s if s > 0.1 else np.nan)   # 面板缺模块 → NaN 插补 (CCRN 教训)
        F[name] = scores + [float(b.mean()), float(b.std()), float((b > 0).mean())]
        F[name] += list(np.quantile(b, [0.05, 0.25, 0.5, 0.75, 0.95]))
    Fm = {k: np.array(v, np.float32) for k, v in F.items()}
    med = np.nanmedian(np.stack([Fm[c] for c in crispr]), axis=0)
    for k in Fm:
        Fm[k] = np.where(np.isnan(Fm[k]), med, Fm[k])
    Bc = np.stack([raw[c] for c in crispr])            # PCA: top-var 基因 SVD, k=5
    top = np.argsort(-Bc.var(axis=0))[:2000]
    Xc = Bc[:, top]
    mu = Xc.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    k = 5
    for name in Fm:
        Fm[name] = np.concatenate([Fm[name], ((raw[name][top] - mu) @ Vt[:k].T).astype(np.float32)])
    out = np.stack([Fm[n] for n in ctx_names(cfg)]).astype(np.float32)
    np.savez(wd(cfg) / "pca_meta.npz", top=top, mu=mu, Vt=Vt[:k],
             ctx_names=np.array(ctx_names(cfg)))
    np.save(wd(cfg) / "fingerprints.npy", out)
    print(f"[fp] {out.shape} for {ctx_names(cfg)}", flush=True)
    return out


def make_seq_table(group_key: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """连续行按 group_key 分组 → 切 N 行/序列。返回 (seq_rows (S,N) i32, -1=PAD, 组内首行索引)。"""
    starts = np.flatnonzero(np.r_[True, group_key[1:] != group_key[:-1]])
    counts = np.diff(np.r_[starts, len(group_key)])
    n_seq = int(np.ceil(counts / N).sum()) if len(counts) else 0
    seq_rows = np.full((n_seq, N), -1, np.int64)
    meta_row = np.empty(n_seq, np.int64)
    si = 0
    for st, ct in zip(starts.tolist(), counts.tolist()):
        for off in range(0, ct, N):
            n = min(N, ct - off)
            seq_rows[si, :n] = np.arange(st + off, st + off + n)
            meta_row[si] = st + off
            si += 1
    assert si == n_seq
    return seq_rows, meta_row


def _std_params(X: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = X[rows].astype(np.float64)
    mu = np.nanmean(S, axis=0)
    sd = np.nanstd(S, axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def _std_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    fin = np.isfinite(X)
    return np.where(fin, (np.where(fin, X, 0.0) - mu) / sd, 0.0).astype(np.float32)


def vocab_pos(vocab: np.ndarray, names: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(vocab, names)
    pos_c = np.clip(pos, 0, len(vocab) - 1)
    return np.where(vocab[pos_c] == names, pos_c, -1).astype(np.int64)


def cmd_build(cfg: dict) -> None:
    t0 = time.time()
    out = wd(cfg)
    vocab, mflags = load_vocab_and_modules(cfg)
    V = len(vocab)
    fp = build_fingerprints(cfg, mflags)

    z = np.load(ROOT / cfg["fusion_v35_dir"] / "feat_train.npz")
    names = [str(x) for x in z["names"]]
    gen_idx = np.array([i for i, nm in enumerate(names)
                        if nm in GEN_COLS or nm.startswith("sig_") or nm.startswith("mod_")])
    trg_idx = np.array([i for i, nm in enumerate(names) if nm in TRG_X_COLS])
    X, y, w = z["X"], z["y"], z["w"]
    ctx, gpos, tpos = z["ctx"].astype(np.int64), z["gpos"].astype(np.int64), z["tpos"].astype(np.int64)
    n_bad_t = int((tpos < 0).sum())
    print(f"[{time.time()-t0:.0f}s] train rows={len(y)} gen_dim={len(gen_idx)} "
          f"trg_dim={len(trg_idx)} tpos<0: {n_bad_t}", flush=True)

    rng = np.random.default_rng(cfg["train"]["seed"])
    samp = rng.choice(len(y), size=min(len(y) // 10, 3_000_000), replace=False)
    mu_g, sd_g = _std_params(X[:, gen_idx], samp)
    mu_t, sd_t = _std_params(X[:, trg_idx], samp)
    genF = _std_apply(X[:, gen_idx], mu_g, sd_g)
    trgF = np.concatenate([_std_apply(X[:, trg_idx], mu_t, sd_t),
                           mflags[:, np.maximum(tpos, 0)].T,
                           (tpos < 0)[:, None].astype(np.float32)], 1)
    del X, z

    order = np.lexsort((tpos, ctx))
    ctx_s, tpos_s, gpos_s = ctx[order], tpos[order], gpos[order]
    y_s = np.clip(y[order], -cfg["train"]["label_clip"], cfg["train"]["label_clip"])
    genF_s, trgF_s, w_s = genF[order], trgF[order].astype(np.float32), w[order]
    del genF, trgF, order
    N = cfg["model"]["seq_genes"]
    seq_rows, meta_row = make_seq_table(ctx_s * (V + 1) + np.maximum(tpos_s, 0), N)
    print(f"[{time.time()-t0:.0f}s] seqs={len(seq_rows)} (N={N})", flush=True)

    d = out / "train"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "genF.npy", genF_s)
    np.save(d / "trgF.npy", trgF_s)
    np.save(d / "y.npy", y_s.astype(np.float32))
    np.save(d / "w.npy", w_s.astype(np.float32))
    np.save(d / "ctx.npy", ctx_s.astype(np.int8))
    np.save(d / "gpos.npy", gpos_s.astype(np.int32))
    np.save(d / "tpos.npy", tpos_s.astype(np.int32))
    np.save(d / "seq_rows.npy", seq_rows.astype(np.int32))
    np.save(d / "seq_ctx.npy", ctx_s[meta_row].astype(np.int8))
    np.save(d / "seq_tpos.npy", tpos_s[meta_row].astype(np.int32))
    np.savez(d / "norm.npz", mu_g=mu_g, sd_g=sd_g, mu_t=mu_t, sd_t=sd_t)
    (d / "dims.json").write_text(json.dumps(
        {"d_gen": int(genF_s.shape[1]), "d_trg": int(trgF_s.shape[1]), "d_ctx": int(fp.shape[1]),
         "d_pre": 256, "V": V, "n_seqs": int(len(seq_rows)), "n_rows": int(len(y_s)),
         "tpos_neg": n_bad_t}))
    del genF_s, trgF_s, seq_rows
    print(f"[{time.time()-t0:.0f}s] train pack saved", flush=True)

    for src in ["bench_hepg2", "submit_A", "submit_B", "submit_C"]:
        f = ROOT / cfg["fusion_v35_dir"] / f"feat_{src}.npz"
        if not f.exists():
            print(f"[skip] {f} 不存在", flush=True)
            continue
        zs = np.load(f, allow_pickle=True)
        tgt, gen = zs["target"].astype(str), zs["gene"].astype(str)
        gp, tp = vocab_pos(vocab, gen), vocab_pos(vocab, tgt)
        cn = ctx_names(cfg).index("hepg2" if src == "bench_hepg2" else src.split("_")[1].lower())
        gf = _std_apply(zs["X"][:, gen_idx], mu_g, sd_g)
        tf = np.concatenate([_std_apply(zs["X"][:, trg_idx], mu_t, sd_t),
                             mflags[:, np.maximum(tp, 0)].T,
                             (tp < 0)[:, None].astype(np.float32)], 1).astype(np.float32)
        ord2 = np.argsort(tgt, kind="stable")           # 稳定排 → target 连续, 行序可还原
        key = ctx_names(cfg).index("hepg2" if src == "bench_hepg2"
                                   else src.split("_")[1].lower()) * (V + 1) + np.maximum(tp[ord2], 0)
        seq_rows2, meta2 = make_seq_table(key, N)
        d2 = out / src
        d2.mkdir(parents=True, exist_ok=True)
        np.save(d2 / "genF.npy", gf[ord2])
        np.save(d2 / "trgF.npy", tf[ord2])
        np.save(d2 / "gpos.npy", gp[ord2].astype(np.int32))
        np.save(d2 / "tpos.npy", tp[ord2].astype(np.int32))
        np.save(d2 / "seq_rows.npy", seq_rows2.astype(np.int32))
        np.save(d2 / "seq_tpos.npy", tp[ord2][meta2].astype(np.int32))
        np.savez(d2 / "order.npz", ord=ord2, meta_row=meta2,
                 target=tgt[ord2].astype("U"), gene=gen[ord2].astype("U"))
        print(f"[{time.time()-t0:.0f}s] {src}: rows={len(tgt)} seqs={len(seq_rows2)} ctx_id={cn}",
              flush=True)
    print("build complete", flush=True)


# ------------------------------------------------------------------- model
def make_model(cfg: dict, dims: dict, fp: np.ndarray, G: np.ndarray, device: str):
    import torch
    import torch.nn as nn

    class BizardNet(nn.Module):
        """[CTX]+[TRG]+[GEN]×N → 6L pre-LN encoder → 每基因 logFC。"""

        def __init__(self):
            super().__init__()
            dm = cfg["model"]["d_model"]
            self.register_buffer("F", torch.from_numpy(fp))
            self.register_buffer("G", torch.from_numpy(G))
            self.use_gene_emb = cfg["model"].get("use_gene_emb", True)
            self.residual_gate = cfg["model"].get("residual_gate", False)
            # v1.1: 训练时以 p 概率把整条上下文指纹换成可学 null 向量，
            # 阻断指纹记忆、强迫跨上下文路由 (iter013 方向 a)
            self.fp_drop = cfg["train"].get("context_dropout", 0.0)
            if self.fp_drop > 0:
                self.fp_null = nn.Parameter(torch.zeros(dims["d_ctx"]))
            self.type_emb = nn.Embedding(3, dm)
            if self.use_gene_emb:
                self.gene_emb = nn.Embedding(dims["V"] + 1, dm)
            self.ctx_proj = nn.Linear(dims["d_ctx"], dm)
            self.trg_proj = nn.Linear(dm + dims["d_pre"] + dims["d_trg"], dm)
            self.gen_proj = nn.Linear(dm + dims["d_pre"] + dims["d_gen"], dm)
            self.in_ln = nn.LayerNorm(dm)
            layer = nn.TransformerEncoderLayer(
                dm, cfg["model"]["n_heads"], cfg["model"]["d_ff"],
                dropout=cfg["model"]["dropout"], activation="gelu",
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg["model"]["n_layers"])
            self.head = nn.Linear(dm, 1)
            self.skip_head = cfg["train"].get("skip_head", False)
            if self.skip_head:                      # 零初始化直连: 起步=fusion-MLP 式基线
                self.skip = nn.Linear(dims["d_gen"], 1)
                nn.init.zeros_(self.skip.weight)
                nn.init.zeros_(self.skip.bias)
            if self.residual_gate:                  # ReZero: transformer 必须挣得贡献
                self.gate = nn.Parameter(torch.zeros(1))
            if self.use_gene_emb:
                nn.init.normal_(self.gene_emb.weight, std=0.02)

        def forward(self, ctx_id, tpos, gpos, gen_x, trg_x, pad):
            B = gpos.shape[0]
            ctx_feat = self.F[ctx_id]
            if self.training and self.fp_drop > 0:
                keep = (torch.rand(B, 1, device=ctx_feat.device) >= self.fp_drop
                        ).to(ctx_feat.dtype)
                ctx_feat = ctx_feat * keep + self.fp_null * (1.0 - keep)
            ctx_t = self.ctx_proj(ctx_feat) + self.type_emb.weight[0]
            dm = ctx_t.shape[-1]
            z_t = torch.zeros(B, dm, device=ctx_t.device)
            ge_t = self.gene_emb(tpos + 1) if self.use_gene_emb else z_t
            ge_g = self.gene_emb(gpos + 1) if self.use_gene_emb \
                else z_t.unsqueeze(1).expand(-1, gpos.shape[1], -1)
            trg_t = self.trg_proj(torch.cat(
                [ge_t, self.G[tpos + 1], trg_x], -1)) + self.type_emb.weight[1]
            gen_t = self.gen_proj(torch.cat(
                [ge_g, self.G[gpos + 1], gen_x], -1)) + self.type_emb.weight[2]
            x = self.in_ln(torch.cat([ctx_t[:, None], trg_t[:, None], gen_t], 1))
            kp = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), pad], 1)
            x = self.encoder(x, src_key_padding_mask=kp)
            out = self.head(x[:, 2:]).squeeze(-1)
            if self.residual_gate:
                out = self.gate * out
            if self.skip_head:
                out = out + self.skip(gen_x).squeeze(-1)
            return out

    return BizardNet().to(device)


def load_ckpt_net(cfg: dict, arm: str, device: str):
    import torch
    out = wd(cfg)
    ck = torch.load(out / f"bizardnet_{arm}.pt", map_location=device, weights_only=False)
    net = make_model(cfg, ck["dims"], np.load(out / "fingerprints.npy"), pre_emb(cfg), device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


# ------------------------------------------------------------------- train
def cmd_train(cfg: dict, arm: str) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    out = wd(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, mc = cfg["train"], cfg["model"]
    d = out / "train"
    dims = json.loads((d / "dims.json").read_text())
    fp = np.load(out / "fingerprints.npy")
    G = pre_emb(cfg)

    seq_rows = np.load(d / "seq_rows.npy")
    seq_ctx = np.load(d / "seq_ctx.npy").astype(np.int64)
    seq_tpos = np.load(d / "seq_tpos.npy").astype(np.int64)
    hold = ctx_names(cfg).index(cfg["loco_holdout"])
    if arm == "loco":
        tr_s = np.flatnonzero(seq_ctx != hold)
        va_s = np.flatnonzero(seq_ctx == hold)
    else:
        rng = np.random.default_rng(tc["seed"])
        u = rng.random(len(seq_ctx))
        tr_s, va_s = np.flatnonzero(u >= 0.02), np.flatnonzero(u < 0.02)
    print(f"[{time.time()-t0:.0f}s] arm={arm} seqs train={len(tr_s)} val={len(va_s)} "
          f"rows_train={int((seq_rows[tr_s] >= 0).sum())}", flush=True)

    genF_mm = np.load(d / "genF.npy", mmap_mode="r")
    trgF_mm = np.load(d / "trgF.npy", mmap_mode="r")
    genF_g = torch.empty(genF_mm.shape, dtype=torch.float32, device=device)
    trgF_g = torch.empty(trgF_mm.shape, dtype=torch.float32, device=device)
    for s in range(0, genF_mm.shape[0], 2_000_000):
        genF_g[s:s + 2_000_000] = torch.from_numpy(np.asarray(genF_mm[s:s + 2_000_000])).to(device)
    for s in range(0, trgF_mm.shape[0], 2_000_000):
        trgF_g[s:s + 2_000_000] = torch.from_numpy(np.asarray(trgF_mm[s:s + 2_000_000])).to(device)
    del genF_mm, trgF_mm
    y_g = torch.from_numpy(np.load(d / "y.npy")).to(device)
    w_g = torch.from_numpy(np.load(d / "w.npy")).to(device)
    gpos_g = torch.from_numpy(np.load(d / "gpos.npy").astype(np.int64)).to(device)
    seq_rows_g = torch.from_numpy(seq_rows).to(device)
    seq_ctx_g = torch.from_numpy(seq_ctx).to(device)
    seq_tpos_g = torch.from_numpy(seq_tpos).to(device)
    tr_s_g = torch.from_numpy(tr_s).to(device)
    va_s_g = torch.from_numpy(va_s).to(device)
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] GPU ready mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
          flush=True)

    torch.manual_seed(tc["seed"])
    net = make_model(cfg, dims, fp, G, device)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    bs, N = tc["batch_seqs"], mc["seq_genes"]
    steps_per_ep = int(np.ceil(len(tr_s) / bs))
    n_steps = steps_per_ep * tc["max_epochs"]
    warm = min(tc["warmup_steps"], n_steps // 5)

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(n_steps - warm, 1)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))

    huber = torch.nn.HuberLoss(delta=tc["huber_beta"], reduction="none")
    best, best_state, bad, hist, step = 1e9, None, 0, [], 0
    for ep in range(tc["max_epochs"]):
        net.train()
        perm = tr_s_g[torch.randperm(len(tr_s), device=device)]
        tot = wtot = 0.0
        for s in range(0, len(perm), bs):
            ids = perm[s:s + bs]
            rows = seq_rows_g[ids]
            pad = rows < 0
            rc = rows.clamp_min(0).long()
            for pg in opt.param_groups:
                pg["lr"] = tc["lr"] * lr_at(step)
            step += 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                outp = net(seq_ctx_g[ids], seq_tpos_g[ids].clamp_min(0), gpos_g[rc],
                           genF_g[rc], trgF_g[rc[:, 0]], pad)
                lv = huber(outp.float(), y_g[rc])
                m = (~pad).float()
                wb = w_g[rc]
                loss = (lv * wb * m).sum() / ((wb * m).sum() + 1e-9)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(net.parameters(), tc["grad_clip"])
            opt.step()
            tot += float(loss.detach()) * float((wb * m).sum())
            wtot += float((wb * m).sum())
        net.eval()
        vl = vn = 0.0
        with torch.no_grad():
            for s in range(0, len(va_s_g), bs):
                ids = va_s_g[s:s + bs]
                rows = seq_rows_g[ids]
                pad = rows < 0
                rc = rows.clamp_min(0).long()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    outp = net(seq_ctx_g[ids], seq_tpos_g[ids].clamp_min(0), gpos_g[rc],
                               genF_g[rc], trgF_g[rc[:, 0]], pad)
                    lv = huber(outp.float(), y_g[rc])
                m = (~pad).float()
                vl += float((lv * m).sum())
                vn += float(m.sum())
        vl /= max(vn, 1)
        hist.append({"epoch": ep, "train": tot / max(wtot, 1), "val": vl,
                     "lr": tc["lr"] * lr_at(step), "sec": round(time.time() - t0, 1)})
        print(f"  ep{ep}: train={tot/max(wtot,1):.4f} val={vl:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save({"state": net.state_dict(), "dims": dims, "arm": arm, "n_params": int(npar)},
               out / f"bizardnet_{arm}.pt")
    (out / f"bizardnet_{arm}_hist.json").write_text(json.dumps(
        {"hist": hist, "best_val": best, "n_params": int(npar), "arm": arm,
         "train_seqs": int(len(tr_s)), "val_seqs": int(len(va_s)),
         "train_rows": int((seq_rows[tr_s] >= 0).sum())}))
    print(f"train[{arm}] complete: best_val={best:.4f} params={npar} ({time.time()-t0:.0f}s)",
          flush=True)


# ----------------------------------------------------------------- predict
def cmd_predict(cfg: dict, src: str, arm: str, clip: float) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    out = wd(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = load_ckpt_net(cfg, arm, device)
    ds = out / src
    genF_mm = np.load(ds / "genF.npy", mmap_mode="r")
    trgF_mm = np.load(ds / "trgF.npy", mmap_mode="r")
    genF_g = torch.empty(genF_mm.shape, dtype=torch.float32, device=device)
    trgF_g = torch.empty(trgF_mm.shape, dtype=torch.float32, device=device)
    for s in range(0, genF_mm.shape[0], 2_000_000):
        genF_g[s:s + 2_000_000] = torch.from_numpy(np.asarray(genF_mm[s:s + 2_000_000])).to(device)
    for s in range(0, trgF_mm.shape[0], 2_000_000):
        trgF_g[s:s + 2_000_000] = torch.from_numpy(np.asarray(trgF_mm[s:s + 2_000_000])).to(device)
    gpos_g = torch.from_numpy(np.load(ds / "gpos.npy").astype(np.int64)).to(device)
    ctx_id = ctx_names(cfg).index("hepg2" if src == "bench_hepg2" else src.split("_")[1].lower())
    seq_rows = torch.from_numpy(np.load(ds / "seq_rows.npy")).to(device)
    seq_tpos = torch.from_numpy(np.load(ds / "seq_tpos.npy").astype(np.int64)).to(device)
    preds = torch.full((genF_mm.shape[0],), 0.0, dtype=torch.float32, device=device)
    bs = 256
    with torch.no_grad():
        for s in range(0, len(seq_rows), bs):
            rows = seq_rows[s:s + bs]
            pad = rows < 0
            rc = rows.clamp_min(0).long()
            ctx_b = torch.full((rc.shape[0],), ctx_id, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                outp = net(ctx_b, seq_tpos[s:s + bs].clamp_min(0), gpos_g[rc],
                           genF_g[rc], trgF_g[rc[:, 0]], pad)
            valid = ~pad
            preds[rc[valid]] = outp.float()[valid]
    p = preds.cpu().numpy()
    ord2 = np.load(ds / "order.npz", allow_pickle=True)["ord"]
    orig = np.empty(len(p), np.float32)
    orig[ord2] = p
    orig = np.clip(np.nan_to_num(orig, nan=0.0), -clip, clip).astype(np.float32)
    tag = f"bizardnet_{arm}_{src}"
    np.save(out / f"pred_{tag}.npy", orig)
    meta = np.load(ds / "order.npz", allow_pickle=True)
    pd.DataFrame({"target_gene": meta["target"].astype(str),
                  "gene": meta["gene"].astype(str), "delta": orig}).to_parquet(
        out / f"pred_{tag}.parquet")
    print(f"predict[{tag}]: n={len(orig)} mean={orig.mean():.4f} sd={orig.std():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------- eval-loco
def cmd_eval(cfg: dict, arm: str = "loco", max_rows: int = 2_000_000) -> None:
    """LOCO 模型在 hepg2 训练行上的 proxy 指标 (spearman_med/recall@100/sign_acc)。"""
    import torch
    sys.path.insert(0, str(ROOT))
    from src.models.l15_fusion.eval_loco import metrics
    out = wd(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(ROOT / cfg["fusion_v35_dir"] / "feat_train.npz")
    names = [str(x) for x in z["names"]]
    gen_idx = np.array([i for i, nm in enumerate(names)
                        if nm in GEN_COLS or nm.startswith("sig_") or nm.startswith("mod_")])
    trg_idx = np.array([i for i, nm in enumerate(names) if nm in TRG_X_COLS])
    hold = ctx_names(cfg).index(cfg["loco_holdout"])
    idx = np.flatnonzero(z["ctx"].astype(np.int64) == hold)
    if len(idx) > max_rows:
        idx = np.random.default_rng(0).choice(idx, max_rows, replace=False)
    y, tpos, gpos = z["y"][idx], z["tpos"][idx].astype(np.int64), z["gpos"].astype(np.int64)[idx]
    norm = np.load(out / "train/norm.npz")
    genF = torch.from_numpy(_std_apply(z["X"][idx][:, gen_idx], norm["mu_g"], norm["sd_g"])).to(device)
    vocab, mflags = load_vocab_and_modules(cfg)
    trgF = torch.from_numpy(np.concatenate(
        [_std_apply(z["X"][idx][:, trg_idx], norm["mu_t"], norm["sd_t"]),
         mflags[:, np.maximum(tpos, 0)].T,
         (tpos < 0)[:, None].astype(np.float32)], 1)).to(device)
    net, _ = load_ckpt_net(cfg, arm, device)
    tg = torch.from_numpy(np.maximum(gpos, 0)).to(device)
    tt = torch.from_numpy(np.maximum(tpos, 0)).to(device)
    cid = torch.full((len(idx),), hold, dtype=torch.long, device=device)
    preds = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            sl = slice(s, min(s + 8192, len(idx)))
            B = sl.stop - sl.start
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                o = net(cid[sl], tt[sl], tg[sl, None], genF[sl, None], trgF[sl],
                        torch.zeros(B, 1, dtype=torch.bool, device=device))
            preds[sl] = o[:, 0].float().cpu().numpy()
    preds = np.clip(preds, -cfg["predict"]["clip"], cfg["predict"]["clip"])
    tgt_names = np.array(json.load(open(ROOT / cfg["vocab_json"])))[np.maximum(tpos, 0)]
    res = metrics(y, preds, tgt_names)
    res["n_rows"] = int(len(idx))
    (out / f"eval_loco_hepg2_rows_{arm}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "predict", "eval-loco"])
    ap.add_argument("--arm", default="loco", choices=["loco", "full"])
    ap.add_argument("--src", default="bench_hepg2")
    ap.add_argument("--clip", type=float, default=None)
    args = ap.parse_args()
    cfg = load_cfg()
    if args.cmd == "build":
        cmd_build(cfg)
    elif args.cmd == "train":
        cmd_train(cfg, args.arm)
    elif args.cmd == "predict":
        cmd_predict(cfg, args.src, args.arm,
                    args.clip if args.clip is not None else cfg["predict"]["clip"])
    else:
        cmd_eval(cfg, args.arm)


if __name__ == "__main__":
    main()
