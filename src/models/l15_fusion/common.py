"""L1.5-fusion 共享工具：统一基因词表、签名 CSR 存储、逐行特征构建。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
CFG_PATH = ROOT / "configs/l15_fusion/v0.yaml"
ALL_CONTEXTS = ["gwps", "k562_ess", "rpe1", "jurkat", "hepg2"]


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def work_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["workdir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["outdir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def sig_file(context: str) -> Path:
    f = ROOT / f"data/interim/signatures/{context}_all.parquet"
    return f if f.exists() else ROOT / "data/interim/signatures/gwps.parquet"


# ---------------------------------------------------------------- gene vocab
def build_vocab(cfg: dict) -> pd.Index:
    """统一基因词表 = 全上下文签名面板并集（缓存）。"""
    p = work_dir(cfg) / "vocab.json"
    if p.exists():
        return pd.Index(json.loads(p.read_text()))
    genes = set()
    for c in ALL_CONTEXTS:
        s = pd.read_parquet(sig_file(c), columns=["gene"])
        genes.update(s["gene"].unique())
    vocab = pd.Index(sorted(genes))
    p.write_text(json.dumps(vocab.tolist()))
    return vocab


# ------------------------------------------------------- signature CSR store
def build_sig_store(context: str, vocab: pd.Index, cache_dir: Path) -> dict:
    """签名 → 向量化可查 store：keys = target_row*V + gene_pos（全局有序 int64）。

    同一 (target, gene) 多行取均值；gene 不在词表丢弃。缓存 npz。
    """
    p = cache_dir / f"sigstore_{context}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    df = pd.read_parquet(sig_file(context))
    df = df[df["gene"].isin(vocab)]
    g = df.groupby(["target_gene", "gene"], sort=False)["logfc"].mean().reset_index()
    tnames_sorted = pd.Index(sorted(g["target_gene"].unique()))
    trow = tnames_sorted.get_indexer(g["target_gene"])
    gpos = vocab.get_indexer(g["gene"])
    # 同 key 去重均值：先排序 (trow, gpos) 后相邻合并
    order = np.lexsort((gpos, trow))
    trow, gpos = trow[order], gpos[order]
    vals = g["logfc"].to_numpy(np.float32)[order]
    keys = trow.astype(np.int64) * len(vocab) + gpos.astype(np.int64)
    uk, uidx, ucnt = np.unique(keys, return_index=True, return_counts=True)
    data = vals[uidx]
    dup = ucnt > 1  # 理论上 groupby 已去重，保险
    if dup.any():
        data = data.copy()
    store = {"keys": uk.astype(np.int64), "data": data.astype(np.float32),
             "target_names": np.array(tnames_sorted.tolist(), dtype="U"),
             "n_vocab": np.int64(len(vocab))}
    np.savez(p, **store)
    return store


def sig_lookup(store: dict, targets: np.ndarray, gpos: np.ndarray) -> np.ndarray:
    """批量查 (target, gene_pos) 签名值；target 缺失或 gene 缺失 → NaN。全向量化。"""
    V = int(store["n_vocab"])
    tn = store["target_names"]
    rows = np.searchsorted(tn, targets.astype(tn.dtype))
    rows_c = np.clip(rows, 0, len(tn) - 1)
    found_t = (tn[rows_c] == targets.astype(tn.dtype)) & (gpos >= 0)
    q = rows_c.astype(np.int64) * V + np.maximum(gpos, 0).astype(np.int64)
    pos = np.searchsorted(store["keys"], q)
    pos_c = np.clip(pos, 0, len(store["keys"]) - 1)
    hit = found_t & (store["keys"][pos_c] == q)
    out = np.where(hit, store["data"][pos_c], np.nan).astype(np.float32)
    return out


# ------------------------------------------------------------------- basals
def load_basal(context: str, vocab: pd.Index, cache_dir: Path) -> np.ndarray:
    """NTC 基线表达：raw counts 均值 → CPM → log1p（词表对齐，缺失=0）。"""
    p = cache_dir / f"basal_{context}.npy"
    if p.exists():
        return np.load(p)
    import anndata as ad
    import scipy.sparse as sp
    cfg = load_cfg()
    a = ad.read_h5ad(ROOT / cfg["basal_h5ad"][context], backed="r")
    mask = np.asarray(a.obs[cfg["pert_col"]].astype(str) == cfg["control"])
    idx = np.where(mask)[0]
    acc = np.zeros(a.n_vars, np.float64)
    CH = 20000
    for s in range(0, len(idx), CH):
        X = sp.csr_matrix(a.X[idx[s:s + CH]])
        acc += np.asarray(X.sum(axis=0)).ravel()
    a.file.close()
    mean_counts = acc / max(len(idx), 1)
    cpm = mean_counts / max(mean_counts.sum(), 1e-9) * 1e6
    lg = np.log1p(cpm).astype(np.float32)
    var_pos = vocab.get_indexer(a.var_names.astype(str))
    out = np.zeros(len(vocab), np.float32)
    ok = var_pos >= 0
    out[var_pos[ok]] = lg[ok]
    np.save(p, out)
    return out


# ------------------------------------------------------------------ modules
def module_flags(vocab: pd.Index, cfg: dict) -> np.ndarray:
    """6 个模块 one-hot（j 的标志位），形状 (n_mod, len(vocab))。"""
    ms = cfg["features"]["module_sets"]
    names = list(ms.keys())
    flags = np.zeros((len(names), len(vocab)), np.float32)
    genes = np.asarray(vocab.tolist())
    for i, m in enumerate(names):
        spec = ms[m]
        if isinstance(spec, list):
            spec = {"genes": spec}
        hit = np.zeros(len(vocab), bool)
        for pre in spec.get("prefix", []):
            hit |= np.char.startswith(genes, pre)
        for gg in spec.get("genes", []):
            hit |= (genes == gg)
        flags[i] = hit
    return flags


# --------------------------------------------------------------- task tables
def build_task_table(context: str, cfg: dict, vocab: pd.Index, cache_dir: Path) -> pd.DataFrame:
    """上下文 c 的训练任务表：每 (c,g) 采样 top-K |logfc| + 随机 K。"""
    p = cache_dir / f"tasks_{context}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_parquet(sig_file(context))
    df = df[df["gene"].isin(vocab)]
    seed = cfg["sampling"]["seed"] + (hash(context) % 997)
    rng = np.random.default_rng(seed)
    if context == "gwps" and cfg["sampling"].get("gwps_target_cap"):
        tg = pd.Index(df["target_gene"].unique())
        keep = rng.choice(len(tg), size=min(cfg["sampling"]["gwps_target_cap"], len(tg)),
                          replace=False)
        df = df[df["target_gene"].isin(set(tg[keep].tolist()))]
    df = df.sample(frac=1.0, random_state=seed)  # 打散平局
    df["_abs"] = df["logfc"].abs()
    df = df.sort_values(["target_gene", "_abs"], ascending=[True, False], kind="mergesort")
    grp = df.groupby("target_gene", sort=False)
    rank = grp.cumcount().to_numpy()
    size = grp["target_gene"].transform("size").to_numpy()
    topk, randk = cfg["sampling"]["top_k"], cfg["sampling"]["rand_k"]
    is_top = rank < topk
    rest_ok = (rank >= topk) & (size > topk)
    u = rng.random(len(df))
    take_prob = np.where(size > topk, randk / np.maximum(size - topk, 1), 0.0)
    keep = is_top | (rest_ok & (u < take_prob))
    out = df.loc[keep, ["target_gene", "gene", "logfc"]].reset_index(drop=True)
    out.to_parquet(p)
    del df
    return out


# ------------------------------------------------------------------ features
class FeatureBuilder:
    """逐批构建特征矩阵（float32, NaN=缺签名）。输入: 行上下文 + (target, gene) 数组。"""

    def __init__(self, cfg: dict, vocab: pd.Index, cache_dir: Path):
        self.cfg = cfg
        self.vocab = vocab
        self.all_sources = list(cfg["train_contexts"])
        self._stores: dict[str, dict] = {}
        self.basal = {c: load_basal(c, vocab, cache_dir)
                      for c in cfg["train_contexts"] + ["hepg2"]}
        self.modules = module_flags(vocab, cfg)

    def _store(self, ctx: str) -> dict:
        if ctx not in self._stores:
            self._stores[ctx] = build_sig_store(ctx, self.vocab, work_dir(self.cfg))
        return self._stores[ctx]

    def names(self, row_ctx: str | None = None) -> list[str]:
        # 固定布局：4 个签名源列（行自身上下文→NaN），与行上下文无关
        n = (["cross_mean", "cross_std", "cross_maxabs", "basal_j", "basal_j_zero",
              "basal_g", "basal_g_zero"] + [f"sig_{s}" for s in self.all_sources]
             + [f"mod_{m}" for m in self.cfg["features"]["module_sets"].keys()]
             + ["x_sign_mean_basal", "x_absmean_basal"])
        return n

    def transform(self, row_ctx: str, targets: np.ndarray, genes: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets, dtype="U")
        genes = np.asarray(genes, dtype="U")
        gpos = self.vocab.get_indexer(pd.Index(genes))
        tpos = self.vocab.get_indexer(pd.Index(targets))
        src = list(self.all_sources)
        sig = {s: sig_lookup(self._store(s), targets, gpos) for s in src}
        for s in src:  # 行自身上下文的签名是标签的镜像 → 置 NaN 防泄漏
            if s == row_ctx:
                sig[s] = np.full(len(gpos), np.nan, np.float32)
        S = np.stack([sig[s] for s in src])
        with np.errstate(all="ignore"):
            mean = np.nanmean(S, axis=0)
            std = np.nanstd(S, axis=0)
            mx = np.nanmax(np.abs(S), axis=0)
        npres = np.sum(np.isfinite(S), axis=0)
        mean = np.where(npres > 0, mean, 0.0).astype(np.float32)
        std = np.where(npres > 1, std, 0.0).astype(np.float32)
        mx = np.where(npres > 0, mx, 0.0).astype(np.float32)
        bj_full = self.basal[row_ctx]
        bj = np.where(gpos >= 0, bj_full[np.maximum(gpos, 0)], 0.0)
        bz = (gpos < 0) | (bj <= 0.0)
        bj = np.where(bz, 0.0, bj).astype(np.float32)
        bg = np.where(tpos >= 0, bj_full[np.maximum(tpos, 0)], 0.0)
        bgz = (tpos < 0) | (bg <= 0.0)
        bg = np.where(bgz, 0.0, bg).astype(np.float32)
        cols = [mean, std, mx, bj, bz.astype(np.float32), bg, bgz.astype(np.float32)]
        cols += [sig[s] for s in src]
        ok = gpos >= 0
        mods = self.modules[:, np.maximum(gpos, 0)] * ok[None, :]
        cols += [mods[i] for i in range(self.modules.shape[0])]
        cols += [(np.sign(mean) * bj).astype(np.float32),
                 (np.abs(mean) * bj).astype(np.float32)]
        return np.stack(cols, axis=1).astype(np.float32)
