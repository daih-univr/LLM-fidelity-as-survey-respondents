from __future__ import annotations

import argparse
from html import parser
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, norm
import copy
import re
import shutil















DATASETS: Dict[str, str] = {
    "human": "data/humans/features",
    "gemma": "data/machines_gemma/features",
    "gpt-oss": "data/machines_gpt-oss/features",
    "llama": "data/machines_llama/features",
    "qwen": "data/machines_qwen/features",
    "gpt": "data/machines_gpt/features"
}


OUTDIR = "data/analysis_outputs"



SEED = 42


N_SPLIT_HALF = 400          
N_PERMUTATIONS = 800        
N_BOOTSTRAP = 500           


KEYNESS_MIN_TOTAL_COUNT = 5
KEYNESS_ALPHA = 0.01  


WORD_QS = [f"q{i}" for i in range(3, 17)]
OPEN_QS = ["q1", "q2"]
LIKERT_QS = ["q17", "q18", "q19"]






SYSTEM_ORDER: List[str] = ["human", "gemma", "gpt-oss", "llama", "qwen", "gpt"]


def ordered_systems(packs: Dict[str, "DatasetPack"]) -> List[str]:

    return [s for s in SYSTEM_ORDER if s in packs]


def ordered_models(packs: Dict[str, "DatasetPack"]) -> List[str]:
    return [s for s in ordered_systems(packs) if s != "human"]


def order_df_by_system(df: pd.DataFrame, packs: Dict[str, "DatasetPack"], col: str = "dataset") -> pd.DataFrame:
    cats = ordered_systems(packs)
    out = df.copy()
    if col in out.columns:
        out[col] = pd.Categorical(out[col], categories=cats, ordered=True)
        out = out.sort_values(col)
    return out






def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.array([1.0 if (pv is None or np.isnan(pv)) else float(pv) for pv in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def median_iqr(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    q1 = np.quantile(x, 0.25)
    q2 = np.quantile(x, 0.50)
    q3 = np.quantile(x, 0.75)
    return (q2, q1, q3)


def q95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, 0.95)) if len(x) else np.nan


def _apply_fdr_inplace(rows: List[Dict[str, Any]], p_key: str = "p", out_key: str = "p_adj") -> None:
    adj = bh_fdr([r.get(p_key, np.nan) for r in rows]) if rows else []
    for r, pa in zip(rows, adj):
        r[out_key] = pa






def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _compute_word_count_from_text(series: pd.Series) -> pd.Series:
    
    s = series.fillna("").astype(str)
    return s.apply(lambda x: len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ]+\b", x)))

def _open_wordcount_series(dfq: pd.DataFrame) -> pd.Series:
    col = _find_first_column(dfq, [
        "n_words", "num_words", "word_count", "n_word", "words",
        "len_words", "length_words", "n_tokens", "token_count"
    ])
    if col is not None:
        return pd.to_numeric(dfq[col], errors="coerce")

    
    tcol = _find_first_column(dfq, [
        "text", "response", "answer", "open_text", "raw_text", "final_text"
    ])
    if tcol is not None:
        return _compute_word_count_from_text(dfq[tcol])

    
    return pd.Series(np.nan, index=dfq.index)

def _wordassoc_answer_key_series(dfq: pd.DataFrame) -> pd.Series:
    col = _find_first_column(dfq, [
        "entropy_key", "normalized_raw", "normalized", "lemma", "token", "answer"
    ])
    if col is None:
        col = _find_first_column(dfq, ["text", "response", "raw"])
    if col is None:
        return pd.Series("", index=dfq.index)

    s = dfq[col].fillna("").astype(str).str.strip()
    
    s = s.where(s != "", np.nan)
    return s






@dataclass(frozen=True)
class CachePolicy:
    recompute: bool = False   
    plot: bool = True         
    compute: bool = True      


def cache_ok(path: Path, cache: CachePolicy) -> bool:
    return (not cache.recompute) and path.exists()


def save_npy(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    np.save(path, arr)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def load_or_compute_df(path: Path, cache: CachePolicy, compute_fn):
    if cache_ok(path, cache):
        return pd.read_csv(path)
    df = compute_fn()
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return df


def load_or_compute_npy(path: Path, cache: CachePolicy, compute_fn):
    if cache_ok(path, cache):
        return load_npy(path)
    arr = compute_fn()
    save_npy(arr, path)
    return arr






@dataclass
class DatasetPack:
    label: str
    path: Path
    profiles: pd.DataFrame
    open_long: pd.DataFrame
    word_long: pd.DataFrame
    likert: pd.DataFrame
    key_open: pd.DataFrame
    key_word: pd.DataFrame
    meta: Dict[str, Any]
    
    emb_open_q1: Optional[np.ndarray] = None
    emb_open_q2: Optional[np.ndarray] = None
    map_open_q1: Optional[pd.DataFrame] = None
    map_open_q2: Optional[pd.DataFrame] = None
    emb_wordassoc: Optional[np.ndarray] = None
    map_wordassoc: Optional[pd.DataFrame] = None


def load_pack(label: str, dirpath: str) -> DatasetPack:
    p = Path(dirpath)
    required = [
        "profiles.csv",
        "open_long.csv",
        "wordassoc_long.csv",
        "likert.csv",
        "keyness_tokens_open.csv",
        "keyness_tokens_wordassoc.csv",
        "metadata.json",
    ]
    for f in required:
        if not (p / f).exists():
            raise FileNotFoundError(f"Missing {f} in {p}")

    pack = DatasetPack(
        label=label,
        path=p,
        profiles=pd.read_csv(p / "profiles.csv"),
        open_long=pd.read_csv(p / "open_long.csv"),
        word_long=pd.read_csv(p / "wordassoc_long.csv"),
        likert=pd.read_csv(p / "likert.csv"),
        key_open=pd.read_csv(p / "keyness_tokens_open.csv"),
        key_word=pd.read_csv(p / "keyness_tokens_wordassoc.csv"),
        meta=json.loads((p / "metadata.json").read_text(encoding="utf-8")),
    )

    
    if (p / "embeddings_open_q1.npy").exists() and (p / "map_open_q1.csv").exists():
        pack.emb_open_q1 = np.load(p / "embeddings_open_q1.npy")
        pack.map_open_q1 = pd.read_csv(p / "map_open_q1.csv")
    if (p / "embeddings_open_q2.npy").exists() and (p / "map_open_q2.csv").exists():
        pack.emb_open_q2 = np.load(p / "embeddings_open_q2.npy")
        pack.map_open_q2 = pd.read_csv(p / "map_open_q2.csv")
    if (p / "embeddings_wordassoc.npy").exists() and (p / "map_wordassoc.csv").exists():
        pack.emb_wordassoc = np.load(p / "embeddings_wordassoc.npy")
        pack.map_wordassoc = pd.read_csv(p / "map_wordassoc.csv")

    return pack

def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def add_subgroup_bins_to_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    df = profiles.copy()

    
    if "ruolo" in df.columns:
        df["ruolo_bin"] = df["ruolo"].astype(str).str.strip().str.lower()
    else:
        df["ruolo_bin"] = np.nan

    
    
    exp_col = "anni_esperienza"
    if exp_col in df.columns:
        x = pd.to_numeric(df[exp_col], errors="coerce")
        
        df["esperienza_bin"] = pd.cut(
            x,
            bins=[-0.5, 4.5, 14.5, np.inf],
            labels=["0-4", "5-14", "15+"],
            include_lowest=True,
        ).astype(str)
        df.loc[x.isna(), "esperienza_bin"] = np.nan
    else:
        df["esperienza_bin"] = np.nan

    
    if "grado" in df.columns:
        df["grado_bin"] = df["grado"].astype(str).str.strip().str.lower()
    else:
        df["grado_bin"] = np.nan

    return df


def respondent_ids_for_bin(packs: Dict[str, DatasetPack], bin_col: str, bin_value: str) -> np.ndarray:
    hprof = add_subgroup_bins_to_profiles(packs["human"].profiles)
    sub = hprof[hprof[bin_col].astype(str) == str(bin_value)]
    ids = pd.to_numeric(sub["respondent_id"], errors="coerce").dropna().astype(int).values
    return np.unique(ids)


def filter_pack_by_respondent_ids(pack: DatasetPack, ids: np.ndarray) -> DatasetPack:
    ids = np.asarray(ids)
    out = copy.copy(pack)

    
    if "respondent_id" in pack.profiles.columns:
        out.profiles = pack.profiles[pack.profiles["respondent_id"].isin(ids)].copy()

    
    for attr in ["open_long", "word_long", "likert", "key_open", "key_word"]:
        df = getattr(pack, attr)
        if isinstance(df, pd.DataFrame) and ("respondent_id" in df.columns):
            setattr(out, attr, df[df["respondent_id"].isin(ids)].copy())
        else:
            setattr(out, attr, df)

    
    def _filter_emb(emb: Optional[np.ndarray], mp: Optional[pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        if emb is None or mp is None or ("respondent_id" not in mp.columns):
            return emb, mp
        mask = mp["respondent_id"].isin(ids).values
        if mask.sum() < 2:
            
            return emb[mask], mp.loc[mask].copy()
        return emb[mask], mp.loc[mask].copy()

    out.emb_open_q1, out.map_open_q1 = _filter_emb(pack.emb_open_q1, pack.map_open_q1)
    out.emb_open_q2, out.map_open_q2 = _filter_emb(pack.emb_open_q2, pack.map_open_q2)
    out.emb_wordassoc, out.map_wordassoc = _filter_emb(pack.emb_wordassoc, pack.map_wordassoc)

    return out


def filter_packs_by_respondent_ids(
    packs: Dict[str, DatasetPack],
    ids: np.ndarray,
) -> Dict[str, DatasetPack]:
    idset = set(pd.Series(ids).dropna().tolist())

    def _filt_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "respondent_id" not in df.columns:
            return df
        return df[df["respondent_id"].isin(idset)].copy()

    def _filt_keyness_df(df: pd.DataFrame) -> pd.DataFrame:
        
        if df is None or df.empty:
            return df
        if "respondent_id" not in df.columns:
            return df
        return df[df["respondent_id"].isin(idset)].copy()

    def _filt_emb(emb: Optional[np.ndarray], mp: Optional[pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        if emb is None or mp is None or mp.empty:
            return emb, mp
        if "respondent_id" not in mp.columns:
            return emb, mp
        m = mp["respondent_id"].isin(idset).values
        if m.sum() == 0:
            return emb[:0], mp.iloc[:0].copy()
        return emb[m], mp.loc[m].copy()

    out: Dict[str, DatasetPack] = {}

    for lbl, pack in packs.items():
        
        profiles_f = _filt_df(pack.profiles) if pack.profiles is not None else pack.profiles
        open_long_f = _filt_df(pack.open_long)
        word_long_f = _filt_df(pack.word_long)
        likert_f = _filt_df(pack.likert)
        key_open_f = _filt_keyness_df(pack.key_open)
        key_word_f = _filt_keyness_df(pack.key_word)

        
        emb_q1_f, map_q1_f = _filt_emb(pack.emb_open_q1, pack.map_open_q1)
        emb_q2_f, map_q2_f = _filt_emb(pack.emb_open_q2, pack.map_open_q2)
        emb_wa_f, map_wa_f = _filt_emb(pack.emb_wordassoc, pack.map_wordassoc)

        out[lbl] = DatasetPack(
            label=pack.label,
            path=pack.path,
            profiles=profiles_f,
            open_long=open_long_f,
            word_long=word_long_f,
            likert=likert_f,
            key_open=key_open_f,
            key_word=key_word_f,
            meta=pack.meta,
            emb_open_q1=emb_q1_f,
            emb_open_q2=emb_q2_f,
            map_open_q1=map_q1_f,
            map_open_q2=map_q2_f,
            emb_wordassoc=emb_wa_f,
            map_wordassoc=map_wa_f,
        )

    return out


def add_subgroup_labels_to_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    df = profiles.copy()

    
    exp_map = {
        "Nessuna esperienza": "Breve Esperienza",
        "Meno di 5 anni": "Breve Esperienza",
        "Da 5 a 10 anni": "Breve Esperienza",
        "Da 10 a 20 anni": "Lunga Esperienza",
        "Più di 20 anni": "Lunga Esperienza",
    }
    if "Anni esp" in df.columns:
        s = df["Anni esp"].astype(str).str.strip()
        
        s = s.where(~s.str.lower().isin(["nan", "none", ""]), np.nan)
        df["esperienza_2"] = s.map(exp_map)
    else:
        df["esperienza_2"] = np.nan

    
    ruolo_map = {
        "Insegnante di sostegno": "Sostegno",
        "Insegnante curricolare": "Curricolare",
    }
    if "Ruolo" in df.columns:
        s = df["Ruolo"].astype(str).str.strip()
        s = s.where(~s.str.lower().isin(["nan", "none", ""]), np.nan)
        df["ruolo_2"] = s.map(ruolo_map)
    else:
        df["ruolo_2"] = np.nan

    
    grado_map = {
        "Primaria": "Primaria+Infanzia",
        "Infanzia": "Primaria+Infanzia",
        "Secondaria di I grado": "Secondaria (I+II)",
        "Secondaria di II grado": "Secondaria (I+II)",
    }
    if "Grado" in df.columns:
        s = df["Grado"].astype(str).str.strip()
        s = s.where(~s.str.lower().isin(["nan", "none", ""]), np.nan)
        df["grado_2"] = s.map(grado_map)
    else:
        df["grado_2"] = np.nan

    return df





def chi_square_p(series_a: pd.Series, series_b: pd.Series, cats: List[str]) -> float:
    ca = series_a.value_counts(dropna=False)
    cb = series_b.value_counts(dropna=False)
    table = np.array([[ca.get(c, 0) for c in cats], [cb.get(c, 0) for c in cats]], dtype=float)
    if table.sum() == 0 or np.any(table.sum(axis=1) == 0):
        return np.nan
    _, p, _, _ = chi2_contingency(table)
    return float(p)


def mwu_p(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)






def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return np.nan

    XX = np.clip(2.0 - 2.0 * (X @ X.T), 0.0, None)
    YY = np.clip(2.0 - 2.0 * (Y @ Y.T), 0.0, None)
    XY = np.clip(2.0 - 2.0 * (X @ Y.T), 0.0, None)

    d_xx = np.sqrt(XX)
    d_yy = np.sqrt(YY)
    d_xy = np.sqrt(XY)

    exy = d_xy.mean()
    exx = d_xx[np.triu_indices(n, k=1)].mean()
    eyy = d_yy[np.triu_indices(m, k=1)].mean()
    val = 2.0 * exy - exx - eyy
    return float(max(0.0, val)) 
    


def perm_test_energy(X: np.ndarray, Y: np.ndarray, n_perm: int = N_PERMUTATIONS) -> Tuple[float, float]:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    obs = energy_distance(X, Y)
    if not np.isfinite(obs):
        return obs, np.nan

    Z = np.vstack([X, Y])
    n = X.shape[0]
    idx = np.arange(Z.shape[0])

    ge = 0
    for _ in range(n_perm):
        np.random.shuffle(idx)
        Xp = Z[idx[:n]]
        Yp = Z[idx[n:]]
        val = energy_distance(Xp, Yp)
        if val >= obs - 1e-12:
            ge += 1

    p = (ge + 1) / (n_perm + 1)
    return obs, float(p)






def split_half_ids(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.array(ids)
    np.random.shuffle(ids)
    mid = len(ids) // 2
    return ids[:mid], ids[mid:]


def noise_floor_split_half(human_ids: np.ndarray, metric_fn, n_rep: int = N_SPLIT_HALF) -> np.ndarray:
    vals = []
    for _ in range(n_rep):
        a, b = split_half_ids(human_ids.copy())
        vals.append(metric_fn(a, b))
    return np.asarray(vals, dtype=float)






def barplot_with_noise_band(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    p_col: Optional[str],
    title: str,
    ylabel: str,
    noise_vals: Optional[np.ndarray],
    outpath: Path,
    *,
    signed_y: bool = False,
    stars_at_top: bool = True,
    headroom_frac: float = 0.08,
    
    title_fs: int = 12,
    label_fs: int = 10,
    tick_fs: int = 10,
    star_fs: int = 10,
) -> None:
    import matplotlib.transforms as mtransforms

    ensure_dir(outpath.parent)

    labels = df[x_col].tolist()
    ys = pd.to_numeric(df[y_col], errors="coerce").astype(float).tolist()
    ps = df[p_col].tolist() if (p_col and p_col in df.columns) else [None] * len(labels)

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4.2))
    ax.bar(range(len(labels)), ys)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)

    if title:
        ax.set_title(title, fontsize=title_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    if signed_y:
        ax.axhline(0.0, linestyle=":", linewidth=1.0)

    
    y_candidates = [v for v in ys if np.isfinite(v)]
    if noise_vals is not None and np.isfinite(noise_vals).any():
        med, q1, q3 = median_iqr(noise_vals)

        ax.axhspan(q1, q3, alpha=0.2)

        
        for v in (q1, q3):
            if np.isfinite(v):
                y_candidates.append(float(v))

        if signed_y:
            q95_abs = float(np.quantile(np.abs(noise_vals[np.isfinite(noise_vals)]), 0.95))
            ax.axhline(+q95_abs, linestyle=":", linewidth=1.2)
            ax.axhline(-q95_abs, linestyle=":", linewidth=1.2)
            y_candidates.extend([+q95_abs, -q95_abs])
        else:
            ax.axhline(med, linestyle="--", linewidth=1.2)
            q_95 = q95(noise_vals)
            ax.axhline(q_95, linestyle=":", linewidth=1.2)
            for v in (med, q_95):
                if np.isfinite(v):
                    y_candidates.append(float(v))

    
    if y_candidates:
        y_min, y_max = min(y_candidates), max(y_candidates)
        span = (y_max - y_min) + 1e-9
        pad = headroom_frac * span
        ax.set_ylim(y_min - 0.10 * span, y_max + pad)

    
    if stars_at_top:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        y_star_ax = 0.98
        for i, p in enumerate(ps):
            st = stars(p)
            if st:
                ax.text(
                    i, y_star_ax, st,
                    transform=trans,
                    ha="center", va="top",
                    fontsize=star_fs,
                    clip_on=True,
                )
    else:
        for i, (y, p) in enumerate(zip(ys, ps)):
            st = stars(p)
            if st and np.isfinite(y):
                ax.text(i, y, st, ha="center", va="bottom", fontsize=star_fs, clip_on=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def two_panel_barplots_with_noise(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_noise: Optional[np.ndarray],
    right_noise: Optional[np.ndarray],
    *,
    title: str,
    ylabel: str,
    outpath: Path,
    signed_y: bool,
    y_col: str = "value",
    p_col: str = "p_adj",
    stars_at_top: bool = True,
    headroom_frac: float = 0.08,   
) -> None:
    ensure_dir(outpath.parent)

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)

    def _extract(df: pd.DataFrame) -> Tuple[List[str], List[float], List[Optional[float]]]:
        labels = df["dataset"].tolist()
        yobj = df[y_col]
        if isinstance(yobj, pd.DataFrame):
            yobj = yobj.iloc[:, 0]
        ys = pd.to_numeric(yobj, errors="coerce").astype(float).tolist()
        ps = df[p_col].tolist() if (p_col and p_col in df.columns) else [None] * len(ys)
        return labels, ys, ps

    
    panel_payloads: List[Tuple[Any, List[float], List[Optional[float]]]] = []

    for ax, df, noise_vals, panel_title in [
        (axes[0], left_df, left_noise, "q1"),
        (axes[1], right_df, right_noise, "q2"),
    ]:
        labels, ys, ps = _extract(df)
        panel_payloads.append((ax, ys, ps))

        ax.bar(range(len(labels)), ys)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_title(panel_title)
        if ax == axes[0]:
            ax.set_ylabel(ylabel)

        if signed_y:
            ax.axhline(0.0, linestyle=":", linewidth=1.0)

        if noise_vals is not None and np.isfinite(noise_vals).any():
            med, q1, q3 = median_iqr(noise_vals)

            ax.axhspan(q1, q3, alpha=0.2)

            if signed_y:
                q95_abs = float(np.quantile(np.abs(noise_vals[np.isfinite(noise_vals)]), 0.95))
                ax.axhline(+q95_abs, linestyle=":", linewidth=1.2)
                ax.axhline(-q95_abs, linestyle=":", linewidth=1.2)

                
                
                
                

            else:
                ax.axhline(med, linestyle="--", linewidth=1.2)
                ax.axhline(q95(noise_vals), linestyle=":", linewidth=1.2)

        
        y_finite = [v for v in ys if np.isfinite(v)]
        if y_finite:
            y_min, y_max = min(y_finite), max(y_finite)
            pad = 0.12 * (y_max - y_min + 1e-9)
            cur0, cur1 = ax.get_ylim()
            ax.set_ylim(min(cur0, y_min - pad), max(cur1, y_max + pad))

    
    
    y0s, y1s = zip(*(ax.get_ylim() for ax in axes))
    y0 = float(min(y0s))
    y1 = float(max(y1s))
    span = (y1 - y0) if np.isfinite(y1 - y0) else 1.0
    y1 = y1 + headroom_frac * span

    for ax in axes:
        ax.set_ylim(y0, y1)

    
    for ax, ys, ps in panel_payloads:
        if stars_at_top:
            
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            y_star_ax = 0.98  
            for i, p in enumerate(ps):
                st = stars(p)
                if st:
                    ax.text(
                        i, y_star_ax, st,
                        transform=trans,
                        ha="center", va="top",
                        fontsize=10,
                        clip_on=True,   
                    )
        else:
            for i, (y, p) in enumerate(zip(ys, ps)):
                st = stars(p)
                if st and np.isfinite(y):
                    ax.text(i, y, st, ha="center", va="bottom", fontsize=10, clip_on=True)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def heatmap_with_stars(
    heat: pd.DataFrame,
    p_df: pd.DataFrame,
    *,
    title: str,
    outpath: Path,
    p_col: str = "p_adj",
    figsize: Tuple[float, float] = (8.0, 6.0),
    xtick_rotation: float = 0.0,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    
    title_fs: int = 17,
    tick_fs: int = 15,
    star_fs: int = 13,
    cbar_fs: int = 12,
    
    tick_pad: int = 2,
) -> None:
    ensure_dir(outpath.parent)

    fig, ax = plt.subplots(figsize=figsize)

    
    norm = None
    if (vmin is not None) and (vmax is not None):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    im = ax.imshow(
        heat.values,
        aspect="auto",
        cmap=(cmap if cmap is not None else None),
        norm=norm,
    )

    ax.set_xticks(range(heat.shape[1]))
    ax.set_yticks(range(heat.shape[0]))

    ax.set_xticklabels(list(heat.columns), rotation=xtick_rotation, ha="center", fontsize=tick_fs)
    ax.set_yticklabels(list(heat.index), fontsize=tick_fs)

    
    ax.tick_params(axis="both", which="major", labelsize=tick_fs, pad=tick_pad)

    ax.set_title(title, fontsize=title_fs)

    for i, q in enumerate(heat.index):
        for j, d in enumerate(heat.columns):
            sub = p_df[(p_df["question"] == q) & (p_df["dataset"] == d)]
            pa = float(sub[p_col].values[0]) if len(sub) else np.nan
            st = stars(pa)
            if st:
                ax.text(j, i, st, ha="center", va="center", fontsize=star_fs)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=cbar_fs)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def combine_wordassoc_heatmaps_triptych(outdir: Path) -> None:
    plots_dir = outdir / "plots"
    paths = [
        plots_dir / "heat_wordassoc_energy.png",
        plots_dir / "heat_wordassoc_entropy_delta.png",
        plots_dir / "heat_wordassoc_sentiment_shift.png",
    ]
    outpath = plots_dir / "heat_wordassoc_triptych.png"

    missing = [p for p in paths if not p.exists()]
    if missing:
        print("[WORDASSOC] triptych: missing inputs, skipping:", ", ".join(str(p.name) for p in missing))
        return

    ensure_dir(outpath.parent)

    imgs = [plt.imread(str(p)) for p in paths]

    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))

    for ax, img in zip(axes, imgs):
        ax.imshow(img, aspect="auto")  
        ax.axis("off")

    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0)

    
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"[WORDASSOC] triptych saved -> {outpath}")



def combine_open_double(outdir: Path) -> None:
    plots_dir = outdir / "plots"
    paths = [
        plots_dir / "open_embed_energy_q1_q2.png",
        plots_dir / "open_sentiment_shift_signed_q1_q2.png",
    ]
    outpath = plots_dir / "open_double.png"

    missing = [p for p in paths if not p.exists()]
    if missing:
        print("[OPEN] double: missing inputs, skipping:", ", ".join(str(p.name) for p in missing))
        return

    ensure_dir(outpath.parent)

    imgs = [plt.imread(str(p)) for p in paths]

    
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    for ax, img in zip(axes, imgs):
        ax.imshow(img, aspect="auto")   
        ax.axis("off")

    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0)

    
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"[OPEN] double saved -> {outpath}")






def parse_tokens_column(df: pd.DataFrame) -> List[List[str]]:
    toks: List[List[str]] = []
    for s in df["tokens"].fillna("").astype(str).tolist():
        s = s.strip()
        toks.append(s.split() if s else [])
    return toks


def token_counts(token_lists: Iterable[Iterable[str]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for toks in token_lists:
        for t in toks:
            c[t] = c.get(t, 0) + 1
    return c


def keyness_log_odds(
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    alpha: float = KEYNESS_ALPHA,
    min_total: int = KEYNESS_MIN_TOTAL_COUNT,
) -> pd.DataFrame:
    vocab = set(counts_a) | set(counts_b)
    na = sum(counts_a.values())
    nb = sum(counts_b.values())
    V = len(vocab)

    rows = []
    for w in vocab:
        ca = counts_a.get(w, 0)
        cb = counts_b.get(w, 0)
        if ca + cb < min_total:
            continue

        pa = (ca + alpha) / (na + alpha * V)
        pb = (cb + alpha) / (nb + alpha * V)

        loa = math.log(pa / (1 - pa))
        lob = math.log(pb / (1 - pb))
        log_odds = lob - loa

        var = 1.0 / (cb + alpha) + 1.0 / (ca + alpha)
        z = log_odds / math.sqrt(var)

        rows.append({"token": w, "count_a": ca, "count_b": cb, "log_odds": log_odds, "z": z})

    out = pd.DataFrame(rows).sort_values("z", ascending=False)
    if len(out):
        out["p"] = 2 * (1 - norm.cdf(np.abs(out["z"].values)))
        out["p_adj"] = bh_fdr(out["p"].tolist())
    return out


def top_terms_from_keyness(kdf: pd.DataFrame, *, direction: str, k: int = 10, alpha: float = 0.05) -> List[str]:
    if kdf is None or len(kdf) == 0:
        return []
    df = kdf.copy()
    if "p_adj" not in df.columns:
        df["p_adj"] = df.get("p", np.nan)

    df = df[np.isfinite(df["p_adj"]) & (df["p_adj"] < alpha)]

    if direction == "system":
        df = df[df["z"] > 0].sort_values("z", ascending=False)
    elif direction == "human":
        df = df[df["z"] < 0].sort_values("z", ascending=True)
    else:
        raise ValueError(direction)

    return df["token"].head(k).astype(str).tolist()


def merged_keyness_table(
    human_df: pd.DataFrame,
    system_dfs: List[pd.DataFrame],
    questions: List[str],
    *,
    question_col: str = "question",
) -> pd.DataFrame:
    out_rows = []
    pooled = pd.concat(system_dfs, ignore_index=True) if len(system_dfs) else pd.DataFrame(columns=human_df.columns)

    for q in questions:
        h_q = human_df[human_df[question_col] == q].copy()
        s_q = pooled[pooled[question_col] == q].copy()

        h_counts = token_counts(parse_tokens_column(h_q))
        s_counts = token_counts(parse_tokens_column(s_q))

        kdf = keyness_log_odds(h_counts, s_counts, alpha=KEYNESS_ALPHA, min_total=KEYNESS_MIN_TOTAL_COUNT)
        top_h = top_terms_from_keyness(kdf, direction="human", k=10, alpha=0.05)
        top_s = top_terms_from_keyness(kdf, direction="system", k=10, alpha=0.05)

        out_rows.append({"question": q, "human": ";".join(top_h), "system": ";".join(top_s)})

    return pd.DataFrame(out_rows)






def _load_open_embeddings(pack: DatasetPack, q: str) -> Tuple[np.ndarray, np.ndarray]:
    if q == "q1":
        if pack.emb_open_q1 is None or pack.map_open_q1 is None:
            raise RuntimeError(f"{pack.label}: missing embeddings_open_q1.npy / map_open_q1.csv")
        return pack.emb_open_q1, pack.map_open_q1["respondent_id"].values
    if q == "q2":
        if pack.emb_open_q2 is None or pack.map_open_q2 is None:
            raise RuntimeError(f"{pack.label}: missing embeddings_open_q2.npy / map_open_q2.csv")
        return pack.emb_open_q2, pack.map_open_q2["respondent_id"].values
    raise ValueError(q)


def _load_wordassoc_embeddings(pack: DatasetPack) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pack.emb_wordassoc is None or pack.map_wordassoc is None:
        raise RuntimeError(f"{pack.label}: missing embeddings_wordassoc.npy / map_wordassoc.csv")
    X = pack.emb_wordassoc
    ids = pack.map_wordassoc["respondent_id"].values
    qs = pack.map_wordassoc["question"].values
    return X, ids, qs






def compute_open_embedding_fidelity(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def compute_all() -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        for q in OPEN_QS:
            Xh, idh = _load_open_embeddings(human, q)
            human_ids = np.unique(idh)

            def nf_metric(a_ids, b_ids):
                Xa = Xh[np.isin(idh, a_ids)]
                Xb = Xh[np.isin(idh, b_ids)]
                return energy_distance(Xa, Xb)

            noise_path = cache_dir / f"noise_open_embed_energy_{q}.npy"
            _ = load_or_compute_npy(noise_path, cache, lambda: noise_floor_split_half(human_ids, nf_metric))

            sub_rows: List[Dict[str, Any]] = []
            for lbl in models:
                Xm, _ = _load_open_embeddings(packs[lbl], q)

                
                if q == "q1" and lbl == models[0]:   
                    obs, ptmp = perm_test_energy(Xh, Xm, n_perm=50)
                    print(f"[DEBUG open_embed] {q} vs {lbl}: obs={obs:.6f}, p(50)={ptmp:.4f}")

                val, pval = perm_test_energy(Xh, Xm, n_perm=N_PERMUTATIONS)
                sub_rows.append({"group": "open_embedding_fidelity", "question": q, "dataset": lbl, "value": val, "p": pval})

            _apply_fdr_inplace(sub_rows, p_key="p", out_key="p_adj")
            results.extend(sub_rows)

        df = pd.DataFrame(results)
        df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
        df = df.sort_values(["question", "dataset"])
        return df

    return load_or_compute_df(cache_dir / "open_embedding_fidelity.csv", cache, compute_all)


def plot_open_embedding_fidelity(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    models = ordered_models(packs)
    df = pd.read_csv(cache_dir / "open_embedding_fidelity.csv")

    panels = {}
    for q in OPEN_QS:
        dfq = df[df["question"] == q].copy()
        dfq["dataset"] = pd.Categorical(dfq["dataset"], categories=models, ordered=True)
        dfq = dfq.sort_values("dataset")
        noise = load_npy(cache_dir / f"noise_open_embed_energy_{q}.npy")
        print(f"open_embed {q}:", noise.min(), noise.mean(), noise.max(), noise.std())
        panels[q] = (dfq, noise)

    if "q1" in panels and "q2" in panels:
        df_q1, noise_q1 = panels["q1"]
        
        df_q2, noise_q2 = panels["q2"]
        
        two_panel_barplots_with_noise(
            left_df=df_q1,
            right_df=df_q2,
            left_noise=noise_q1,
            right_noise=noise_q2,
            title="Open-ended embedding fidelity",
            ylabel="Energy distance",
            outpath=outdir / "plots" / "open_embed_energy_q1_q2.png",
            signed_y=False,
            y_col="value",
            p_col="p_adj",
        )


def compute_open_sentiment(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def compute_all() -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for q in OPEN_QS:
            hq = human.open_long[human.open_long["question"] == q]
            human_ids = np.unique(hq["respondent_id"].values)

            def nf_mean_shift(a_ids, b_ids):
                A = hq[hq["respondent_id"].isin(a_ids)]["sentiment_score"].values
                B = hq[hq["respondent_id"].isin(b_ids)]["sentiment_score"].values
                return float(np.nanmean(A) - np.nanmean(B))

            noise_path = cache_dir / f"noise_open_sentiment_shift_{q}.npy"
            _ = load_or_compute_npy(noise_path, cache, lambda: noise_floor_split_half(human_ids, nf_mean_shift))

            mh = float(np.nanmean(hq["sentiment_score"].values))
            sub_rows: List[Dict[str, Any]] = []
            for lbl in models:
                mq = packs[lbl].open_long[packs[lbl].open_long["question"] == q]
                mm = float(np.nanmean(mq["sentiment_score"].values))
                mean_shift = mm - mh
                p_mwu = mwu_p(hq["sentiment_score"].values, mq["sentiment_score"].values)
                sub_rows.append({"group": "open_sentiment_mean_shift", "question": q, "dataset": lbl, "value": mean_shift, "p": p_mwu})

            _apply_fdr_inplace(sub_rows, p_key="p", out_key="p_adj")
            rows.extend(sub_rows)

        df = pd.DataFrame(rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
        df = df.sort_values(["question", "dataset"])
        return df

    return load_or_compute_df(cache_dir / "open_sentiment.csv", cache, compute_all)


def plot_open_sentiment(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    models = ordered_models(packs)
    df = pd.read_csv(cache_dir / "open_sentiment.csv")

    panels = {}
    for q in OPEN_QS:
        dfq = df[df["question"] == q].copy()
        dfq["dataset"] = pd.Categorical(dfq["dataset"], categories=models, ordered=True)
        dfq = dfq.sort_values("dataset")
        noise = load_npy(cache_dir / f"noise_open_sentiment_shift_{q}.npy")
        panels[q] = (dfq, noise)

    if "q1" in panels and "q2" in panels:
        df1, n1 = panels["q1"]
        print(n1.mean(), n1.std())
        df2, n2 = panels["q2"]
        print(n2.mean(), n2.std())
        two_panel_barplots_with_noise(
            left_df=df1,
            right_df=df2,
            left_noise=n1,
            right_noise=n2,
            title="Open-ended sentiment mean shift",
            ylabel="(model − human) mean sentiment score",
            outpath=outdir / "plots" / "open_sentiment_shift_signed_q1_q2.png",
            signed_y=True,
            y_col="value",
            p_col="p_adj",
        )


def compute_open_keyness_merged(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    def compute_fn():
        human = packs["human"]
        models = ordered_models(packs)
        out = merged_keyness_table(
            human_df=human.key_open,
            system_dfs=[packs[m].key_open for m in models],
            questions=OPEN_QS,
            question_col="question",
        )
        return out

    return load_or_compute_df(cache_dir / "top10_open.csv", cache, compute_fn)


def compute_open_wordcounts(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def compute_fn() -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        
        for q in OPEN_QS:
            dfq = human.open_long[human.open_long["question"] == q].copy()
            wc = _open_wordcount_series(dfq)
            for v in wc.to_numpy(dtype=float):
                if np.isfinite(v):
                    rows.append({"question": q, "dataset": "human", "word_count": float(v)})

        
        for lbl in models:
            for q in OPEN_QS:
                dfq = packs[lbl].open_long[packs[lbl].open_long["question"] == q].copy()
                wc = _open_wordcount_series(dfq)
                for v in wc.to_numpy(dtype=float):
                    if np.isfinite(v):
                        rows.append({"question": q, "dataset": lbl, "word_count": float(v)})

        df = pd.DataFrame(rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=ordered_systems(packs), ordered=True)
        df = df.sort_values(["question", "dataset"])
        return df

    return load_or_compute_df(cache_dir / "open_wordcounts.csv", cache, compute_fn)


def plot_open_wordcounts_boxplots(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    df = pd.read_csv(cache_dir / "open_wordcounts.csv")
    if df.empty:
        return

    datasets = ordered_systems(packs)
    models = [d for d in datasets if d != "human"]

    
    p_rows: List[Dict[str, Any]] = []
    for q in OPEN_QS:
        dh = df[(df.question == q) & (df.dataset == "human")]["word_count"].to_numpy(dtype=float)
        sub = []
        for m in models:
            dm = df[(df.question == q) & (df.dataset == m)]["word_count"].to_numpy(dtype=float)
            p = mwu_p(dh, dm)
            sub.append({"question": q, "dataset": m, "p": p})
        _apply_fdr_inplace(sub, p_key="p", out_key="p_adj")
        p_rows.extend(sub)

    dfp = pd.DataFrame(p_rows)

    
    ensure_dir(outdir / "plots")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, q in zip(axes, OPEN_QS):
        
        data = []
        for d in datasets:
            vals = df[(df.question == q) & (df.dataset == d)]["word_count"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            data.append(vals)

        ax.boxplot(
            data,
            tick_labels=datasets,
            showfliers=False,
        )
        ax.set_title(q)
        ax.set_xlabel("")  
        if ax == axes[0]:
           ax.set_ylabel("Number of words")
        ax.tick_params(axis="x", rotation=0)

        
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) + 1e-9
        ax.set_ylim(y0, y1 + 0.15 * span)  

        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        y_star_ax = 0.98

        for i, d in enumerate(datasets, start=1):  
            if d == "human":
                continue
            sub = dfp[(dfp.question == q) & (dfp.dataset == d)]
            pa = float(sub["p_adj"].values[0]) if len(sub) else np.nan
            st = stars(pa)
            if st:
                ax.text(i, y_star_ax, st, transform=trans, ha="center", va="top", fontsize=11, clip_on=True)

    
    fig.tight_layout()
    fig.savefig(outdir / "plots" / "open_wordcounts_box_q1_q2.png", dpi=200)
    plt.close(fig)







def compute_wordassoc_unique_answer_pct(
    packs: Dict[str, DatasetPack],
    cache_dir: Path,
    cache: CachePolicy,
) -> pd.DataFrame:
    models = ordered_models(packs)
    groups = ["human"] + models

    def compute_fn() -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for g in groups:
            pack = packs[g]
            for q in WORD_QS:
                dfq = pack.word_long[pack.word_long["question"] == q].copy()
                s = _wordassoc_answer_key_series(dfq).dropna()

                total_n = int(s.shape[0])
                uniq_n = int(s.nunique()) if total_n else 0
                pct = 100.0 * uniq_n / total_n if total_n else np.nan

                rows.append(
                    {
                        "question": q,
                        "group": g,
                        "total_n": total_n,
                        "unique_n": uniq_n,
                        "unique_pct": pct,
                    }
                )

        out = pd.DataFrame(rows)
        out["question"] = pd.Categorical(out["question"], categories=WORD_QS, ordered=True)
        out["group"] = pd.Categorical(out["group"], categories=groups, ordered=True)
        out = out.sort_values(["question", "group"])
        return out

    return load_or_compute_df(cache_dir / "wordassoc_unique_answer_pct_long.csv", cache, compute_fn)


def export_wordassoc_unique_answer_pct_tables(cache_dir: Path, outdir: Path) -> Tuple[Path, Path]:
    in_path = cache_dir / "wordassoc_unique_answer_pct_long.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    df = pd.read_csv(in_path)
    ensure_dir(outdir / "plots")

    
    
    present = df["group"].astype(str).unique().tolist()
    canonical = ["human"] + [m for m in SYSTEM_ORDER if m != "human"]  
    group_order = [g for g in canonical if g in present] + [g for g in present if g not in canonical]

    
    wide = (
        df.pivot_table(index="question", columns="group", values="unique_pct", aggfunc="mean")
          .reindex(index=WORD_QS)
          .reindex(columns=group_order)
    )

    csv_path = outdir / "plots" / "wordassoc_unique_answer_pct.csv"
    wide.to_csv(csv_path, index=True)

    
    tex_path = outdir / "plots" / "table_wordassoc_unique_answer_pct.tex"

    def _fmt(x):
        return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{float(x):.1f}"

    lines: List[str] = []
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")

    cols = "l" + "r" * len(group_order)
    lines.append(rf"\begin{ tabular} { {cols}} ")
    lines.append(r"\toprule")

    header = "Question & " + " & ".join([str(g).replace("_", r"\_") for g in group_order]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for q in WORD_QS:
        row = [q.replace("_", r"\_")]
        for g in group_order:
            val = wide.loc[q, g] if (q in wide.index and g in wide.columns) else np.nan
            row.append(_fmt(val))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Percentage of unique word-association answers per item (q3--q16). "
        r"Systems are reported individually (not pooled).}"
    )
    lines.append(r"\label{tab:wa-unique-pct}")
    lines.append(r"\end{table}")

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WORDASSOC] Unique% CSV  -> {csv_path}")
    print(f"[WORDASSOC] Unique% LaTeX -> {tex_path}")
    return csv_path, tex_path


def compute_wordassoc_sentiment_shift(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def compute_fn():
        rows: List[Dict[str, Any]] = []
        for q in WORD_QS:
            hq = human.word_long[human.word_long["question"] == q]
            xh = hq["sentiment_score"].values.astype(float)
            mh = float(np.nanmean(xh))

            sub_rows: List[Dict[str, Any]] = []
            for lbl in models:
                mq = packs[lbl].word_long[packs[lbl].word_long["question"] == q]
                xm = mq["sentiment_score"].values.astype(float)
                mm = float(np.nanmean(xm))
                shift = mm - mh
                p = mwu_p(xh, xm)
                sub_rows.append({"question": q, "dataset": lbl, "shift": shift, "p": p})

            _apply_fdr_inplace(sub_rows, p_key="p", out_key="p_adj")
            rows.extend(sub_rows)

        df = pd.DataFrame(rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
        df = df.sort_values(["question", "dataset"])
        return df

    return load_or_compute_df(cache_dir / "wordassoc_sentiment_shift.csv", cache, compute_fn)


def plot_wordassoc_sentiment_shift(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    models = ordered_models(packs)

    df = pd.read_csv(cache_dir / "wordassoc_sentiment_shift.csv")
    if df.empty:
        return

    
    df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
    df = df.sort_values(["question", "dataset"])

    
    heat = (
        df.pivot_table(index="question", columns="dataset", values="shift", aggfunc="mean", observed=False)
          .reindex(WORD_QS)
          .reindex(columns=models)
    )


    heatmap_with_stars(
        heat=heat,
        p_df=df,
        title="Sentiment mean shift (model - human)",
        outpath=outdir / "plots" / "heat_wordassoc_sentiment_shift.png",
        p_col="p_adj",
        figsize=(max(6, 1 + 0.9 * heat.shape[1]), 6),
        
        cmap="RdYlGn",           
        vmin=-1.0,               
        vmax=+1.0,               
    )


def compute_wordassoc_semantic_distance_energy(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def compute_fn():
        Xh, _, qh = _load_wordassoc_embeddings(human)

        rows: List[Dict[str, Any]] = []
        for q in WORD_QS:
            H = Xh[qh == q]

            sub_rows: List[Dict[str, Any]] = []
            for lbl in models:
                Xm, _, qm = _load_wordassoc_embeddings(packs[lbl])
                M = Xm[qm == q]
                val, p = perm_test_energy(H, M, n_perm=N_PERMUTATIONS)
                sub_rows.append({"group": "wordassoc_energy", "question": q, "dataset": lbl, "value": val, "p": p})

            _apply_fdr_inplace(sub_rows, p_key="p", out_key="p_adj")
            rows.extend(sub_rows)

        df = pd.DataFrame(rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
        df = df.sort_values(["question", "dataset"])
        return df

    return load_or_compute_df(cache_dir / "wordassoc_energy_distance.csv", cache, compute_fn)


def plot_wordassoc_semantic_distance_energy(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    models = ordered_models(packs)
    df = pd.read_csv(cache_dir / "wordassoc_energy_distance.csv")
    df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
    df = df.sort_values(["question", "dataset"])

    heat = df.pivot_table(index="question", columns="dataset", values="value", aggfunc="mean", observed=False).reindex(WORD_QS)
    heat = heat.reindex(columns=models)

    heatmap_with_stars(
        heat=heat,
        p_df=df,
        title="Energy distance (model vs human)",
        outpath=outdir / "plots" / "heat_wordassoc_energy.png",
        p_col="p_adj",
        figsize=(max(6, 1 + 0.9 * heat.shape[1]), 6),
        xtick_rotation=0.0,
    )


def compute_wordassoc_entropy_delta(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def entropy_norm(keys: List[str]) -> float:
        if len(keys) < 2:
            return np.nan
        vc = pd.Series(keys).value_counts()
        p = (vc / vc.sum()).values.astype(float)
        H = -np.sum(p * np.log(p + 1e-12))
        K = len(vc)
        return float(H / np.log(K + 1e-12)) if K > 1 else 0.0

    def bootstrap_entropy(pack: DatasetPack, q: str) -> np.ndarray:
        dfq = pack.word_long[pack.word_long["question"] == q]
        ids = np.unique(dfq["respondent_id"].values)
        if len(ids) < 10:
            return np.array([], dtype=float)

        vals = []
        for _ in range(N_BOOTSTRAP):
            samp = np.random.choice(ids, size=len(ids), replace=True)
            keys = dfq[dfq["respondent_id"].isin(samp)]["entropy_key"].astype(str).tolist()
            vals.append(entropy_norm(keys))
        return np.asarray(vals, dtype=float)

    def compute_fn():
        rows: List[Dict[str, Any]] = []

        for q in WORD_QS:
            hq = human.word_long[human.word_long["question"] == q]
            keys_h = hq["entropy_key"].astype(str).tolist()

            Hh = entropy_norm(keys_h)
            Hh_bs = bootstrap_entropy(human, q)

            sub_rows: List[Dict[str, Any]] = []
            for lbl in models:
                pack = packs[lbl]
                mq = pack.word_long[pack.word_long["question"] == q]
                keys_m = mq["entropy_key"].astype(str).tolist()

                Hm = entropy_norm(keys_m)
                dH = (Hm - Hh)  

                Hm_bs = bootstrap_entropy(pack, q)
                sub_rows.append({
                    "group": "wordassoc_entropy_delta",
                    "question": q,
                    "dataset": lbl,
                    "value": dH,
                    "human": Hh,
                    "model": Hm,
                    "p": mwu_p(Hh_bs, Hm_bs) if len(Hh_bs) and len(Hm_bs) else np.nan,
                })

            _apply_fdr_inplace(sub_rows, p_key="p", out_key="p_adj")
            rows.extend(sub_rows)

        df = pd.DataFrame(rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
        df = df.sort_values(["question", "dataset"])
        return df

    return load_or_compute_df(cache_dir / "wordassoc_entropy_delta.csv", cache, compute_fn)

def plot_wordassoc_entropy_delta(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    models = ordered_models(packs)
    df = pd.read_csv(cache_dir / "wordassoc_entropy_delta.csv")
    df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
    df = df.sort_values(["question", "dataset"])

    heat = df.pivot_table(index="question", columns="dataset", values="value", aggfunc="mean", observed=False).reindex(WORD_QS)
    heat = heat.reindex(columns=models)

    
    
    extra_kwargs = {}
    for k in ("center", "vmin", "vmax"):
        if "center" in getattr(heatmap_with_stars, "__code__", type("x", (), {"co_varnames": ()})) .co_varnames:
            extra_kwargs["center"] = 0.0
            break

    heatmap_with_stars(
        heat=heat,
        p_df=df,
        title="Entropy difference (model - human)",
        outpath=outdir / "plots" / "heat_wordassoc_entropy_delta.png",
        p_col="p_adj",
        figsize=(max(6, 1 + 0.9 * heat.shape[1]), 6),
        xtick_rotation=0.0,
        **extra_kwargs,
    )



def compute_wordassoc_keyness_merged(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    def compute_fn():
        human = packs["human"]
        models = ordered_models(packs)
        out = merged_keyness_table(
            human_df=human.key_word,
            system_dfs=[packs[m].key_word for m in models],
            questions=WORD_QS,
            question_col="question",
        )
        return out

    return load_or_compute_df(cache_dir / "top10_wordassoc.csv", cache, compute_fn)






def compute_likert_grouped_stacked_tables(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> Tuple[pd.DataFrame, pd.DataFrame]:
    human = packs["human"]
    options = [1, 2, 3, 4]
    datasets = ordered_systems(packs)
    models = [d for d in datasets if d != "human"]

    prop_path = cache_dir / "likert_props.csv"
    p_path = cache_dir / "likert_chi2.csv"

    if cache_ok(prop_path, cache) and cache_ok(p_path, cache):
        return pd.read_csv(prop_path), pd.read_csv(p_path)

    prop_rows: List[Dict[str, Any]] = []
    p_rows: List[Dict[str, Any]] = []

    for q in LIKERT_QS:
        
        for d in datasets:
            s = packs[d].likert[f"{q}_score"].dropna().astype(int)
            vc = s.value_counts().reindex(options, fill_value=0)
            prop = (vc / vc.sum()).values.astype(float) if vc.sum() else np.zeros(4, dtype=float)
            for opt, pr in zip(options, prop):
                prop_rows.append({"question": q, "dataset": d, "option": opt, "prop": float(pr)})

        
        h = human.likert[f"{q}_score"].dropna().astype(int).astype(str)
        cats = [str(o) for o in options]
        sub_p: List[Dict[str, Any]] = []
        for d in models:
            m = packs[d].likert[f"{q}_score"].dropna().astype(int).astype(str)
            p = chi_square_p(h, m, cats)
            sub_p.append({"question": q, "dataset": d, "p": p})

        _apply_fdr_inplace(sub_p, p_key="p", out_key="p_adj")
        p_rows.extend(sub_p)

    df_prop = pd.DataFrame(prop_rows)
    df_prop["dataset"] = pd.Categorical(df_prop["dataset"], categories=datasets, ordered=True)
    df_prop = df_prop.sort_values(["question", "dataset", "option"])

    df_p = pd.DataFrame(p_rows)
    df_p["dataset"] = pd.Categorical(df_p["dataset"], categories=models, ordered=True)
    df_p = df_p.sort_values(["question", "dataset"])

    save_df(df_prop, prop_path)
    save_df(df_p, p_path)
    return df_prop, df_p


def plot_likert_grouped_stacked(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path,
                                    fs_tick: int = 14,
                                    fs_label: int = 16,
                                    fs_legend: int = 13,
                                    fs_star: int = 13,
                                ) -> None:
    datasets = ordered_systems(packs)
    models = [d for d in datasets if d != "human"]
    options = [1, 2, 3, 4]

    df_prop = pd.read_csv(cache_dir / "likert_props.csv")
    df_p = pd.read_csv(cache_dir / "likert_chi2.csv")

    
    group_gap = 0.05
    bar_w = 0.1
    step = 0.125  

    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(datasets) * len(LIKERT_QS)), 5.4))

    
    base_x: List[float] = []
    x = 0.0
    for _q in LIKERT_QS:
        for _d in datasets:
            base_x.append(x)
            x += step
        x += group_gap
    base_x = np.array(base_x, dtype=float)

    
    bottoms = np.zeros(len(base_x), dtype=float)
    for opt in options:
        heights = []
        for q in LIKERT_QS:
            for d in datasets:
                v = df_prop[(df_prop.question == q) & (df_prop.dataset == d) & (df_prop.option == opt)]["prop"]
                heights.append(float(v.values[0]) if len(v) else 0.0)
        heights = np.array(heights, dtype=float)
        ax.bar(base_x, heights, width=bar_w, bottom=bottoms, label=f"{opt}")
        bottoms += heights

    ax.set_ylabel("Proportion")
    
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=fs_legend)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, fontsize=fs_legend)
    fig.tight_layout(rect=[0, 0, 0.82, 1])  


    
    y_star = 1.02
    idx = 0
    for q in LIKERT_QS:
        for d in datasets:
            if d in models:
                psub = df_p[(df_p.question == q) & (df_p.dataset == d)]
                p_adj = float(psub["p_adj"].values[0]) if len(psub) else np.nan
                st = stars(p_adj)
                if st:
                    ax.text(base_x[idx], y_star, st, ha="center", va="bottom", fontsize=fs_star)
            idx += 1

    
    ax.set_xticks([])

    
    sys_y = -0.06
    idx = 0
    for _q in LIKERT_QS:
        for d in datasets:
            ax.text(
                base_x[idx],
                sys_y,
                d,
                rotation=0,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=fs_tick,
                clip_on=False,
            )
            idx += 1

    
    q_y = -0.14
    n_d = len(datasets)
    for gi, q in enumerate(LIKERT_QS):
        start = gi * n_d
        end = start + n_d  
        group_xs = base_x[start:end]
        if len(group_xs) == 0:
            continue
        center = 0.5 * (group_xs[0] + group_xs[-1])
        ax.text(
            center,
            q_y,
            q,
            rotation=0,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=fs_label,
            fontweight="bold",
            clip_on=False,
        )



    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_ylabel("Proportion", fontsize=fs_label)

    
    ax.set_ylim(0, 1.10)
    if len(base_x):
        pad = max(bar_w, 0.05)
        ax.set_xlim(base_x.min() - pad, base_x.max() + pad)

    fig.tight_layout()
    
    ensure_dir((outdir / "plots").resolve())
    fig.savefig(outdir / "plots" / "likert_grouped_stacked.png", dpi=200)
    plt.close(fig)


def compute_skepticism_diff(packs: Dict[str, DatasetPack], cache_dir: Path, cache: CachePolicy) -> pd.DataFrame:
    human = packs["human"]
    models = ordered_models(packs)

    def compute_fn() -> pd.DataFrame:
        h_all = human.likert["skepticism_index_S"].values.astype(float)
        h_all = h_all[np.isfinite(h_all)]
        if len(h_all) < 5:
            return pd.DataFrame()

        
        if "respondent_id" in human.likert.columns:
            ids = human.likert["respondent_id"].values
            uniq = np.unique(ids)

            def nf(a_ids, b_ids):
                A = human.likert[human.likert["respondent_id"].isin(a_ids)]["skepticism_index_S"].values.astype(float)
                B = human.likert[human.likert["respondent_id"].isin(b_ids)]["skepticism_index_S"].values.astype(float)
                return float(np.nanmean(A) - np.nanmean(B))

            noise = noise_floor_split_half(uniq, nf)
        else:
            idx = np.arange(len(h_all))
            noise_vals = []
            for _ in range(N_SPLIT_HALF):
                np.random.shuffle(idx)
                mid = len(idx) // 2
                noise_vals.append(float(np.nanmean(h_all[idx[:mid]]) - np.nanmean(h_all[idx[mid:]])))
            noise = np.asarray(noise_vals, dtype=float)

        save_npy(noise, cache_dir / "noise_skepticism_split_half.npy")

        rows: List[Dict[str, Any]] = []
        for d in models:
            x = packs[d].likert["skepticism_index_S"].values.astype(float)
            x = x[np.isfinite(x)]
            rows.append({
                "dataset": d,
                "mean_diff_vs_human": float(np.nanmean(x) - np.nanmean(h_all)),
                "p": mwu_p(h_all, x),
            })

        _apply_fdr_inplace(rows, p_key="p", out_key="p_adj")

        df = pd.DataFrame(rows)
        df["dataset"] = pd.Categorical(df["dataset"], categories=models, ordered=True)
        df = df.sort_values("dataset")
        return df

    
    noise_path = cache_dir / "noise_skepticism_split_half.npy"
    if (not cache_ok(noise_path, cache)):
        
        pass

    return load_or_compute_df(cache_dir / "skepticism_diff_vs_human.csv", cache, compute_fn)


def plot_skepticism_diff(packs: Dict[str, DatasetPack], cache_dir: Path, outdir: Path) -> None:
    df = pd.read_csv(cache_dir / "skepticism_diff_vs_human.csv")
    if df.empty:
        return
    noise = load_npy(cache_dir / "noise_skepticism_split_half.npy")

    barplot_with_noise_band(
        df=df,
        x_col="dataset",
        y_col="mean_diff_vs_human",
        p_col="p_adj",
        title="",
        ylabel="(model − human) mean skepticism index (S)",
        noise_vals=noise,
        outpath=outdir / "plots" / "skepticism_diff_vs_human.png",
        signed_y=True,
        tick_fs=11,
        label_fs=12,
        star_fs=11,
    )


def export_keyness_merged_latex_table(cache_dir: Path, outdir: Path) -> Path:
    open_path = cache_dir / "top10_open.csv"
    wa_path = cache_dir / "top10_wordassoc.csv"
    outpath = outdir / "plots" / "table_keyness_top10.tex"
    ensure_dir(outpath.parent)

    if (not open_path.exists()) or (not wa_path.exists()):
        missing = [p.name for p in [open_path, wa_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required cache files: {missing}")

    df_open = pd.read_csv(open_path)
    df_wa = pd.read_csv(wa_path)

    def _qnum(q: str) -> int:
        m = re.search(r"(\d+)", str(q))
        return int(m.group(1)) if m else 10**9

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        
        for c in ["question", "human", "system"]:
            if c not in d.columns:
                raise ValueError(f"Keyness table missing column '{c}' in {d.columns.tolist()}")

        d["Question"] = d["question"].astype(str)
        d["Human"] = d["human"].fillna("").astype(str)
        d["System"] = d["system"].fillna("").astype(str)

        
        d["Human"] = d["Human"].apply(lambda s: ", ".join([t.strip() for t in s.split(";") if t.strip()]))
        d["System"] = d["System"].apply(lambda s: ", ".join([t.strip() for t in s.split(";") if t.strip()]))

        d["qnum"] = d["Question"].apply(_qnum)
        d = d.sort_values("qnum").drop(columns=["qnum", "question", "human", "system"])
        return d[["Question", "Human", "System"]]

    open_block = _prep(df_open)
    wa_block = _prep(df_wa)

    
    def _latex_escape(s: str) -> str:
        s = "" if s is None else str(s)
        s = s.replace("\\", r"\textbackslash{}")
        s = s.replace("&", r"\&")
        s = s.replace("%", r"\%")
        s = s.replace("$", r"\$")
        s = s.replace("#", r"\#")
        s = s.replace("_", r"\_")
        s = s.replace("{", r"\{")
        s = s.replace("}", r"\}")
        s = s.replace("~", r"\textasciitilde{}")
        s = s.replace("^", r"\textasciicircum{}")
        return s

    header = r"""
% Requires in preamble:
% \usepackage{booktabs}
% \usepackage{tabularx}
%
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\linewidth}{l p{7cm} p{7cm}}
\toprule
\textbf{Q.} & \textbf{Human distinctive} & \textbf{System distinctive} \\
\midrule
""".lstrip("\n")

    footer = r"""
\bottomrule
\end{tabularx}
\caption{Top-10 distinctive tokens (log-odds with Dirichlet prior) for humans vs pooled systems.}
\label{tab:keyness-top10}
\end{table*}
""".lstrip("\n")

    lines: List[str] = [header]

    
    for _, r in open_block.iterrows():
        q = _latex_escape(r["Question"])
        h = _latex_escape(r["Human"])
        s = _latex_escape(r["System"])
        lines.append(f"{q} & {h} & {s} \\\\")

    
    if len(open_block) and len(wa_block):
        lines.append(r"\midrule")

    
    for _, r in wa_block.iterrows():
        q = _latex_escape(r["Question"])
        h = _latex_escape(r["Human"])
        s = _latex_escape(r["System"])
        lines.append(f"{q} & {h} & {s} \\\\")

    lines.append(footer)

    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"[KEYNESS] LaTeX table saved -> {outpath}")
    return outpath






def analyze_open(packs: Dict[str, DatasetPack], outdir: Path, cache_dir: Path, cache: CachePolicy) -> None:
    if cache.compute:
        print("[OPEN] embedding fidelity (compute/cache)...")
        compute_open_embedding_fidelity(packs, cache_dir, cache)

        print("[OPEN] sentiment shift (compute/cache)...")
        compute_open_sentiment(packs, cache_dir, cache)

        print("[OPEN] keyness merged (compute/cache)...")
        compute_open_keyness_merged(packs, cache_dir, cache)

        
        print("[OPEN] word counts (compute/cache)...")
        compute_open_wordcounts(packs, cache_dir, cache)        

    if cache.plot:
        print("[OPEN] embedding fidelity (plot)...")
        plot_open_embedding_fidelity(packs, cache_dir, outdir)

        print("[OPEN] sentiment shift (plot)...")
        plot_open_sentiment(packs, cache_dir, outdir)

        combine_open_double(outdir)  

        
        print("[OPEN] word counts (plot)...")
        plot_open_wordcounts_boxplots(packs, cache_dir, outdir)

def analyze_wordassoc(packs: Dict[str, DatasetPack], outdir: Path, cache_dir: Path, cache: CachePolicy) -> None:
    if cache.compute:
        print("[WORDASSOC] semantic distance energy (compute/cache)...")
        compute_wordassoc_semantic_distance_energy(packs, cache_dir, cache)

        print("[WORDASSOC] entropy |Δ| (compute/cache)...")
        compute_wordassoc_entropy_delta(packs, cache_dir, cache)

        print("[WORDASSOC] sentiment shift (compute/cache)...")
        compute_wordassoc_sentiment_shift(packs, cache_dir, cache)

        print("[WORDASSOC] keyness merged (compute/cache)...")
        compute_wordassoc_keyness_merged(packs, cache_dir, cache)

        
        print("[WORDASSOC] unique-answer % (compute/cache)...")
        compute_wordassoc_unique_answer_pct(packs, cache_dir, cache)

    if cache.plot:
        print("[WORDASSOC] semantic distance energy (plot)...")
        plot_wordassoc_semantic_distance_energy(packs, cache_dir, outdir)

        print("[WORDASSOC] entropy |Δ| (plot)...")
        plot_wordassoc_entropy_delta(packs, cache_dir, outdir)

        print("[WORDASSOC] sentiment shift (plot)...")
        plot_wordassoc_sentiment_shift(packs, cache_dir, outdir)
        
        
        combine_wordassoc_heatmaps_triptych(outdir)

        export_wordassoc_unique_answer_pct_tables(cache_dir, outdir)

def analyze_likert(packs: Dict[str, DatasetPack], outdir: Path, cache_dir: Path, cache: CachePolicy) -> None:
    if cache.compute:
        print("[LIKERT] grouped stacked tables (compute/cache)...")
        compute_likert_grouped_stacked_tables(packs, cache_dir, cache)

        print("[LIKERT] skepticism diff (compute/cache)...")
        compute_skepticism_diff(packs, cache_dir, cache)

    if cache.plot:
        print("[LIKERT] grouped stacked (plot)...")
        plot_likert_grouped_stacked(packs, cache_dir, outdir)

        print("[LIKERT] skepticism diff (plot)...")
        plot_skepticism_diff(packs, cache_dir, outdir)


def run_subgroup_suite(
    packs: Dict[str, DatasetPack],
    outdir: Path,
    cache: CachePolicy,
    *,
    bin_col: str,
    bin_value: str,
) -> None:
    ids = respondent_ids_for_bin(packs, bin_col, bin_value)
    if len(ids) < 10:
        print(f"[SUBGROUP] {bin_col}={bin_value}: too few respondents ({len(ids)}), skipping.")
        return

    packs_sub = filter_packs_by_respondent_ids(packs, ids)

    sub_slug = f"{_slug(bin_col)}={_slug(bin_value)}"
    cache_dir = outdir / "cache" / "subgroups" / sub_slug
    ensure_dir(cache_dir)

    
    plots_root = outdir / "plots" / "subgroups" / sub_slug
    ensure_dir(plots_root)

    
    
    tmp_outdir = outdir / "tmp_subgroup_out" / sub_slug
    ensure_dir(tmp_outdir / "plots")

    print(f"\n[SUBGROUP] Running {bin_col}={bin_value} (n={len(ids)})")
    analyze_open(packs_sub, tmp_outdir, cache_dir, cache)
    analyze_wordassoc(packs_sub, tmp_outdir, cache_dir, cache)
    analyze_likert(packs_sub, tmp_outdir, cache_dir, cache)

    
    
    for f in (tmp_outdir / "plots").rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(tmp_outdir / "plots")
        target = plots_root / rel
        ensure_dir(target.parent)
        try:
            f.replace(target)
        except Exception:
            shutil.copy2(f, target)

    





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of cached artifacts")
    parser.add_argument("--compute-only", action="store_true", help="Only compute + cache (no plots)")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from cache (no recomputation)")
    parser.add_argument("--subgroups", action="store_true", help="Run subgroup analyses (bins) in addition to overall")
    parser.add_argument("--subgroup-only", action="store_true", help="Run ONLY subgroup analyses (skip overall)")
    args = parser.parse_args()

    cache = CachePolicy(
        recompute=args.recompute,
        compute=not args.plot_only,
        plot=not args.compute_only,
    )

    set_seeds(SEED)
    outdir = Path(OUTDIR)
    ensure_dir(outdir)
    ensure_dir(outdir / "plots")

    cache_dir = outdir / "cache"
    ensure_dir(cache_dir)

    if "human" not in DATASETS:
        raise SystemExit("DATASETS must include a 'human' key.")

    packs: Dict[str, DatasetPack] = {}
    for lbl, p in DATASETS.items():
        packs[lbl] = load_pack(lbl, p)

    for lbl, pack in packs.items():
        pack.profiles = add_subgroup_labels_to_profiles(pack.profiles)

    print("Loaded datasets:", ", ".join(packs.keys()))
    print("Output:", outdir.resolve())
    print("Cache :", cache_dir.resolve())
    print("Mode  :", "plot-only" if args.plot_only else "compute-only" if args.compute_only else "compute+plot",
          "| recompute=" + str(args.recompute))

    
    if not args.subgroup_only:
        analyze_open(packs, outdir, cache_dir, cache)
        analyze_wordassoc(packs, outdir, cache_dir, cache)
        analyze_likert(packs, outdir, cache_dir, cache)

        
    export_keyness_merged_latex_table(cache_dir, outdir)

    
    if args.subgroups or args.subgroup_only:
        
        
        subgroup_plan = [
            ("esperienza_2", "Breve Esperienza"),
            ("esperienza_2", "Lunga Esperienza"),
            ("ruolo_2", "Sostegno"),
            ("ruolo_2", "Curricolare"),
            ("grado_2", "Primaria+Infanzia"),
            ("grado_2", "Secondaria (I+II)"),
        ]

        for bin_col, bin_value in subgroup_plan:
            run_subgroup_suite(packs, outdir, cache, bin_col=bin_col, bin_value=bin_value)

        print("Done.")


if __name__ == "__main__":
    main()