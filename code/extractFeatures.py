
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import spacy






DEFAULT_SPACY_MODEL = "it_core_news_lg"

CANONICAL_SENTIMENT = {"positivo": 1, "neutro": 0, "negativo": -1}

LIKERT_TEXT_TO_SCORE = {
    "a: per niente d'accordo": 1,
    "per niente d'accordo": 1,
    "a": 1,
    "b: abbastanza in disaccordo": 2,
    "abbastanza in disaccordo": 2,
    "b": 2,
    "c: un po' d'accordo": 3,
    "un po' d'accordo": 3,
    "c": 3,
    "d: completamente d'accordo": 4,
    "completamente d'accordo": 4,
    "d": 4
}

LETTER_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")






def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

def norm_ws(s: str) -> str:
    return " ".join(s.strip().split())

def has_letter(s: str) -> bool:
    return bool(LETTER_RE.search(s))

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_spacy(model_name: str):
    try:
        nlp = spacy.load(model_name, disable=["ner"])
        global ITALIAN_STOPWORDS
        ITALIAN_STOPWORDS = set(w.lower() for w in nlp.Defaults.stop_words)
        STOPWORDS_KEEP = {
            "futuro", "generale", "nessuno", "nuovo", "gruppo", "lavoro", "tutti", "tutto",
            "insieme", "nulla", "nessuna", "talvolta", "volte", "troppo", "lungo", "mancanza",
            "abbastanza", "vario", "varia", "altro", "dentro", "persone", "sempre", "minimi", "forza",
            "male", "lontano", "relativo", "magari", "poco", "niente", "più", "meno", "mai",
            "pochissimo", "vita", "spesso", "tempo", "contro", "grande", "brava", "pieno",
            "molto", "dovunque", "parecchio", "esempio", "grazie", "comunque", "finalmente", "bene"
        }
        ITALIAN_STOPWORDS.difference_update(STOPWORDS_KEEP)

    except OSError as e:
        raise SystemExit(
            f"spaCy model '{model_name}' not found. Install with:\n"
            f"  python -m spacy download {model_name}"
        ) from e
    return nlp






@dataclass
class SentimentEntry:
    label: Optional[str]
    score: Optional[int]
    source: str  

def load_sentiment_json(path: Optional[str]) -> Dict[str, Tuple[Optional[str], List[str]]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    out: Dict[str, Tuple[Optional[str], List[str]]] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            txt = safe_str(k)
            if not txt:
                continue
            sent = None
            emos: List[str] = []
            if isinstance(v, list) and len(v) >= 2:
                sent = safe_str(v[0]).strip().lower() if safe_str(v[0]).strip() else None
                emos = v[1] if isinstance(v[1], list) else []
            elif isinstance(v, dict):
                sent = safe_str(v.get("sentiment", "")).strip().lower() or None
                emos = v.get("emotions", []) or []
            out[txt] = (sent, list(emos))
        return out

    if isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            txt = safe_str(rec.get("text", ""))
            if not txt:
                continue
            sent = safe_str(rec.get("sentiment", "")).strip().lower() or None
            emos = rec.get("emotions", []) or []
            out[txt] = (sent, list(emos))
        return out

    raise ValueError(f"Unsupported sentiment JSON format: {path}")

def build_sent_lookup(raw_map: Dict[str, Tuple[Optional[str], List[str]]]) -> Dict[str, Tuple[Optional[str], List[str]]]:
    lookup: Dict[str, Tuple[Optional[str], List[str]]] = {}
    for k, v in raw_map.items():
        if not k:
            continue
        lookup[k] = v
        lookup[norm_ws(k)] = v
        lookup[norm_ws(k).lower()] = v
    return lookup

def lookup_sentiment(text: str, lookup: Dict[str, Tuple[Optional[str], List[str]]]) -> SentimentEntry:
    raw = safe_str(text)
    if not raw.strip():
        return SentimentEntry(None, None, "missing")

    if raw in lookup:
        lab = lookup[raw][0]
        return SentimentEntry(lab, CANONICAL_SENTIMENT.get((lab or "").lower(), None), "raw")

    n = norm_ws(raw)
    if n in lookup:
        lab = lookup[n][0]
        return SentimentEntry(lab, CANONICAL_SENTIMENT.get((lab or "").lower(), None), "norm")

    l = n.lower()
    if l in lookup:
        lab = lookup[l][0]
        return SentimentEntry(lab, CANONICAL_SENTIMENT.get((lab or "").lower(), None), "lower")

    return SentimentEntry(None, None, "missing")






@dataclass
class ColumnMap:
    id_col: str
    profile_cols: List[str]
    q1_col: str
    q2_col: str
    q_cols: Dict[str, str]  

def infer_column_map(df: pd.DataFrame, id_col: str) -> ColumnMap:
    cols = list(df.columns)
    if id_col not in cols:
        raise ValueError(f"ID column '{id_col}' not found in Excel. Columns: {cols[:15]}...")

    if "Positiva" not in cols or "Negativa" not in cols:
        raise ValueError("Expected columns 'Positiva' and 'Negativa' in the Excel file.")

    q1_col = "Positiva"
    q2_col = "Negativa"

    idx_q2 = cols.index(q2_col)
    after = cols[idx_q2 + 1 : idx_q2 + 1 + 17]  
    if len(after) < 17:
        raise ValueError(f"Expected 17 columns after '{q2_col}' (for q3..q19), found {len(after)}.")

    q_cols: Dict[str, str] = {}
    for i, col in enumerate(after, start=3):
        q_cols[f"q{i}"] = col

    q1_idx = cols.index(q1_col)
    pre_q1 = cols[:q1_idx]
    profile_cols = [c for c in pre_q1 if c != id_col]

    return ColumnMap(
        id_col=id_col,
        profile_cols=profile_cols,
        q1_col=q1_col,
        q2_col=q2_col,
        q_cols=q_cols,
    )






def gulpease_from_doc(doc) -> Optional[float]:
    if doc is None:
        return None
    words = 0
    letters = 0
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if t.text.strip():
            words += 1
            letters += sum(1 for ch in t.text if ch.isalpha())
    sents = sum(1 for _ in doc.sents)
    if words == 0:
        return None
    return float(89 + (300 * sents - 10 * letters) / words)

def n_words_from_doc(doc) -> int:
    if doc is None:
        return 0
    return int(sum(1 for t in doc if (not t.is_space and not t.is_punct)))

def n_tokens_nonpunct(doc) -> int:
    if doc is None:
        return 0
    return int(sum(1 for t in doc if (not t.is_space and not t.is_punct)))

def tokens_for_keyness(doc) -> List[str]:
    if doc is None:
        return []
    out = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        
        
        
        if t.is_stop and t.text.lower().strip() in ITALIAN_STOPWORDS:
            continue
        txt = t.lemma_.lower().strip() if t.lemma_ else t.text.lower().strip()
        if len(txt) < 3:
            continue
        if len(txt) < 6 and " " in txt:
            print(f"Warning: short token with space in lemma: {txt}")
            continue
        if not has_letter(txt):
            continue
        out.append(txt)
    if len(out) == 0:
        print(f"Warning: no valid tokens for keyness in doc: {doc.text}")
    return out

def spellcheck_counts(doc) -> Tuple[int, int]:
	if doc is None:
        return 0, 0

    n_err = 0
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        txt = t.text.strip()
        if len(txt) < 3:
            continue
        if not has_letter(txt):
            continue

        
        if txt.isupper() and len(txt) <= 5:
            continue

        
        if t.is_oov:
            n_err += 1

    return (1 if n_err > 0 else 0), n_err






def build_profiles(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    cols = [cmap.id_col] + cmap.profile_cols
    cols = [c for c in cols if c in df.columns]
    prof = df[cols].copy()
    prof = prof.rename(columns={cmap.id_col: "respondent_id"})
    return prof

def build_open_long(
    df: pd.DataFrame,
    cmap: ColumnMap,
    nlp,
    sent_q1: Dict[str, Tuple[Optional[str], List[str]]],
    sent_q2: Dict[str, Tuple[Optional[str], List[str]]],
    dataset_label: str,
    min_chars: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    cov_rows = []

    for qkey, col, sent_lookup in [("q1", cmap.q1_col, sent_q1), ("q2", cmap.q2_col, sent_q2)]:
        raw_series = df[col].apply(safe_str)
        norm_series = raw_series.apply(norm_ws)

        keep_mask = norm_series.apply(lambda s: len(s) >= min_chars and has_letter(s))
        cov_rows.append({
            "dataset": dataset_label,
            "question": qkey,
            "n_total": int(len(df)),
            "n_kept": int(keep_mask.sum()),
            "keep_rate": float(keep_mask.mean()),
        })

        sub = df.loc[keep_mask, [cmap.id_col, col]].copy()
        for _, r in sub.iterrows():
            rid = r[cmap.id_col]
            raw = safe_str(r[col])
            norm = norm_ws(raw)
            doc = nlp(norm)

            sent = lookup_sentiment(raw, sent_lookup)
            spell_has, spell_n = spellcheck_counts(doc)

            rows.append({
                "dataset": dataset_label,
                "respondent_id": rid,
                "question": qkey,
                "raw": raw,
                "normalized_raw": norm,
                "entropy_key": norm.lower(),
                "n_words": n_words_from_doc(doc),
                "gulpease": gulpease_from_doc(doc),
                "spell_has_errors": int(spell_has),
                "spell_n_errors": int(spell_n),
                "sentiment_label": sent.label,
                "sentiment_score": sent.score,
                "sentiment_source": sent.source,
            })

    open_long = pd.DataFrame(rows)
    coverage = pd.DataFrame(cov_rows)
    return open_long, coverage

def build_wordassoc_long(
    df: pd.DataFrame,
    cmap: ColumnMap,
    nlp,
    sent_q3q16: Dict[str, Tuple[Optional[str], List[str]]],
    dataset_label: str,
    min_chars: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    cov_rows = []

    for qn in range(3, 17):
        qkey = f"q{qn}"
        col = cmap.q_cols[qkey]
        raw_series = df[col].apply(safe_str)
        norm_series = raw_series.apply(norm_ws)

        keep_mask = norm_series.apply(lambda s: len(s) >= min_chars and has_letter(s))
        cov_rows.append({
            "dataset": dataset_label,
            "question": qkey,
            "n_total": int(len(df)),
            "n_kept": int(keep_mask.sum()),
            "keep_rate": float(keep_mask.mean()),
        })

        sub = df.loc[keep_mask, [cmap.id_col, col]].copy()

        for _, r in sub.iterrows():
            rid = r[cmap.id_col]
            raw = safe_str(r[col])
            norm = norm_ws(raw)
            doc = nlp(norm)

            sent = lookup_sentiment(raw, sent_q3q16)

            n_tokens = n_tokens_nonpunct(doc)
            one_word = int(n_tokens == 1)
            spell_has, spell_n = spellcheck_counts(doc)

            rows.append({
                "dataset": dataset_label,
                "respondent_id": rid,
                "question": qkey,
                "raw": raw,
                "normalized_raw": norm,
                "entropy_key": norm.lower(),
                "n_tokens": n_tokens,
                "one_word": one_word,
                "spell_has_errors": int(spell_has),
                "spell_n_errors": int(spell_n),
                "sentiment_label": sent.label,
                "sentiment_score": sent.score,
                "sentiment_source": sent.source,
            })

    word_long = pd.DataFrame(rows)
    coverage = pd.DataFrame(cov_rows)
    return word_long, coverage

def build_likert(
    df: pd.DataFrame,
    cmap: ColumnMap,
    dataset_label: str,
) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rid = r[cmap.id_col]
        rec = {"dataset": dataset_label, "respondent_id": rid}

        scores = []
        for qn in (17, 18, 19):
            qkey = f"q{qn}"
            col = cmap.q_cols[qkey]
            txt = norm_ws(safe_str(r[col])).lower()
            score = LIKERT_TEXT_TO_SCORE.get(txt, None)
            rec[f"{qkey}_score"] = score
            scores.append(score)

        valid = [s for s in scores if s is not None]
        rec["skepticism_index_S"] = float(np.mean(valid)) if valid else None

        rows.append(rec)
    return pd.DataFrame(rows)

def build_keyness_token_tables(
    open_long: pd.DataFrame,
    word_long: pd.DataFrame,
    nlp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    open_rows = []
    for _, r in open_long.iterrows():
        doc = nlp(r["normalized_raw"])
        toks = tokens_for_keyness(doc)
        open_rows.append({
            "dataset": r["dataset"],
            "respondent_id": r["respondent_id"],
            "question": r["question"],
            "tokens": " ".join(toks),
            "n_tokens_keyness": int(len(toks)),
        })

    wa_rows = []
    for _, r in word_long.iterrows():
        doc = nlp(r["normalized_raw"])
        toks = tokens_for_keyness(doc)
        wa_rows.append({
            "dataset": r["dataset"],
            "respondent_id": r["respondent_id"],
            "question": r["question"],
            "tokens": " ".join(toks),
            "n_tokens_keyness": int(len(toks)),
        })

    return pd.DataFrame(open_rows), pd.DataFrame(wa_rows)






class Embedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit(
                "Embeddings requested but sentence-transformers is not installed.\n"
                "Install with: pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

def compute_and_save_embeddings(
    outdir: Path,
    open_long: pd.DataFrame,
    word_long: pd.DataFrame,
    embed_model: str,
    batch_size: int,
) -> Dict[str, Any]:
    emb = Embedder(embed_model, batch_size=batch_size)
    meta: Dict[str, Any] = {"embedding_model": embed_model, "batch_size": batch_size}

    for q in ("q1", "q2"):
        sub = open_long[open_long["question"] == q].copy()
        texts = sub["normalized_raw"].astype(str).tolist()
        vecs = emb.encode(texts)
        np.save(outdir / f"embeddings_open_{q}.npy", vecs)
        sub_map = sub[["respondent_id"]].reset_index(drop=True)
        sub_map.to_csv(outdir / f"map_open_{q}.csv", index=False)
        meta[f"open_{q}_shape"] = list(vecs.shape)

    sub = word_long.copy()
    texts = sub["normalized_raw"].astype(str).tolist()
    vecs = emb.encode(texts)
    np.save(outdir / "embeddings_wordassoc.npy", vecs)
    sub_map = sub[["respondent_id", "question"]].reset_index(drop=True)
    sub_map.to_csv(outdir / "map_wordassoc.csv", index=False)
    meta["wordassoc_shape"] = list(vecs.shape)

    return meta






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to Excel file (profile + questionnaire)")
    ap.add_argument("--id-col", default="ID", help="Name of respondent ID column in Excel")
    ap.add_argument("--dataset", required=True, help="Dataset label (e.g., human, gemma, llama)")
    ap.add_argument("--sent-q1", required=True, help="Sentiment JSON for q1")
    ap.add_argument("--sent-q2", required=True, help="Sentiment JSON for q2")
    ap.add_argument("--sent-q3q16", required=True, help="Sentiment JSON for q3-q16")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    ap.add_argument("--min-chars", type=int, default=3)

    ap.add_argument("--embeddings", action="store_true", help="Also compute and save embeddings (expensive).")
    ap.add_argument("--embed-model", default="sentence-transformers/LaBSE")
    ap.add_argument("--embed-batch-size", type=int, default=64)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = pd.read_excel(args.excel)
    nlp = load_spacy(args.spacy_model)

    cmap = infer_column_map(df, id_col=args.id_col)

    sent1 = build_sent_lookup(load_sentiment_json(args.sent_q1))
    sent2 = build_sent_lookup(load_sentiment_json(args.sent_q2))
    sent3 = build_sent_lookup(load_sentiment_json(args.sent_q3q16))

    profiles = build_profiles(df, cmap)
    open_long, cov_open = build_open_long(df, cmap, nlp, sent1, sent2, args.dataset, min_chars=args.min_chars)
    word_long, cov_word = build_wordassoc_long(df, cmap, nlp, sent3, args.dataset, min_chars=args.min_chars)
    likert = build_likert(df, cmap, args.dataset)

    key_open, key_word = build_keyness_token_tables(open_long, word_long, nlp)

    coverage = pd.concat([cov_open, cov_word], ignore_index=True)
    coverage.to_csv(outdir / "coverage_summary.csv", index=False)

    profiles.to_csv(outdir / "profiles.csv", index=False)
    open_long.to_csv(outdir / "open_long.csv", index=False)
    word_long.to_csv(outdir / "wordassoc_long.csv", index=False)
    likert.to_csv(outdir / "likert.csv", index=False)
    key_open.to_csv(outdir / "keyness_tokens_open.csv", index=False)
    key_word.to_csv(outdir / "keyness_tokens_wordassoc.csv", index=False)

    meta: Dict[str, Any] = {
        "dataset": args.dataset,
        "excel": str(Path(args.excel).resolve()),
        "id_col": args.id_col,
        "spacy_model": args.spacy_model,
        "min_chars": args.min_chars,
        "column_map": {
            "profile_cols": cmap.profile_cols,
            "q1_col": cmap.q1_col,
            "q2_col": cmap.q2_col,
            "q_cols": cmap.q_cols,
        },
        "sentiment_files": {
            "q1": str(Path(args.sent_q1).resolve()),
            "q2": str(Path(args.sent_q2).resolve()),
            "q3q16": str(Path(args.sent_q3q16).resolve()),
        },
        "n_rows_excel": int(len(df)),
        "coverage_overall": {
            "open_kept": int(len(open_long)),
            "wordassoc_kept": int(len(word_long)),
        }
    }

    if args.embeddings:
        emb_meta = compute_and_save_embeddings(
            outdir=outdir,
            open_long=open_long,
            word_long=word_long,
            embed_model=args.embed_model,
            batch_size=args.embed_batch_size,
        )
        meta["embeddings"] = emb_meta

    write_json(outdir / "metadata.json", meta)

    print("Done.")
    print(f"Outputs in: {outdir.resolve()}")
    print("Core tables:")
    print(" - profiles.csv")
    print(" - open_long.csv (now includes spell_has_errors, spell_n_errors)")
    print(" - wordassoc_long.csv (now includes spell_has_errors, spell_n_errors)")
    print(" - likert.csv")
    print(" - keyness_tokens_open.csv")
    print(" - keyness_tokens_wordassoc.csv")
    print(" - coverage_summary.csv")
    if args.embeddings:
        print("Embeddings saved as .npy + mapping CSVs.")


if __name__ == "__main__":
    main()