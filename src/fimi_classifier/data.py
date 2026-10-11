"""Load the EUvsDisinfo dataset and normalise it to a common schema.

The Kaggle export (and other EUvsDisinfo dumps) do not always use the same
column names, so the loader maps known aliases onto a canonical schema:

    case_id, date, title, summary, disproof, keywords, countries,
    languages, outlets, url

Only ``title`` or ``summary`` is strictly required.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "case_id",
    "date",
    "title",
    "summary",
    "disproof",
    "keywords",
    "countries",
    "languages",
    "outlets",
    "url",
]

# Aliases are matched case-insensitively after stripping non-alphanumerics.
ALIASES: dict[str, list[str]] = {
    "case_id": ["case_id", "id", "caseid", "claim_id", "uuid"],
    "date": ["date", "date_of_publication", "published", "publication_date", "created", "case_date"],
    "title": ["title", "titles", "claim", "headline", "case_title", "name"],
    "summary": ["summary", "summary_text", "text", "description", "disinformation", "summary_of_the_disinformation", "content"],
    "disproof": ["disproof", "response", "information", "debunk", "disproof_text"],
    "keywords": ["keywords", "tags", "keyword", "issues"],
    "countries": ["countries", "country", "countries_discussed", "countriesdiscussed", "target_country", "countries_mentioned"],
    "languages": ["languages", "language", "target_language", "lang"],
    "outlets": ["outlets", "outlet", "publications", "media", "media_outlets", "sources"],
    "url": ["url", "news_url", "link", "links", "case_url", "permalink"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def map_columns(columns: list[str]) -> dict[str, str]:
    """Return a mapping {original_column: canonical_column}."""
    lookup = {_norm(c): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            original = lookup.get(_norm(alias))
            if original is not None and original not in mapping:
                mapping[original] = canonical
                break
    return mapping


def _sniff_sep(path: Path) -> str:
    """The Kaggle file (euvsdisinfo_v1_2.csv) is tab-separated despite its
    extension, so pick the delimiter that appears most in the header line."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    return max(["\t", ",", ";", "|"], key=header.count)


def _pick_csv(directory: Path) -> Path:
    """If a directory is given, pick the CSV that looks like the case table."""
    candidates = sorted(directory.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {directory}")
    best, best_score = None, -1
    for path in candidates:
        try:
            head = pd.read_csv(path, sep=_sniff_sep(path), nrows=50)
        except Exception:  # pragma: no cover - unreadable file
            continue
        mapped = set(map_columns(list(head.columns)).values())
        score = len(mapped) * 10 + (5 if {"title", "summary"} & mapped else 0)
        if score > best_score:
            best, best_score = path, score
    if best is None:
        raise ValueError(f"No readable CSV under {directory}")
    log.info("Using %s as the case table", best)
    return best


def _parse_dates(series: pd.Series) -> pd.Series:
    """Kaggle dates are dd/mm/yyyy; fall back to ISO / free-form parsing."""
    raw = series.astype("string").str.strip()
    # Some scraped cells contain junk around the date: keep the first dd/mm/yyyy.
    extracted = raw.str.extract(r"(\d{1,2}/\d{1,2}/\d{4})", expand=False)
    raw = extracted.fillna(raw)
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce", utc=True)
    missing = parsed.isna() & raw.notna()
    if missing.any():
        parsed[missing] = pd.to_datetime(raw[missing], errors="coerce", utc=True, dayfirst=True, format="mixed")
    return parsed


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)  # stray HTML
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_list(value) -> list[str]:
    """Split list-like cells: "a, b", "['a', 'b']", "a|b", "a;b"."""
    text = _clean_text(value)
    if not text:
        return []
    text = text.strip("[]")
    parts = re.split(r"\s*[,;|]\s*", text)
    return [p.strip(" '\"") for p in parts if p.strip(" '\"")]


def load_cases(path: str | Path) -> pd.DataFrame:
    """Load a CSV (or a directory containing CSVs) into the canonical schema."""
    path = Path(path)
    if path.is_dir():
        path = _pick_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        raw = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raw = pd.read_csv(path, sep=_sniff_sep(path), low_memory=False)

    mapping = map_columns(list(raw.columns))
    log.info("Column mapping: %s", mapping)
    df = raw.rename(columns=mapping)
    if not {"title", "summary"} & set(df.columns):
        raise ValueError(
            "Could not find a title or summary column. Columns found: "
            f"{list(raw.columns)}. Add an alias in fimi_classifier.data.ALIASES."
        )
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[CANONICAL_COLUMNS].copy()

    for col in ["title", "summary", "disproof", "url"]:
        df[col] = df[col].map(_clean_text)
    for col in ["keywords", "countries", "languages", "outlets"]:
        df[col] = df[col].map(_split_list)

    df["date"] = _parse_dates(df["date"])

    # Stable IDs: keep the dataset's own ID, otherwise hash title + summary.
    def _make_id(row) -> str:
        if _clean_text(row["case_id"]):
            return _clean_text(row["case_id"])
        if row["url"]:  # EUvsDisinfo report URLs end in a unique slug
            slug = row["url"].rstrip("/").rsplit("/", 1)[-1]
            if slug:
                return slug[:120]
        digest = hashlib.sha1((row["title"] + "|" + row["summary"]).encode()).hexdigest()
        return f"case-{digest[:12]}"

    df["case_id"] = df.apply(_make_id, axis=1)
    df["text"] = (df["title"] + ". " + df["summary"]).str.strip(". ")
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="case_id").reset_index(drop=True)
    log.info("Loaded %d cases", len(df))
    return df
