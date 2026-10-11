"""End-to-end pipeline: load -> classify -> DISARM -> export (CSV, JSON, STIX 2.1).

Usage::

    python -m fimi_classifier.pipeline --input data/raw/ --out outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .countries import extract_countries
from .data import load_cases
from .disarm import suggest_techniques
from .evaluate import evaluate
from .narratives import NARRATIVE_BY_ID, NarrativeClassifier, discover_topics
from .stix_export import HAS_STIX2, export_stix

log = logging.getLogger("fimi_classifier")


def classify(df: pd.DataFrame, n_topics: int = 12) -> tuple[pd.DataFrame, dict]:
    clf = NarrativeClassifier()
    narratives = clf.fit_predict(df)
    out = pd.concat([df, narratives], axis=1)

    countries = pd.DataFrame(
        [extract_countries(t, s, m) for t, s, m in zip(out["title"], out["summary"], out["countries"])],
        index=out.index,
    )
    out = pd.concat([out, countries], axis=1)

    objectives = {nid: n.objective for nid, n in NARRATIVE_BY_ID.items()}
    out["disarm_techniques"] = [
        suggest_techniques(text, ids, objectives, confs)
        for text, ids, confs in zip(out["text"], out["narrative_ids"], out["narrative_confidence"])
    ]
    out["disarm_ids"] = out["disarm_techniques"].map(lambda ts: [t["id"] for t in ts])

    topics, topic_idx = discover_topics(out, n_topics=n_topics)
    out["topic"] = topic_idx
    out["topic_label"] = [topics[k]["label"] for k in topic_idx]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": int(len(out)),
        "date_range": [str(out["date"].min()), str(out["date"].max())],
        "share_classified": round(float((out["primary_narrative"] != "Other / unclassified").mean()), 3),
        "silver_label_cv_metrics": clf.metrics,
        "topics": topics,
        "stix2_library_used": HAS_STIX2,
    }
    return out, report


def save_outputs(out: pd.DataFrame, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = out.copy()
    flat["date"] = flat["date"].dt.strftime("%Y-%m-%d")
    list_cols = [c for c in flat.columns if flat[c].map(lambda v: isinstance(v, (list, dict))).any()]
    for col in list_cols:
        flat[col] = flat[col].map(lambda v: json.dumps(v, ensure_ascii=False))
    flat.drop(columns=["text"]).to_csv(out_dir / "classified_cases.csv", index=False)

    # Long tables are what Power BI / Tableau users want.
    rows_n, rows_c, rows_t = [], [], []
    for r in out.itertuples(index=False):
        for nid, name, conf in zip(r.narrative_ids, r.narratives, r.narrative_confidence):
            rows_n.append({"case_id": r.case_id, "narrative_id": nid, "narrative": name, "confidence": conf})
        for iso, name in zip(r.target_countries, r.target_country_names):
            rows_c.append({"case_id": r.case_id, "iso2": iso, "country": name})
        for t in r.disarm_techniques:
            rows_t.append({"case_id": r.case_id, "technique_id": t["id"], "technique": t["name"],
                           "tactic_id": t["tactic_id"], "tactic": t["tactic"], "confidence": t["confidence"],
                           "reasons": "; ".join(t["reasons"])})
    pd.DataFrame(rows_n).to_csv(out_dir / "case_narratives.csv", index=False)
    pd.DataFrame(rows_c).to_csv(out_dir / "case_countries.csv", index=False)
    pd.DataFrame(rows_t).to_csv(out_dir / "case_disarm.csv", index=False)
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run(input_path: str | Path, out_dir: str | Path, limit: int | None = None, n_topics: int = 12) -> dict:
    out_dir = Path(out_dir)
    df = load_cases(input_path)
    if limit:
        df = df.sort_values("date", ascending=False).head(limit).reset_index(drop=True)
    out, report = classify(df, n_topics=n_topics)
    if (out["keywords"].map(len) > 0).mean() > 0.5:   # dataset carries analyst tags
        ev = evaluate(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ev.to_csv(out_dir / "evaluation_vs_tags.csv", index=False)
        report["evaluation_vs_tags"] = ev.to_dict(orient="records")
    save_outputs(out, report, out_dir)
    bundle = export_stix(out, out_dir / "stix_bundle.json")
    report["stix_objects"] = len(bundle["objects"])
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d cases and %d STIX objects to %s", len(out), len(bundle["objects"]), out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify EUvsDisinfo cases and export STIX 2.1")
    parser.add_argument("--input", required=True, help="CSV file or directory with the EUvsDisinfo export")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only process the N most recent cases")
    parser.add_argument("--topics", type=int, default=12, help="Number of NMF topics")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run(args.input, args.out, args.limit, args.topics)
    print(json.dumps({k: report[k] for k in ("n_cases", "share_classified", "stix_objects")}, indent=2))


if __name__ == "__main__":
    main()
