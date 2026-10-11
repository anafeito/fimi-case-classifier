"""Unit tests. Run with `pytest` (or `python tests/test_pipeline.py`)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fimi_classifier.countries import extract_countries  # noqa: E402
from fimi_classifier.data import load_cases, map_columns  # noqa: E402
from fimi_classifier.disarm import TECHNIQUES, suggest_techniques  # noqa: E402
from fimi_classifier.narratives import NARRATIVE_BY_ID, rule_scores  # noqa: E402
from fimi_classifier.pipeline import classify  # noqa: E402
from fimi_classifier.stix_export import build_objects, to_bundle  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "sample_cases.tsv"
STIX_ID = re.compile(r"^[a-z][a-z0-9-]+--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
STIX_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
REQUIRED = {
    "identity": {"name"}, "threat-actor": {"name"}, "attack-pattern": {"name"},
    "location": set(), "report": {"name", "published", "object_refs"},
    "relationship": {"relationship_type", "source_ref", "target_ref"}, "url": {"value"},
}


def test_column_mapping_kaggle_schema():
    cols = ["titles", "publication_date", "outlet", "country", "target_language", "keywords",
            "summary", "disproof", "news_url"]
    mapping = map_columns(cols)
    assert mapping["titles"] == "title"
    assert mapping["publication_date"] == "date"
    assert mapping["news_url"] == "url"
    assert mapping["country"] == "countries"


def test_load_sample():
    df = load_cases(SAMPLE)
    assert len(df) >= 20
    assert df["date"].notna().all()
    assert df["case_id"].is_unique
    assert isinstance(df.loc[0, "keywords"], list)


def test_rule_scores_find_expected_narratives():
    scores, ev = rule_scores("US biolabs in Ukraine develop bioweapons", "The Pentagon runs secret biolabs.")
    assert "N07" in scores and scores["N07"] >= 2
    scores, _ = rule_scores("MH17 was a Ukrainian provocation", "There is no evidence against Russia.")
    assert "N03" in scores


def test_country_extraction_excludes_russia_and_uses_metadata():
    r = extract_countries("Kyiv regime prepares provocation", "Moscow warns that Ukraine and NATO...", ["Baltic states"])
    assert r["target_countries"][0] in {"UA", "LT", "LV", "EE"}
    assert "RU" not in r["target_countries"]
    assert {"LT", "LV", "EE"} <= set(r["target_countries"])
    assert "NATO" in r["target_orgs"]


def test_disarm_suggestions_are_valid_and_explained():
    objectives = {nid: n.objective for nid, n in NARRATIVE_BY_ID.items()}
    sugg = suggest_techniques("The chemical attack was staged; there is no evidence.", ["N03"], objectives, [0.8])
    ids = [s["id"] for s in sugg]
    assert "T0075" in ids and "T0040" in ids
    for s in sugg:
        assert s["id"] in TECHNIQUES
        assert 0 < s["confidence"] <= 0.95
        assert s["reasons"]


def _classified():
    df = load_cases(SAMPLE)
    out, report = classify(df, n_topics=4)
    return out, report


def test_classify_end_to_end():
    out, report = _classified()
    assert len(out) == report["n_cases"]
    assert out["narrative_ids"].map(len).ge(1).all()
    assert (out["primary_narrative"] != "Other / unclassified").mean() > 0.4


def test_stix_bundle_is_spec_shaped_and_deterministic():
    out, _ = _classified()
    objs = build_objects(out, created="2026-01-01T00:00:00.000Z")
    again = build_objects(out, created="2026-01-01T00:00:00.000Z")
    assert [o["id"] for o in objs] == [o["id"] for o in again], "IDs must be deterministic"

    bundle = to_bundle(objs)
    assert bundle["type"] == "bundle" and STIX_ID.match(bundle["id"])
    ids = {o["id"] for o in bundle["objects"]}
    assert len(ids) == len(bundle["objects"]), "duplicate ids"
    for o in bundle["objects"]:
        assert o["spec_version"] == "2.1"
        assert STIX_ID.match(o["id"]) and o["id"].startswith(o["type"] + "--")
        assert REQUIRED[o["type"]] <= set(o), (o["type"], set(o))
        for key in ("created", "modified", "published"):
            if key in o:
                assert STIX_TS.match(o[key]), (key, o[key])
        for ref in o.get("object_refs", []) + [o.get("source_ref"), o.get("target_ref"), o.get("created_by_ref")]:
            if ref:
                assert ref in ids, f"dangling reference {ref}"
        for k in o:
            assert re.match(r"^[a-z0-9_]{3,250}$", k) or k in {"id"}, k
        if o["type"] == "attack-pattern":
            ref = o["external_references"][0]
            assert ref["source_name"] == "DISARM" and ref["external_id"] in TECHNIQUES
        if o["type"] == "location":
            assert re.match(r"^[A-Z]{2}$", o["country"])
    json.dumps(bundle)  # serialisable


if __name__ == "__main__":  # minimal runner when pytest is not installed
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
