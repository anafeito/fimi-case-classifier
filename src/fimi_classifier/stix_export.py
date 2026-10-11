"""Export classified cases as a STIX 2.1 bundle.

Mapping
-------
=====================  ==============================================================
STIX object            Represents
=====================  ==============================================================
identity (system)      This classifier, as producer (``created_by_ref``)
identity (org)         Media outlets that published the claim
threat-actor           "Pro-Kremlin disinformation ecosystem" (EUvsDisinfo's attribution)
attack-pattern         DISARM techniques (``external_id`` + DISARM kill chain phase)
location               Target countries (ISO 3166-1 alpha-2)
url                    Link to the EUvsDisinfo case page
report                 One per case: narratives as labels, DISARM suggestions and
                       evidence as ``x_fimi_*`` custom properties
relationship           threat-actor *uses* attack-pattern / *targets* location;
                       report-level *related-to* outlet links
=====================  ==============================================================

Objects are built as plain dicts with deterministic IDs (re-running the pipeline
produces the same IDs, which matters for de-duplication on a TAXII server),
then parsed and serialised with the official ``stix2`` library, which validates
every object against the STIX 2.1 specification.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .disarm import DISARM_URL, TECHNIQUES
from .countries import ISO2_TO_NAME

try:  # the stix2 library is the reference implementation and validator
    import stix2
    HAS_STIX2 = True
except ImportError:  # pragma: no cover - exercised only without stix2 installed
    stix2 = None
    HAS_STIX2 = False

NAMESPACE = uuid.UUID("7c1f8f3e-2b4a-4d8e-9f3a-5e6b7c8d9e0f")  # project namespace for SDO ids
SCO_NAMESPACE = uuid.UUID("00abedb4-aa42-466c-9c01-fed23315a9b7")  # fixed by the STIX 2.1 spec
DISARM_KILL_CHAIN = "disarm"


def _sdo_id(stix_type: str, key: str) -> str:
    return f"{stix_type}--{uuid.uuid5(NAMESPACE, f'{stix_type}:{key}')}"


def _url_id(value: str) -> str:
    # STIX 2.1 §2.9: UUIDv5 over the JCS-canonicalised ID-contributing properties.
    canonical = json.dumps({"value": value}, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    return f"url--{uuid.uuid5(SCO_NAMESPACE, canonical)}"


def _ts(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        value = datetime(2015, 1, 1, tzinfo=timezone.utc)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def build_objects(df: pd.DataFrame, created: str | None = None) -> list[dict]:
    created = created or _ts(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))
    base = {"spec_version": "2.1", "created": created, "modified": created}
    objects: dict[str, dict] = {}

    producer_id = _sdo_id("identity", "fimi-case-classifier")
    objects[producer_id] = {
        **base, "type": "identity", "id": producer_id,
        "name": "FIMI Case Classifier",
        "identity_class": "system",
        "description": "Open-source NLP pipeline that classifies EUvsDisinfo cases by narrative and "
                       "target country and suggests DISARM techniques. Machine-generated; requires analyst review.",
    }
    common = {**base, "created_by_ref": producer_id}

    actor_id = _sdo_id("threat-actor", "pro-kremlin-ecosystem")
    objects[actor_id] = {
        **common, "type": "threat-actor", "id": actor_id,
        "name": "Pro-Kremlin disinformation ecosystem",
        "description": "Aggregate actor for outlets whose content EUvsDisinfo (EEAS East StratCom Task Force) "
                       "documented as pro-Kremlin disinformation. Attribution is EUvsDisinfo's, not this tool's.",
        "external_references": [{"source_name": "EUvsDisinfo", "url": "https://euvsdisinfo.eu/disinformation-cases/"}],
    }

    for row in df.itertuples(index=False):
        refs: list[str] = []

        for tech in row.disarm_techniques:
            tid = tech["id"]
            ap_id = _sdo_id("attack-pattern", tid)
            if ap_id not in objects:
                t = TECHNIQUES[tid]
                objects[ap_id] = {
                    **common, "type": "attack-pattern", "id": ap_id,
                    "name": t.name,
                    "external_references": [{"source_name": "DISARM", "external_id": tid,
                                             "url": DISARM_URL.format(tid=tid)}],
                    "kill_chain_phases": [{"kill_chain_name": DISARM_KILL_CHAIN, "phase_name": t.phase}],
                }
                rel = _sdo_id("relationship", f"{actor_id}-uses-{ap_id}")
                objects[rel] = {**common, "type": "relationship", "id": rel, "relationship_type": "uses",
                                "source_ref": actor_id, "target_ref": ap_id}
            refs.append(ap_id)

        for iso in row.target_countries:
            loc_id = _sdo_id("location", iso)
            if loc_id not in objects:
                objects[loc_id] = {**common, "type": "location", "id": loc_id,
                                   "name": ISO2_TO_NAME.get(iso, iso), "country": iso}
                rel = _sdo_id("relationship", f"{actor_id}-targets-{loc_id}")
                objects[rel] = {**common, "type": "relationship", "id": rel, "relationship_type": "targets",
                                "source_ref": actor_id, "target_ref": loc_id}
            refs.append(loc_id)

        for outlet in (row.outlets or [])[:10]:
            out_id = _sdo_id("identity", f"outlet:{outlet.lower()}")
            if out_id not in objects:
                objects[out_id] = {**common, "type": "identity", "id": out_id, "name": outlet,
                                   "identity_class": "organization", "description": "Media outlet (as listed by EUvsDisinfo)"}
            refs.append(out_id)

        if row.url:
            u_id = _url_id(row.url)
            objects.setdefault(u_id, {"type": "url", "spec_version": "2.1", "id": u_id, "value": row.url})
            refs.append(u_id)

        refs.append(actor_id)
        report_id = _sdo_id("report", str(row.case_id))
        published = _ts(row.date)
        report = {
            **common, "type": "report", "id": report_id,
            "name": (row.title or row.summary[:120] or str(row.case_id))[:250],
            "description": row.summary or row.title,
            "published": published,
            "report_types": ["threat-report"],
            "labels": ["fimi", "disinformation"] + [f"narrative:{n}" for n in row.narratives if n],
            "confidence": int(round(100 * max(row.narrative_confidence or [0]))),
            "object_refs": list(dict.fromkeys(refs)),
            "x_fimi_case_id": str(row.case_id),
            "x_fimi_narratives": [
                {"id": nid, "name": name, "confidence": conf}
                for nid, name, conf in zip(row.narrative_ids, row.narratives, row.narrative_confidence)
            ],
            "x_fimi_target_countries": list(row.target_countries),
            "x_fimi_target_orgs": list(row.target_orgs),
            "x_fimi_disarm": [
                {"technique_id": t["id"], "name": t["name"], "tactic_id": t["tactic_id"],
                 "confidence": t["confidence"], "reasons": t["reasons"]}
                for t in row.disarm_techniques
            ],
        }
        if row.url:
            report["external_references"] = [{"source_name": "EUvsDisinfo", "url": row.url}]
        objects[report_id] = report

    return list(objects.values())


def to_bundle(objects: list[dict]) -> dict:
    """Validate with stix2 when available and return the bundle as a dict."""
    if HAS_STIX2:
        parsed = [stix2.parse(o, allow_custom=True, version="2.1") for o in objects]
        bundle = stix2.Bundle(objects=parsed, allow_custom=True)
        return json.loads(bundle.serialize())
    bundle_id = f"bundle--{uuid.uuid5(NAMESPACE, 'bundle:' + ','.join(sorted(o['id'] for o in objects)))}"
    return {"type": "bundle", "id": bundle_id, "objects": objects}


def export_stix(df: pd.DataFrame, path: str | Path, created: str | None = None) -> dict:
    bundle = to_bundle(build_objects(df, created=created))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle
