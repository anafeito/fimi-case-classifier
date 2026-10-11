"""Suggest DISARM Red Framework techniques for each case.

The suggestions are *explainable*: every technique comes with the rule that
fired (a narrative, a textual cue, or a baseline) and a confidence score.
They are meant as analyst triage aids, not final attributions.

IDs, names and tactics were checked against the DISARM Red Framework
technique index (https://github.com/DISARMFoundation/DISARMframeworks,
generated_pages/techniques_index.md). Re-check them if your team pins a
different framework version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISARM_URL = "https://github.com/DISARMFoundation/DISARMframeworks/blob/main/generated_pages/techniques/{tid}.md"


@dataclass(frozen=True)
class Technique:
    id: str
    name: str
    tactic_id: str
    tactic: str

    @property
    def phase(self) -> str:
        return self.tactic.lower().replace(" ", "-")


TECHNIQUES: dict[str, Technique] = {t.id: t for t in [
    Technique("T0002", "Facilitate State Propaganda", "TA02", "Plan Objectives"),
    Technique("T0075", "Dismiss", "TA02", "Plan Objectives"),
    Technique("T0076", "Distort", "TA02", "Plan Objectives"),
    Technique("T0077", "Distract", "TA02", "Plan Objectives"),
    Technique("T0078", "Dismay", "TA02", "Plan Objectives"),
    Technique("T0079", "Divide", "TA02", "Plan Objectives"),
    Technique("T0003", "Leverage Existing Narratives", "TA14", "Develop Narratives"),
    Technique("T0022.001", "Amplify Existing Conspiracy Theory Narratives", "TA14", "Develop Narratives"),
    Technique("T0040", "Demand Insurmountable Proof", "TA14", "Develop Narratives"),
    Technique("T0068", "Respond to Breaking News Event or Active Crisis", "TA14", "Develop Narratives"),
    Technique("T0023", "Distort Facts", "TA06", "Develop Content"),
    Technique("T0023.001", "Reframe Context", "TA06", "Develop Content"),
    Technique("T0084.003", "Deceptively Labelled or Translated", "TA06", "Develop Content"),
    Technique("T0086.003", "Deceptively Edit Images (Cheap Fakes)", "TA06", "Develop Content"),
    Technique("T0087.002", "Deceptively Edit Video (Cheap Fakes)", "TA06", "Develop Content"),
    Technique("T0045", "Use Fake Experts", "TA08", "Conduct Pump Priming"),
]}

OBJECTIVE_TO_TECHNIQUE = {
    "dismiss": "T0075", "distort": "T0076", "distract": "T0077", "dismay": "T0078", "divide": "T0079",
}

# Narrative-level suggestions (beyond the objective technique).
NARRATIVE_TECHNIQUES: dict[str, list[str]] = {
    "N02": ["T0003"],
    "N03": ["T0040"],
    "N07": ["T0022.001"],
    "N08": ["T0022.001"],
    "N09": ["T0022.001"],
    "N10": ["T0023.001", "T0003"],
    "N12": ["T0003"],
    "N04": ["T0003"],
    "N14": ["T0003"],
    "N15": ["T0023.001"],
    "N16": ["T0040"],
    "N17": ["T0003"],
}

# Textual cues -> technique. (regex, technique id, confidence, label)
CUES: list[tuple[str, str, float, str]] = [
    (r"\b(?:staged|false flag|hoax|fabricat\w+|crisis actors?)\b", "T0075", 0.75, "denial / 'staged' claim"),
    (r"\bno (?:evidence|proof)\b|\bwithout (?:any )?(?:evidence|proof)\b|\bunproven\b|\bnot a single (?:fact|proof)\b", "T0040", 0.7, "demands proof / 'no evidence'"),
    (r"\bwhat about\b|\bmeanwhile\b.{0,60}\b(?:west|us|nato|eu)\b|\bdouble standards?\b|\bhypocri\w+", "T0077", 0.6, "whataboutism"),
    (r"\b(?:secret(?:ly)?|hidden|behind the scenes|plot|masterminded|orchestrat\w+|controlled by|puppet\w*)\b", "T0022.001", 0.55, "hidden-plot framing"),
    (r"\b(?:catastroph\w+|apocalyp\w+|collapse|famine|freez\w+|world war|nuclear|doom\w*|disaster)\b", "T0078", 0.55, "fear / dismay language"),
    (r"\b(?:divid\w+|split|rift|hatred|civil war|minorit\w+|russian[- ]speak\w+)\b", "T0079", 0.5, "wedge / division language"),
    (r"\b(?:expert|analyst|scientist|professor|political scientist|former (?:cia|officer|diplomat))s?\b", "T0045", 0.4, "appeal to (unnamed) experts"),
    (r"\b(?:western media|the (?:guardian|bbc|new york times|washington post|der spiegel)|cnn)\b.{0,40}\b(?:admit\w*|confess\w*|acknowledg\w+)", "T0084.003", 0.5, "'Western media admits' framing"),
    (r"\b(?:admitted|confessed)\b", "T0084.003", 0.35, "claimed admission"),
    (r"\b(?:photo|image|picture)s?\b", "T0086.003", 0.3, "image-based claim"),
    (r"\b(?:video|footage)\b", "T0087.002", 0.3, "video-based claim"),
    (r"\b(?:explosion|attack|crash|shooting|election|earthquake|outbreak|pandemic|covid|strike|assassinat\w+)\b", "T0068", 0.45, "exploits breaking news / crisis"),
    (r"\b(?:world war (?:ii|two|2)|wwii|19[34]\d|soviet|ussr|historic\w*)\b", "T0023.001", 0.45, "historical reframing"),
]
_CUES = [(re.compile(p, re.IGNORECASE), tid, conf, label) for p, tid, conf, label in CUES]


def suggest_techniques(text: str, narrative_ids: list[str], objectives: dict[str, str],
                       narrative_confidence: list[float] | None = None, max_techniques: int = 6) -> list[dict]:
    """Return DISARM technique suggestions sorted by confidence."""
    found: dict[str, dict] = {}

    def add(tid: str, conf: float, reason: str) -> None:
        tech = TECHNIQUES[tid]
        cur = found.get(tid)
        if cur is None:
            found[tid] = {"id": tid, "name": tech.name, "tactic_id": tech.tactic_id, "tactic": tech.tactic,
                          "confidence": round(conf, 2), "reasons": [reason]}
        else:
            # independent pieces of evidence reinforce each other
            cur["confidence"] = round(min(0.95, 1 - (1 - cur["confidence"]) * (1 - conf)), 2)
            if reason not in cur["reasons"]:
                cur["reasons"].append(reason)

    # Baselines: every EUvsDisinfo case is a debunked claim from pro-Kremlin media.
    add("T0023", 0.6, "baseline: case debunked by EUvsDisinfo")
    add("T0002", 0.5, "baseline: pro-Kremlin state-aligned media ecosystem")

    confs = narrative_confidence or [0.6] * len(narrative_ids)
    for nid, nconf in zip(narrative_ids, confs):
        objective = objectives.get(nid)
        if objective:
            add(OBJECTIVE_TO_TECHNIQUE[objective], 0.4 + 0.4 * nconf, f"narrative {nid} -> objective '{objective}'")
        for tid in NARRATIVE_TECHNIQUES.get(nid, []):
            add(tid, 0.35 + 0.4 * nconf, f"narrative {nid}")

    for pattern, tid, conf, label in _CUES:
        m = pattern.search(text or "")
        if m:
            add(tid, conf, f"cue: {label} ('{m.group(0)[:40]}')")

    ranked = sorted(found.values(), key=lambda d: d["confidence"], reverse=True)
    return ranked[:max_techniques]
