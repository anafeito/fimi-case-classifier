"""Evaluate narrative labels against EUvsDisinfo's own keyword tags.

EUvsDisinfo analysts tag every case with keywords ("Nazi/Fascist", "MH17",
"Colour revolutions", ...). These tags are **never used as features** by the
classifier (it only reads title + summary), so mapping them onto the narrative
taxonomy gives an independent, if imperfect, proxy ground truth.

Caveats: tags are incomplete (a case can push a narrative without carrying
the tag), so measured precision is a lower bound; and the mapping below is
itself an analyst judgement.

Two numbers are reported per narrative:
  * ``weak``       - the weak-supervision pipeline (seed rules + self-trained model)
  * ``supervised`` - a TF-IDF + logistic-regression model trained directly on the
                     tags with 5-fold cross-validation (an upper-bound reference
                     that needs labelled data, which new FIMI data rarely has).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .narratives import NARRATIVE_BY_ID

TAG_TO_NARRATIVE: dict[str, list[str]] = {
    "N01": ["nato", "us presence in europe", "eu/nato enlargement", "encircling russia", "warmongering",
            "war preparation", "missile defence system", "military exercise", "anglo-saxon", "inf treaty"],
    "N02": ["nazi/fascist", "ukrainian statehood", "ukrainian disintegration", "coup", "euromaidan", "junta",
            "abandoned ukraine"],
    "N03": ["mh17", "sergei skripal", "chemical weapons/attack", "false flag", "provocation", "novichok",
            "opcw", "ps752", "douma", "election meddling", "doping"],
    "N04": ["lgbt", "western values", "perversion", "european values"],
    "N05": ["eu disintegration", "euro-scepticism", "brexit"],
    "N06": ["sanctions", "economic difficulties", "energy", "gas", "nord stream 2"],
    "N07": ["coronavirus", "biological weapons", "vaccination", "lugar laboratory", "laboratory",
            "virus / bacteria threat", "bill gates", "health"],
    "N08": ["george soros", "secret elites / global elites", "deep state", "puppets", "new world order"],
    "N09": ["colour revolutions", "protest", "ngo", "destabilising russia"],
    "N10": ["historical revisionism", "wwii", "world war 2", "molotov-ribbentrop pact", "red army",
            "monuments", "joseph stalin", "holocaust"],
    "N12": ["russophobia", "anti-russian", "russian language", "ethnic russians", "russian world"],
    "N13": ["nuclear issues", "wwiii"],
    "N14": ["migration crisis", "refugees", "migration", "islamisation", "muslim/islam"],
    "N15": ["crimea", "illegal annexation", "referendum", "civil war", "dpr", "lpr", "dnr", "lnr"],
    "N16": ["media", "propaganda", "freedom of speech", "censorship", "fake news", "information war",
            "mainstream media"],
    "N17": ["terrorism", "daesh", "al-qaeda", "islamic extremism", "terrorism financing", "jihad", "white helmets"],
}


def tag_labels(keywords: pd.Series) -> pd.DataFrame:
    tags = keywords.map(lambda ks: {k.strip().lower() for k in ks})
    return pd.DataFrame({nid: tags.map(lambda t, v=set(v): bool(t & v)).astype(int)
                         for nid, v in TAG_TO_NARRATIVE.items()})


def evaluate(out: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    has_tags = out["keywords"].map(len) > 0
    df = out[has_tags].reset_index(drop=True)
    gold = tag_labels(df["keywords"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                          stop_words="english", max_features=50_000)
    X = vec.fit_transform(df["text"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rows = []
    for nid in TAG_TO_NARRATIVE:
        y = gold[nid].to_numpy()
        if y.sum() < 10:
            continue
        weak = df["narrative_ids"].map(lambda ids: nid in ids).astype(int).to_numpy()
        sup = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
                                X, y, cv=cv, method="predict_proba")[:, 1] >= 0.5
        rows.append({
            "narrative_id": nid,
            "narrative": NARRATIVE_BY_ID[nid].name,
            "tagged_cases": int(y.sum()),
            "weak_precision": precision_score(y, weak, zero_division=0),
            "weak_recall": recall_score(y, weak, zero_division=0),
            "weak_f1": f1_score(y, weak, zero_division=0),
            "supervised_f1": f1_score(y, sup, zero_division=0),
        })
    res = pd.DataFrame(rows)
    support = res["tagged_cases"]
    macro = res[["weak_precision", "weak_recall", "weak_f1", "supervised_f1"]].mean()
    micro_w = (res[["weak_precision", "weak_recall", "weak_f1", "supervised_f1"]].mul(support, axis=0).sum()
               / support.sum())
    res = pd.concat([res, pd.DataFrame([
        {"narrative_id": "macro", "narrative": "Macro average", "tagged_cases": int(support.sum()), **macro.to_dict()},
        {"narrative_id": "weighted", "narrative": "Support-weighted average", "tagged_cases": int(support.sum()), **micro_w.to_dict()},
    ])], ignore_index=True)
    num = ["weak_precision", "weak_recall", "weak_f1", "supervised_f1"]
    res[num] = res[num].astype(float).round(3)
    return res
