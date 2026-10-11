"""Narrative classification with weak supervision (no LLM, no manual labels).

How it works
------------
1. **Seed lexicons** (``NARRATIVES``) encode analyst knowledge about the
   recurring pro-Kremlin narratives documented by EUvsDisinfo. Each case gets a
   rule score per narrative (title hits count double).
2. Cases with a strong rule score become **silver labels**.
3. A **TF-IDF + logistic regression** model (one-vs-rest, multi-label) is trained
   on the silver labels. It generalises beyond the exact seed words, so cases
   that use different vocabulary for the same story still get classified.
4. Final labels combine rule evidence and model probability. Every label keeps
   the matched seed terms as *evidence*, so an analyst can see why it fired.
5. **NMF topic modelling** runs alongside to surface emergent themes that the
   taxonomy does not yet cover.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class Narrative:
    id: str
    name: str
    description: str
    seeds: tuple[str, ...]
    # Primary DISARM objective ("5D": dismiss, distort, distract, dismay, divide)
    objective: str = "distort"


NARRATIVES: tuple[Narrative, ...] = (
    Narrative(
        "N01", "Western aggression & NATO encirclement",
        "NATO/the West is the aggressor, threatens or encircles Russia; Russia only defends itself.",
        (r"\bnato\b", r"encircl\w*", r"expan(?:sion|ding) (?:to|towards) the east", r"military bloc",
         r"provok\w+ russia", r"proxy war", r"war (?:against|with) russia", r"attack russia", r"(?:american|us|nato) (?:military )?base", r"warmonger\w*", r"military conflict", r"western aggression", r"pentagon", r"anglo-saxon\w*",
         r"military bases?", r"forced (?:russia|moscow) to", r"russia had no choice", r"security guarantees"),
        "distort",
    ),
    Narrative(
        "N02", "Ukraine as Nazi, illegitimate or failed state",
        "Ukraine is run by Nazis/a junta, is not a real state, or is collapsing.",
        (r"\bnazi\w*", r"neo-?nazi\w*", r"denazif\w*", r"\bjunta\b", r"kyiv regime", r"kiev regime",
         r"bandera\w*", r"failed state", r"not a real (?:state|country)", r"artificial state",
         r"azov", r"puppet (?:regime|government)", r"\bcoup\b", r"coup d.?.?etat", r"maidan",
         r"not (?:a )?sovereign", r"no sovereignty", r"neo-?colon\w*", r"external (?:control|governance)", r"ukrain\w* (?:is|will) collaps\w*",
         r"collaps\w+.{0,30}ukrain\w*", r"ukrain\w*.{0,40}collaps\w+", r"western project", r"anti-russia project"),
        "distort",
    ),
    Narrative(
        "N03", "Denial of Russian responsibility",
        "Russia is not responsible for an attack, poisoning, crime or war crime; it was staged or a false flag.",
        (r"\bstaged\b", r"false flag", r"provocation", r"no evidence", r"fabricat\w+", r"\bhoax\b",
         r"mh17", r"skripal", r"navalny", r"bucha", r"novichok", r"nothing to do with (?:russia|moscow)",
         r"\bcrisis actors?\b", r"blamed russia", r"without (?:any )?proof", r"meddl\w+", r"interference in (?:the )?(?:us|american|elections?)",
         r"russiagate", r"mueller", r"not (?:a )?party to the conflict", r"russia (?:is|was) not (?:a )?part", r"no russian (?:troops|soldiers)", r"doping"),
        "dismiss",
    ),
    Narrative(
        "N04", "Western decline & moral decay",
        "The West/Europe is degenerate, collapsing morally, hostile to traditional values or LGBT-dominated.",
        (r"gayropa", r"\blgbt\w*", r"gender ideology", r"traditional values", r"moral (?:decay|decline)",
         r"degenera\w+", r"satanis\w+", r"paedophil\w*", r"pedophil\w*", r"decadent", r"family values",
         r"christian values", r"sodom", r"pervers\w+", r"moral values", r"same-sex", r"homosex\w+", r"spiritual\w*"),
        "divide",
    ),
    Narrative(
        "N05", "EU disintegration & dysfunction",
        "The EU is collapsing, divided, undemocratic, or a vassal of the US.",
        (r"\beu\b.{0,40}\b(?:collaps\w*|disintegrat\w*|fall apart|break up|crumbl\w*)", r"brussels bureaucra\w*",
         r"european union is (?:dying|collapsing|finished)", r"vassal", r"\beu dictat\w*",
         r"end of the (?:eu|european union)", r"(?:eu|europe) (?:has )?lost (?:its )?sovereignty", r"eurocrat\w*",
         r"leave the (?:eu|european union)", r"(?:eu|europe|european union) (?:is|has become) (?:a )?(?:totalitarian|dictatorship|occupied)",
         r"limited sovereignty", r"occupied by the (?:us|united states|americans)", r"\beu exit\b", r"\bexit (?:from )?the eu\b", r"brexit",
         r"take(?:s)? away (?:the )?(?:freedom|sovereignty)", r"eu (?:won.t|will not|refuses)", r"euro-?sceptic\w*"),
        "divide",
    ),
    Narrative(
        "N06", "Sanctions backfire & Western economic collapse",
        "Sanctions hurt Europe more than Russia; energy crisis, de-industrialisation, freezing Europeans.",
        (r"sanction\w*", r"energy crisis", r"\bgas (?:prices?|shortage|supplies)\b", r"freez\w+ (?:in|this) winter",
         r"de-?industriali\w+", r"inflation", r"economic collapse", r"recession", r"nord stream",
         r"\bfamine\b", r"food crisis", r"price cap", r"\bimf\b", r"\bdebt\b", r"poverty", r"food riots", r"gas transit", r"unemploy\w+"),
        "dismay",
    ),
    Narrative(
        "N07", "Biolabs, health & bioweapon conspiracies",
        "Secret US biolabs, engineered viruses, vaccine plots, depopulation.",
        (r"biolab\w*", r"biological weapons?", r"bio-?weapons?", r"\bvirus(?:es)?\b", r"covid", r"coronavirus",
         r"vaccin\w+", r"pandemic", r"depopulat\w+", r"pathogen\w*", r"pfizer", r"bill gates"),
        "dismay",
    ),
    Narrative(
        "N08", "Global elites & hidden puppet masters",
        "A hidden elite (Soros, globalists, deep state, Anglo-Saxons) secretly controls events.",
        (r"\bsoros\b", r"globalis\w+", r"deep state", r"puppet\w*", r"puppeteer\w*", r"world government",
         r"new world order", r"secret plan", r"behind the scenes", r"masterminded", r"rothschild\w*",
         r"golden billion", r"golden million", r"\bcabal\b", r"bilderberg", r"oligarchs? .{0,30}(?:made|control\w*)", r"western elites", r"secret elites?"),
        "distort",
    ),
    Narrative(
        "N09", "Colour revolutions & orchestrated protests",
        "Protests, elections and opposition movements are organised and paid for by the West.",
        (r"colou?r revolution\w*", r"orchestrat\w+ (?:by|from) (?:the )?(?:west|washington|us|cia)",
         r"\bcia\b", r"paid protesters?", r"foreign agents?", r"regime change", r"\bngos?\b.{0,40}(?:fund\w*|financ\w*)",
         r"state department", r"opposition (?:is|was) (?:funded|financed|controlled)", r"usaid", r"protests?.{0,40}(?:paid|funded|organi[sz]ed) by",
         r"behind (?:the )?.{0,25}protests", r"planning revolutions?", r"revolutions? in", r"maidan scenario", r"interfer\w+ (?:in|into) (?:the )?(?:domestic|internal) affairs", r"destabili[sz]\w+"),
        "dismiss",
    ),
    Narrative(
        "N10", "Historical revisionism",
        "Rewriting WWII and Soviet history; neighbours' independence or borders are illegitimate.",
        (r"world war (?:ii|two|2)", r"\bwwii\b", r"great patriotic war", r"molotov-ribbentrop", r"soviet (?:liberat\w+|union)",
         r"historical(?:ly)? (?:russian|part of russia)", r"rewrit\w+ history", r"glorif\w+ (?:nazi|fascis)\w*",
         r"monuments?", r"\bussr\b", r"liberators?", r"one people", r"hitler", r"stalin", r"holocaust", r"red army"),
        "distort",
    ),
    Narrative(
        "N11", "Western aid to Ukraine wasted or stolen",
        "Weapons and money sent to Ukraine are sold on the black market, stolen or pointless.",
        (r"black market", r"weapons? (?:are|were|is) (?:being )?(?:sold|resold|stolen)", r"(?:aid|money|funds) (?:is|are|was|were) (?:stolen|embezzl\w+|laundered)",
         r"\bcorrupt\w*", r"money laundering", r"zelensk\w+.{0,40}(?:villa|yacht|mansion|account)", r"himars", r"leopard",
         r"western weapons?", r"military aid", r"taxpayers?"),
        "dismay",
    ),
    Narrative(
        "N12", "Russophobia & Russia as victim",
        "Criticism of Russia is irrational Russophobia; Russians and Russian speakers are persecuted.",
        (r"russophob\w*", r"anti-russian", r"persecut\w+ (?:of )?russian", r"russian[- ]speak\w+", r"discriminat\w+",
         r"genocide (?:of|against) (?:the )?(?:russians?|donbas\w*)", r"cancel(?:led|ing)? (?:russian )?culture",
         r"hatred (?:of|towards) russia", r"demoni[sz]\w+ russia", r"compatriots", r"russian world", r"fight russia", r"russian language", r"great russian nation"),
        "divide",
    ),
    Narrative(
        "N13", "Nuclear & escalation threats",
        "World War III or nuclear catastrophe is imminent unless the West yields.",
        (r"nuclear", r"world war (?:iii|three|3)", r"\bwwiii\b", r"apocalyp\w+", r"armageddon", r"escalat\w+",
         r"zaporizh\w+ (?:nuclear|npp)", r"dirty bomb", r"radioactive"),
        "dismay",
    ),
    Narrative(
        "N14", "Migration as a threat",
        "Migrants and refugees (including Ukrainian refugees) cause crime and destroy Europe.",
        (r"migra\w+", r"refugees?", r"asylum", r"islamis\w+", r"replacement", r"invasion of europe",
         r"crime wave", r"ukrainian refugees"),
        "divide",
    ),
    Narrative(
        "N15", "'Civil war' in Ukraine & legitimacy of annexation",
        "The war in Donbas is an internal civil war; Crimea and occupied areas legitimately joined Russia.",
        (r"civil war", r"reunifi\w+", r"crimea\w*.{0,40}(?:referendum|returned|joined|russian)", r"referendum", r"annex\w+",
         r"\b(?:dpr|lpr|dnr|lnr)\b", r"people'?s republics?", r"punitive (?:operation|battalions?)", r"ukrainian army.{0,40}(?:shell\w*|kill\w*|attack\w*)",
         r"kyiv.{0,20}(?:shell\w*|bomb\w*)", r"militias?", r"self-defen[cs]e forces",
         r"war in (?:the )?donbas+", r"minsk (?:agreements?|accords?)", r"normandy format", r"separatis\w+"),
        "distort",
    ),
    Narrative(
        "N16", "Western media lies & censorship of 'alternative' voices",
        "Western media are propaganda and lie about Russia; bans on RT/Sputnik prove the West censors the truth.",
        (r"western media", r"mainstream media", r"fake news", r"censor\w*", r"freedom of speech", r"propaganda",
         r"information war\w*", r"\b(?:rt|sputnik)\b.{0,40}(?:ban\w*|block\w*|forbid\w*)", r"(?:ban\w*|block\w*) .{0,20}(?:russian|rt|sputnik) (?:channels?|media|tv)",
         r"radio (?:liberty|free europe)", r"journalists?.{0,30}(?:persecut\w+|arrest\w*|banned)", r"hate-?speech"),
        "dismiss",
    ),
    Narrative(
        "N17", "The West sponsors terrorism",
        "The US/West created or supports ISIS, Al-Qaeda and other terrorist groups, especially in Syria.",
        (r"\bisis\b", r"\bisil\b", r"daesh", r"terroris\w+", r"al-?qa.?ida", r"al-?nusra", r"jihad\w*", r"militants",
         r"white helmets", r"(?:us|american|western)-?(?:backed|supported|trained)", r"islamic state"),
        "distort",
    ),
)

NARRATIVE_BY_ID = {n.id: n for n in NARRATIVES}
OTHER_ID = "N00"
OTHER_NAME = "Other / unclassified"

_COMPILED = {n.id: [re.compile(p, re.IGNORECASE) for p in n.seeds] for n in NARRATIVES}


def rule_scores(title: str, summary: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Score one case against every narrative's seed lexicon."""
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for nid, patterns in _COMPILED.items():
        score, hits = 0.0, []
        for pat in patterns:
            t_hits = pat.findall(title or "")
            s_hits = pat.findall(summary or "")
            if t_hits or s_hits:
                score += 2.0 * min(len(t_hits), 2) + 1.0 * min(len(s_hits), 3)
                hits.append((t_hits or s_hits)[0].lower())
        if score:
            scores[nid] = score
            evidence[nid] = sorted(set(hits))
    return scores, evidence


@dataclass
class NarrativeClassifier:
    silver_threshold: float = 2.0     # rule score needed to become a silver label
    proba_threshold: float = 0.35     # model probability needed to add a label (tuned for F1)
    max_labels: int = 3
    min_positives: int = 5            # narratives with fewer silver labels are rule-only
    random_state: int = 42
    vectorizer: TfidfVectorizer = field(init=False)
    models: dict = field(default_factory=dict, init=False)
    metrics: dict = field(default_factory=dict, init=False)

    def _vectorizer(self, n_docs: int) -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=(1, 2), min_df=2 if n_docs > 50 else 1, max_df=0.9,
            sublinear_tf=True, stop_words="english", max_features=50_000,
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df["text"].fillna("").tolist()
        self.vectorizer = self._vectorizer(len(texts))
        X = self.vectorizer.fit_transform(texts)

        rules = [rule_scores(t, s) for t, s in zip(df["title"], df["summary"])]
        n = len(df)
        proba = np.zeros((n, len(NARRATIVES)))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for j, narrative in enumerate(NARRATIVES):
            y = np.array([r[0].get(narrative.id, 0) >= self.silver_threshold for r in rules], dtype=int)
            if y.sum() < self.min_positives or (n - y.sum()) < self.min_positives:
                proba[:, j] = np.nan
                continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
            # Out-of-fold predictions: an honest estimate of how well the model
            # reproduces the silver labels on cases it has not seen.
            oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            pred = (oof >= self.proba_threshold).astype(int)
            self.metrics[narrative.id] = {
                "narrative": narrative.name,
                "silver_positives": int(y.sum()),
                "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
                "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
                "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            }
            clf.fit(X, y)
            self.models[narrative.id] = clf
            proba[:, j] = clf.predict_proba(X)[:, 1]

        records = []
        for i in range(n):
            scores, evidence = rules[i]
            candidates = []
            for j, narrative in enumerate(NARRATIVES):
                rule = scores.get(narrative.id, 0.0)
                p = proba[i, j]
                rule_conf = min(0.95, rule / 6.0)   # never claim certainty from keywords alone
                model_conf = 0.0 if np.isnan(p) else float(p)
                fired = rule >= self.silver_threshold or model_conf >= self.proba_threshold
                if fired:
                    conf = min(0.95, max(rule_conf, model_conf)) if rule else model_conf * 0.8
                    candidates.append((conf, narrative.id, rule, model_conf))
            candidates.sort(reverse=True)
            candidates = candidates[: self.max_labels]
            if candidates:
                ids = [c[1] for c in candidates]
                confs = [round(c[0], 3) for c in candidates]
                ev = {nid: evidence.get(nid, ["(model: similar vocabulary)"]) for nid in ids}
            else:
                ids, confs, ev = [OTHER_ID], [0.0], {}
            records.append({
                "narrative_ids": ids,
                "narratives": [NARRATIVE_BY_ID[x].name if x in NARRATIVE_BY_ID else OTHER_NAME for x in ids],
                "narrative_confidence": confs,
                "primary_narrative": NARRATIVE_BY_ID[ids[0]].name if ids[0] in NARRATIVE_BY_ID else OTHER_NAME,
                "narrative_evidence": ev,
            })
        return pd.DataFrame(records, index=df.index)


def discover_topics(df: pd.DataFrame, n_topics: int = 12, n_terms: int = 10, random_state: int = 42):
    """Unsupervised NMF topics: helps spot narratives missing from the taxonomy."""
    texts = df["text"].fillna("").tolist()
    n_topics = max(2, min(n_topics, len(texts) // 10 or 2))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(texts) > 50 else 1, max_df=0.8,
                          stop_words="english", sublinear_tf=True, max_features=20_000)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for k, comp in enumerate(nmf.components_):
        top = terms[comp.argsort()[::-1][:n_terms]].tolist()
        topics.append({"topic": k, "label": " · ".join(top[:3]), "top_terms": top,
                       "n_cases": int((W.argmax(axis=1) == k).sum())})
    return topics, W.argmax(axis=1)
