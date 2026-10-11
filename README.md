# FIMI Case Classifier

**Automatic triage of Foreign Information Manipulation and Interference (FIMI) cases: narrative and target-country classification, DISARM technique suggestions, and STIX 2.1 export, using only local NLP (no LLM, no API keys, no manual labels).**

Built on the [EUvsDisinfo database](https://euvsdisinfo.eu/disinformation-cases/) (EEAS East StratCom Task Force), using the [Kaggle export](https://www.kaggle.com/datasets/imuhammad/euvsdisinfo-disinformation-database): **9,071 debunked pro-Kremlin disinformation cases, 2015–2020**.

```
EUvsDisinfo CSV ─► normalise ─► narratives (weak supervision) ─► target countries ─► DISARM techniques ─► CSV / STIX 2.1 ─► Streamlit dashboard
```

| | |
|---|---|
| **Narratives** | 17-narrative taxonomy of recurring pro-Kremlin narratives. Seed rules create silver labels, then a self-trained TF-IDF + logistic-regression model generalises them. Every label keeps its evidence terms. |
| **Target countries** | Gazetteer of ~75 countries (names, demonyms, capitals, leaders) plus the dataset's own "countries discussed" field. Russia is the narrator, so it is excluded from targets. EU / NATO / UN are tracked as targeted organisations. |
| **DISARM** | Explainable technique suggestions from the [DISARM Red Framework](https://github.com/DISARMFoundation/DISARMframeworks): each narrative's objective ("5D": Dismiss, Distort, Distract, Dismay, Divide), narrative-specific techniques and textual cues (e.g. *"staged"* → T0075 Dismiss, *"no evidence"* → T0040 Demand Insurmountable Proof). |
| **STIX 2.1** | One `report` per case linking `attack-pattern` (DISARM), `location`, outlet `identity` and `url` objects; a shared `threat-actor` *uses* techniques and *targets* countries. IDs are deterministic and objects are validated with the official [`stix2`](https://github.com/oasis-open/cti-python-stix2) library. |
| **Dashboard** | Streamlit + Plotly: narratives over time, target-country map, narrative × country heatmap, DISARM treemap, case explorer with evidence, model-quality page, filtered STIX download. |

## Results (full dataset, 9,071 cases)

EUvsDisinfo analysts tag each case with keywords (*"Nazi/Fascist"*, *"MH17"*, *"Colour revolutions"* …). **The classifier never sees these tags**, so I mapped them onto the taxonomy and used them as an independent proxy ground truth (`src/fimi_classifier/evaluate.py`).

<!-- RESULTS:START -->
| Narrative | Tagged cases | Precision | Recall | F1 (weak, 0 labels) | F1 (supervised ref.) |
|---|---:|---:|---:|---:|---:|
| Western aggression & NATO encirclement | 1,316 | 0.83 | 0.55 | 0.66 | 0.76 |
| Ukraine as Nazi, illegitimate or failed state | 1,258 | 0.82 | 0.49 | 0.62 | 0.72 |
| Denial of Russian responsibility | 1,003 | 0.80 | 0.66 | 0.73 | 0.85 |
| Western decline & moral decay | 303 | 0.83 | 0.38 | 0.52 | 0.64 |
| EU disintegration & dysfunction | 193 | 0.56 | 0.34 | 0.42 | 0.55 |
| Sanctions backfire & Western economic collapse | 483 | 0.70 | 0.44 | 0.54 | 0.66 |
| Biolabs, health & bioweapon conspiracies | 625 | 0.92 | 0.82 | 0.87 | 0.89 |
| Global elites & hidden puppet masters | 402 | 0.81 | 0.52 | 0.63 | 0.66 |
| Colour revolutions & orchestrated protests | 431 | 0.47 | 0.30 | 0.37 | 0.59 |
| Historical revisionism | 449 | 0.70 | 0.72 | 0.71 | 0.80 |
| Russophobia & Russia as victim | 1,037 | 0.86 | 0.37 | 0.51 | 0.61 |
| Nuclear & escalation threats | 165 | 0.68 | 0.62 | 0.65 | 0.69 |
| Migration as a threat | 394 | 0.85 | 0.63 | 0.73 | 0.82 |
| 'Civil war' in Ukraine & legitimacy of annexation | 738 | 0.66 | 0.51 | 0.58 | 0.78 |
| Western media lies & censorship of 'alternative' voices | 475 | 0.75 | 0.37 | 0.49 | 0.61 |
| The West sponsors terrorism | 600 | 0.84 | 0.73 | 0.78 | 0.81 |
| **Macro average** | 9,872 | **0.76** | **0.53** | **0.61** | 0.71 |
| **Support-weighted average** | 9,872 | **0.78** | **0.54** | **0.63** | 0.73 |

Coverage: **60%** of cases get at least one narrative label (the rest are kept as *Other / unclassified* rather than forced into a class).
<!-- RESULTS:END -->

Two takeaways:

* **Zero manual labels already gives ~78% precision** (support-weighted). Recall is the weak side: seed lexicons miss paraphrases, and tags are incomplete, so the measured precision is a lower bound.
* The **supervised reference** (trained on the tags, 5-fold CV) shows how much labelled data would add. That gap is the case for an analyst-in-the-loop workflow: review the weak labels, then retrain.

## DISARM techniques used

| ID | Technique | Tactic |
|---|---|---|
| T0002 | Facilitate State Propaganda | TA02 Plan Objectives |
| T0075 / T0076 / T0077 / T0078 / T0079 | Dismiss / Distort / Distract / Dismay / Divide | TA02 Plan Objectives |
| T0003 | Leverage Existing Narratives | TA14 Develop Narratives |
| T0022.001 | Amplify Existing Conspiracy Theory Narratives | TA14 Develop Narratives |
| T0040 | Demand Insurmountable Proof | TA14 Develop Narratives |
| T0068 | Respond to Breaking News Event or Active Crisis | TA14 Develop Narratives |
| T0023 / T0023.001 | Distort Facts / Reframe Context | TA06 Develop Content |
| T0084.003 | Deceptively Labelled or Translated | TA06 Develop Content |
| T0086.003 / T0087.002 | Deceptively Edit Images / Video (Cheap Fakes) | TA06 Develop Content |
| T0045 | Use Fake Experts | TA08 Conduct Pump Priming |

IDs, names and tactics were checked against the DISARM `techniques_index`. Suggestions are **triage aids**: the text of a claim rarely proves *how* it was produced (e.g. fake experts or doctored video), so cue-based techniques carry lower confidence and an explicit reason.

## Quick start

```bash
git clone https://github.com/<your-user>/fimi-case-classifier.git
cd fimi-case-classifier
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Data: download the Kaggle export into data/raw/ (see data/raw/README.md)
kaggle datasets download -d imuhammad/euvsdisinfo-disinformation-database -p data/raw --unzip

# 2) Pipeline (~1-2 min on a laptop)
python -m fimi_classifier.pipeline --input data/raw --out outputs     # add  --limit 1000  for a quick run
#   (or: pip install -e .  then  fimi-classify --input data/raw --out outputs)

# 3) Dashboard
streamlit run app/streamlit_app.py

# 4) Tests (uses the 300-case sample in data/sample/)
pytest
```

If you skip step 1, the repository already includes the classified outputs, so the dashboard works right after cloning. To try the pipeline without downloading anything, run it on `data/sample/sample_cases.tsv`.

> On Windows without `pip install -e .`, set `PYTHONPATH=src` first (`set PYTHONPATH=src`), or run from an IDE with `src` marked as a sources root.

## Outputs

| File | Content |
|---|---|
| `outputs/classified_cases.csv` | One row per case: narratives + confidence + evidence, target countries/orgs, DISARM suggestions with reasons, NMF topic |
| `outputs/case_narratives.csv`, `case_countries.csv`, `case_disarm.csv` | Long (tidy) tables, ready for Power BI / Tableau |
| `outputs/evaluation_vs_tags.csv` | Per-narrative precision / recall / F1 against EUvsDisinfo tags |
| `outputs/run_report.json` | Run metadata, silver-label CV metrics, emergent topics |
| `outputs/stix_bundle.json` | Full STIX 2.1 bundle (~31 MB, git-ignored; regenerate with the pipeline) |
| `examples/stix_bundle_sample.json` | STIX 2.1 bundle for 25 recent cases, for quick inspection |

Example STIX `report` (abridged):

```json
{
  "type": "report", "spec_version": "2.1", "id": "report--…",
  "name": "The UK fabricated evidence on Russia's involvement in the poisoning of the Skripals…",
  "published": "2018-…", "report_types": ["threat-report"],
  "labels": ["fimi", "disinformation", "narrative:Denial of Russian responsibility"],
  "confidence": 95,
  "object_refs": ["attack-pattern--… (T0075 Dismiss)", "attack-pattern--… (T0040)", "location--… (GB)", "identity--… (outlet)", "url--…", "threat-actor--…"],
  "x_fimi_disarm": [{"technique_id": "T0075", "confidence": 0.9, "reasons": ["narrative N03 -> objective 'dismiss'", "cue: denial / 'staged' claim ('fabricated')"]}]
}
```

## Project structure

```
fimi-case-classifier/
├── app/streamlit_app.py           # dashboard
├── src/fimi_classifier/
│   ├── data.py                    # schema-agnostic loader (column aliases, delimiter sniffing, date repair)
│   ├── narratives.py              # taxonomy, weak supervision, NMF topic discovery
│   ├── countries.py               # gazetteer-based target-country extraction
│   ├── disarm.py                  # DISARM technique catalogue + explainable suggestion rules
│   ├── stix_export.py             # STIX 2.1 bundle (deterministic IDs, stix2 validation)
│   ├── evaluate.py                # evaluation against EUvsDisinfo analyst tags
│   └── pipeline.py                # CLI entry point
├── data/sample/sample_cases.tsv   # 300-case sample for tests/demo
├── outputs/                       # pipeline results (used by the dashboard)
├── examples/stix_bundle_sample.json
└── tests/test_pipeline.py
```

## Design choices

* **Why no LLM?** Reproducible, free, fast (the whole dataset runs in about a minute on CPU), auditable, and it can run offline on sensitive data, which matters in government settings. The taxonomy and DISARM rules live in plain Python, so analysts can review and extend them. An LLM could be added later as a second-opinion labeller for low-confidence cases.
* **Explainability first.** Every narrative label lists the terms that triggered it, and every DISARM suggestion lists its reasons. An analyst can accept or reject a label in seconds.
* **Interoperability.** STIX 2.1 with DISARM as `attack-pattern` external references is the way FIMI-ISAC and DISARM-aligned teams exchange data. Deterministic IDs mean re-runs update objects instead of duplicating them on a TAXII server.

## Limitations

* The data covers 2015–2020 and only pro-Kremlin sources; the taxonomy reflects that.
* English summaries only (EUvsDisinfo writes summaries in English, while the original outlets use 20+ languages).
* Target country means "the country the claim is about or aimed at". That is not always the audience being targeted (see `languages` for audience).
* The DISARM suggestions are hypotheses for analysts to check, not attributions.

## Data & attribution

Cases © [EUvsDisinfo](https://euvsdisinfo.eu/) (EEAS East StratCom Task Force), via the Kaggle dataset by *imuhammad*. The raw dataset is not redistributed here. The 300-case sample and the derived outputs are included for demonstration only. Check the source terms before any reuse. The code is MIT-licensed.
