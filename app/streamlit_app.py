"""FIMI Case Classifier - Streamlit dashboard.

Run locally:   streamlit run app/streamlit_app.py
Reads the outputs written by `python -m fimi_classifier.pipeline`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fimi_classifier.stix_export import build_objects, to_bundle  # noqa: E402

OUTPUTS = ROOT / "outputs"

# Palette: one blue for single-series bars, a blue ramp for magnitude, and the
# first categorical slots (fixed order) for the few multi-series charts.
BLUE = "#2a78d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_GRAY = "#b5b4ae"

st.set_page_config(page_title="FIMI Case Classifier", layout="wide")

JSON_COLS = ["keywords", "countries", "languages", "outlets", "narrative_ids", "narratives",
             "narrative_confidence", "narrative_evidence", "target_countries", "target_country_names",
             "country_scores", "target_orgs", "disarm_techniques", "disarm_ids"]


@st.cache_data(show_spinner="Loading classified cases…")
def load(out_dir: str) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    out = Path(out_dir)
    df = pd.read_csv(out / "classified_cases.csv")
    for col in JSON_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) and v[:1] in "[{" else [])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["title", "summary", "disproof", "url"]:
        df[col] = df[col].fillna("")
    report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
    ev_path = out / "evaluation_vs_tags.csv"
    ev = pd.read_csv(ev_path) if ev_path.exists() else None
    return df, report, ev


def style(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), font=dict(size=13),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", title=None)
    return fig


def explode(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    e = df[["case_id", "date", col]].explode(col).dropna(subset=[col])
    return e.rename(columns={col: name})


# ---------------------------------------------------------------- data
if not (OUTPUTS / "classified_cases.csv").exists():
    st.error("No outputs found. Run `python -m fimi_classifier.pipeline --input data/raw --out outputs` first.")
    st.stop()

df, report, evaluation = load(str(OUTPUTS))

# ---------------------------------------------------------------- sidebar filters
st.sidebar.title("Filters")
min_d, max_d = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
all_narratives = sorted({n for ns in df["narratives"] for n in ns})
sel_narr = st.sidebar.multiselect("Narratives", all_narratives)
all_countries = sorted({c for cs in df["target_country_names"] for c in cs})
sel_ctry = st.sidebar.multiselect("Target countries", all_countries)
all_tactics = sorted({t["tactic"] for ts in df["disarm_techniques"] for t in ts})
sel_tactic = st.sidebar.multiselect("DISARM tactic", all_tactics)
min_conf = st.sidebar.slider("Min. narrative confidence", 0.0, 1.0, 0.0, 0.05)
query = st.sidebar.text_input("Search title / summary")

f = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    f = f[(f["date"] >= start) & (f["date"] < end) | f["date"].isna()]
if sel_narr:
    f = f[f["narratives"].map(lambda ns: bool(set(ns) & set(sel_narr)))]
if sel_ctry:
    f = f[f["target_country_names"].map(lambda cs: bool(set(cs) & set(sel_ctry)))]
if sel_tactic:
    f = f[f["disarm_techniques"].map(lambda ts: any(t["tactic"] in sel_tactic for t in ts))]
if min_conf:
    f = f[f["narrative_confidence"].map(lambda cs: max(cs or [0]) >= min_conf)]
if query:
    mask = f["title"].str.contains(query, case=False, regex=False) | f["summary"].str.contains(query, case=False, regex=False)
    f = f[mask]

st.sidebar.caption(f"{len(f):,} of {len(df):,} cases match")

# ---------------------------------------------------------------- header
st.title("FIMI Case Classifier")
st.caption(
    "EUvsDisinfo cases classified by narrative and target country, with suggested DISARM techniques "
    "and STIX 2.1 export. Labels are machine-generated triage aids and need analyst review."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cases", f"{len(f):,}")
k2.metric("Classified", f"{(f['primary_narrative'] != 'Other / unclassified').mean():.0%}" if len(f) else "–")
k3.metric("Narratives", len({n for ns in f["narratives"] for n in ns if n != 'Other / unclassified'}))
k4.metric("Target countries", len({c for cs in f["target_countries"] for c in cs}))
k5.metric("DISARM techniques", len({t for ts in f["disarm_ids"] for t in ts}))

if f.empty:
    st.warning("No cases match the current filters.")
    st.stop()

tab_over, tab_ctry, tab_disarm, tab_cases, tab_model, tab_stix = st.tabs(
    ["Narratives", "Countries", "DISARM", "Case explorer", "Model quality", "STIX export"]
)

narr = explode(f, "narratives", "narrative")
narr = narr[narr["narrative"] != "Other / unclassified"]
ctry = explode(f, "target_country_names", "country")

# ---------------------------------------------------------------- narratives
with tab_over:
    c1, c2 = st.columns([1, 1.3])
    counts = narr["narrative"].value_counts().sort_values()
    top_n = counts.sort_values(ascending=False).index[:12].tolist()
    fig = px.bar(counts.rename_axis("narrative").reset_index(name="cases"), x="cases", y="narrative",
                 orientation="h", title="Cases per narrative")
    fig.update_traces(marker_color=BLUE, hovertemplate="%{y}<br>%{x:,} cases<extra></extra>")
    fig.update_layout(showlegend=False)
    c1.plotly_chart(style(fig, 560), use_container_width=True)

    top = counts.sort_values(ascending=False).index[:5].tolist()
    t = narr.dropna(subset=["date"]).copy()
    t["narrative"] = t["narrative"].where(t["narrative"].isin(top), "Other narratives")
    t["quarter"] = t["date"].dt.to_period("Q").dt.start_time
    ts = t.groupby(["quarter", "narrative"]).size().reset_index(name="cases")
    colors = {n: CATEGORICAL[i] for i, n in enumerate(top)}
    colors["Other narratives"] = OTHER_GRAY
    fig = px.line(ts, x="quarter", y="cases", color="narrative", color_discrete_map=colors,
                  category_orders={"narrative": top + ["Other narratives"]},
                  title="Top 5 narratives over time (cases per quarter)")
    fig.update_traces(line_width=2, mode="lines+markers", marker_size=8)
    fig.update_layout(hovermode="x unified")
    c2.plotly_chart(style(fig, 560), use_container_width=True)

    with st.expander("Emergent topics (NMF) – themes the taxonomy may be missing"):
        st.dataframe(pd.DataFrame(report.get("topics", []))[["topic", "n_cases", "top_terms"]],
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- countries
with tab_ctry:
    from fimi_classifier.countries import GAZETTEER
    name_to_iso3 = {v[0]: v[1] for v in GAZETTEER.values()}
    cc = ctry["country"].value_counts().rename_axis("country").reset_index(name="cases")
    cc["iso3"] = cc["country"].map(name_to_iso3)
    fig = px.choropleth(cc.dropna(subset=["iso3"]), locations="iso3", color="cases", hover_name="country",
                        color_continuous_scale=SEQ, title="Cases by target country")
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)", lataxis_range=[-40, 80], lonaxis_range=[-130, 160])
    fig.update_layout(coloraxis_colorbar=dict(title="Cases"))
    st.plotly_chart(style(fig, 460), use_container_width=True)

    top_c = cc["country"].head(15).tolist()
    m = narr.merge(ctry, on=["case_id", "date"])
    m = m[m["country"].isin(top_c) & m["narrative"].isin(top_n)]
    heat = m.pivot_table(index="narrative", columns="country", values="case_id", aggfunc="count", fill_value=0)
    heat = heat.reindex(index=top_n, columns=[c for c in top_c if c in heat.columns]).fillna(0)
    fig = px.imshow(heat, color_continuous_scale=SEQ, aspect="auto", text_auto=True,
                    title="Narrative × target country (top 12 × top 15)")
    fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:,} cases<extra></extra>", xgap=2, ygap=2)
    st.plotly_chart(style(fig, 520), use_container_width=True)

    orgs = explode(f, "target_orgs", "org")["org"].value_counts()
    if not orgs.empty:
        st.caption("Targeted organisations: " + " · ".join(f"**{k}** {v:,}" for k, v in orgs.items()))

# ---------------------------------------------------------------- DISARM
with tab_disarm:
    rows = [{"case_id": r.case_id, "technique": f"{t['id']} {t['name']}", "tactic": f"{t['tactic_id']} {t['tactic']}",
             "confidence": t["confidence"]}
            for r in f.itertuples() for t in r.disarm_techniques]
    tech = pd.DataFrame(rows)
    c1, c2 = st.columns([1.2, 1])
    tc = tech.groupby(["tactic", "technique"]).agg(cases=("case_id", "nunique"),
                                                   avg_conf=("confidence", "mean")).reset_index()
    fig = px.treemap(tc, path=["tactic", "technique"], values="cases", color="avg_conf",
                     color_continuous_scale=SEQ, range_color=(0.3, 0.95),
                     title="Suggested DISARM techniques, grouped by tactic (colour = mean confidence)")
    fig.update_traces(hovertemplate="%{label}<br>%{value:,} cases<br>mean confidence %{color:.2f}<extra></extra>")
    c1.plotly_chart(style(fig, 560), use_container_width=True)

    nt = narr.merge(tech, on="case_id")
    nt = nt[nt["narrative"].isin(top_n)]
    hm = nt.pivot_table(index="narrative", columns="technique", values="case_id", aggfunc="nunique", fill_value=0)
    hm = hm.div(narr["narrative"].value_counts().reindex(hm.index), axis=0).round(2)
    fig = px.imshow(hm, color_continuous_scale=SEQ, aspect="auto", zmin=0, zmax=1,
                    title="Share of each narrative's cases per technique")
    fig.update_traces(hovertemplate="%{y}<br>%{x}<br>%{z:.0%} of cases<extra></extra>", xgap=2, ygap=2)
    fig.update_xaxes(tickangle=-40)
    c2.plotly_chart(style(fig, 560), use_container_width=True)
    st.caption("Techniques are suggested from narrative objectives (DISARM TA02 '5D') and textual cues; "
               "each case lists the exact reasons in the Case explorer.")

# ---------------------------------------------------------------- case explorer
with tab_cases:
    table = f[["date", "title", "primary_narrative", "target_country_names", "disarm_ids", "outlets"]].copy()
    table["date"] = table["date"].dt.date
    table["target_country_names"] = table["target_country_names"].map(", ".join)
    table["disarm_ids"] = table["disarm_ids"].map(", ".join)
    table["outlets"] = table["outlets"].map(", ".join)
    table = table.sort_values("date", ascending=False)
    st.dataframe(table.rename(columns={"target_country_names": "targets", "disarm_ids": "DISARM",
                                       "primary_narrative": "narrative"}),
                 use_container_width=True, hide_index=True, height=360)

    options = f.sort_values("date", ascending=False)
    pick = st.selectbox("Open a case", options.index,
                        format_func=lambda i: f"{options.at[i, 'date']:%Y-%m-%d} · {options.at[i, 'title'][:110]}"
                        if pd.notna(options.at[i, "date"]) else options.at[i, "title"][:110])
    case = f.loc[pick]
    st.subheader(case["title"])
    meta = [f"{case['date']:%d %b %Y}" if pd.notna(case["date"]) else "", ", ".join(case["outlets"]),
            ", ".join(case["languages"])]
    st.caption(" · ".join(m for m in meta if m))
    a, b = st.columns(2)
    with a:
        st.markdown("**Claim (summary)**")
        st.write(case["summary"] or "–")
        st.markdown("**EUvsDisinfo disproof**")
        st.write(case["disproof"] or "–")
        if case["url"]:
            st.markdown(f"[Open case on EUvsDisinfo]({case['url']})")
    with b:
        st.markdown("**Narratives**")
        for nid, name, conf in zip(case["narrative_ids"], case["narratives"], case["narrative_confidence"]):
            ev = ", ".join(case["narrative_evidence"].get(nid, [])) if isinstance(case["narrative_evidence"], dict) else ""
            st.markdown(f"- {name} — confidence {conf:.2f}" + (f"  \n  <small>evidence: {ev}</small>" if ev else ""),
                        unsafe_allow_html=True)
        st.markdown("**Target countries**: " + (", ".join(case["target_country_names"]) or "Unspecified"))
        if case["target_orgs"]:
            st.markdown("**Targeted organisations**: " + ", ".join(case["target_orgs"]))
        st.markdown("**Suggested DISARM techniques**")
        st.dataframe(pd.DataFrame([{"ID": t["id"], "Technique": t["name"], "Tactic": t["tactic"],
                                    "Confidence": t["confidence"], "Why": "; ".join(t["reasons"])}
                                   for t in case["disarm_techniques"]]),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- model quality
with tab_model:
    st.markdown(
        "The classifier never sees EUvsDisinfo's analyst **keyword tags**, so they serve as an independent proxy "
        "ground truth. *Weak* = this pipeline (seed rules + self-trained model, zero manual labels). "
        "*Supervised* = TF-IDF + logistic regression trained on the tags with 5-fold CV, a reference for what "
        "labelled data would add. Tags are incomplete, so precision is a lower bound."
    )
    if evaluation is not None:
        st.dataframe(evaluation, hide_index=True, use_container_width=True)
        ev = evaluation[~evaluation["narrative_id"].isin(["macro", "weighted"])]
        long = ev.melt(id_vars="narrative", value_vars=["weak_f1", "supervised_f1"], var_name="model", value_name="F1")
        long["model"] = long["model"].map({"weak_f1": "Weak supervision (no labels)", "supervised_f1": "Supervised (tags, 5-fold CV)"})
        fig = px.bar(long, y="narrative", x="F1", color="model", barmode="group", orientation="h",
                     color_discrete_sequence=CATEGORICAL[:2], title="F1 against EUvsDisinfo tags", range_x=(0, 1))
        fig.update_traces(hovertemplate="%{y}<br>F1 %{x:.2f}<extra></extra>")
        st.plotly_chart(style(fig, 620), use_container_width=True)
    with st.expander("Silver-label consistency (out-of-fold, per narrative)"):
        st.dataframe(pd.DataFrame(report.get("silver_label_cv_metrics", {}).values()), hide_index=True,
                     use_container_width=True)

# ---------------------------------------------------------------- STIX
with tab_stix:
    st.markdown(
        "Every case becomes a STIX 2.1 `report` that references `attack-pattern` objects (DISARM techniques), "
        "`location` objects (target countries), outlet `identity` objects and the case `url`. A shared "
        "`threat-actor` *uses* the techniques and *targets* the countries."
    )
    limit = st.number_input("Cases to export (most recent first, current filters)", 1, len(f), min(len(f), 500))
    if st.button("Build STIX bundle"):
        sub = f.sort_values("date", ascending=False).head(int(limit)).copy()
        bundle = to_bundle(build_objects(sub))
        payload = json.dumps(bundle, indent=2, ensure_ascii=False)
        types = pd.Series([o["type"] for o in bundle["objects"]]).value_counts()
        st.success(f"Bundle with {len(bundle['objects']):,} objects: " + ", ".join(f"{v} {k}" for k, v in types.items()))
        st.download_button("Download stix_bundle.json", payload, file_name="fimi_stix_bundle.json",
                           mime="application/json")
        st.code(json.dumps(next(o for o in bundle["objects"] if o["type"] == "report"), indent=2)[:4000], language="json")
