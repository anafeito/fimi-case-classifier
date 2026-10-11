"""FIMI Case Classifier - Streamlit dashboard.

Run locally:   streamlit run app/streamlit_app.py
Reads the outputs written by `python -m fimi_classifier.pipeline`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fimi_classifier.countries import GAZETTEER  # noqa: E402
from fimi_classifier.disarm import TECHNIQUES  # noqa: E402
from fimi_classifier.stix_export import build_objects, to_bundle  # noqa: E402

OUTPUTS = ROOT / "outputs"

# ------------------------------------------------------------------ look & feel
# One accent blue for single-series charts, a blue ramp for magnitudes and the
# first categorical slots (in fixed order) for the few multi-series charts.
BLUE = "#2a78d6"
GREY = "#c3c2b7"
SEQ = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED = "#1f1f1e", "#6b6a64"

# Short labels so long narrative names do not squeeze the charts.
SHORT = {
    "N01": "NATO & Western aggression", "N02": "Ukraine as Nazi / failed state",
    "N03": "Denial of Russian responsibility", "N04": "Western moral decay",
    "N05": "EU disintegration", "N06": "Sanctions backfire", "N07": "Biolabs & health conspiracies",
    "N08": "Hidden global elites", "N09": "Colour revolutions", "N10": "Historical revisionism",
    "N11": "Aid to Ukraine wasted", "N12": "Russophobia / Russia as victim", "N13": "Nuclear escalation",
    "N14": "Migration as a threat", "N15": "'Civil war' & annexation", "N16": "Western media lie",
    "N17": "West sponsors terrorism", "N00": "Unclassified",
}
TACTIC_ORDER = ["Plan Objectives", "Develop Narratives", "Develop Content", "Conduct Pump Priming"]
TACTIC_COLOR = {t: CATEGORICAL[i] for i, t in enumerate(TACTIC_ORDER)}
BASELINE = {"T0023", "T0002"}  # suggested for every case, so hidden from comparisons

st.set_page_config(page_title="FIMI Case Classifier", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
      [data-testid="stMetricValue"] {font-size: 1.8rem;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #6b6a64;}
      h1 {font-size: 2.1rem !important; margin-bottom: 0 !important;}
      h3 {font-size: 1.15rem !important; margin-top: 1.2rem !important;}
      .stTabs [data-baseweb="tab"] {font-size: 0.95rem; padding: 0.4rem 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, height=400, legend=False):
    """Shared Plotly styling: no in-chart titles (headings live in Streamlit), quiet axes."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=8, b=8), title=None,
        font=dict(family="Source Sans Pro, sans-serif", size=13, color=INK),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=legend, hoverlabel=dict(font_size=13, bgcolor="white"),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, title=None, linecolor="#d9d8d2", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", title=None, tickfont=dict(color=MUTED))
    return fig


def hbar(labels, values, hover=None, color=BLUE, height_per_bar=26, colors=None, value_fmt="{:,}"):
    """Horizontal bar chart with the value printed at the end of each bar."""
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors if colors is not None else color, cornerradius=4),
        text=[value_fmt.format(v) for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=MUTED, size=12),
        hovertext=hover if hover is not None else labels, hoverinfo="text+x",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK, size=13))
    fig.update_xaxes(showticklabels=False, showgrid=False, showline=False)
    style(fig, height=max(160, height_per_bar * len(labels) + 20))
    fig.update_layout(bargap=0.3, margin=dict(r=48))
    return fig


JSON_COLS = ["keywords", "countries", "languages", "outlets", "narrative_ids", "narratives",
             "narrative_confidence", "narrative_evidence", "target_countries", "target_country_names",
             "country_scores", "target_orgs", "disarm_techniques", "disarm_ids"]


@st.cache_data(show_spinner="Loading classified cases…")
def load(out_dir: str):
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


if not (OUTPUTS / "classified_cases.csv").exists():
    st.error("No outputs found. Run `python -m fimi_classifier.pipeline --input data/raw --out outputs` first.")
    st.stop()

df, report, evaluation = load(str(OUTPUTS))

# ------------------------------------------------------------------ sidebar filters
with st.sidebar:
    st.header("Filters")
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    narr_options = [nid for nid in SHORT if nid != "N00"]
    sel_narr = st.multiselect("Narratives", narr_options, format_func=lambda n: SHORT[n])
    all_countries = sorted({c for cs in df["target_country_names"] for c in cs})
    sel_ctry = st.multiselect("Target countries", all_countries)
    sel_tactic = st.multiselect("DISARM tactic", TACTIC_ORDER)
    min_conf = st.slider("Min. narrative confidence", 0.0, 0.95, 0.0, 0.05)
    query = st.text_input("Search title / summary", placeholder="e.g. MH17, biolab, Soros")

f = df
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    f = f[((f["date"] >= start) & (f["date"] < end)) | f["date"].isna()]
if sel_narr:
    f = f[f["narrative_ids"].map(lambda ns: bool(set(ns) & set(sel_narr)))]
if sel_ctry:
    f = f[f["target_country_names"].map(lambda cs: bool(set(cs) & set(sel_ctry)))]
if sel_tactic:
    f = f[f["disarm_techniques"].map(lambda ts: any(t["tactic"] in sel_tactic for t in ts))]
if min_conf:
    f = f[f["narrative_confidence"].map(lambda cs: max(cs or [0]) >= min_conf)]
if query:
    f = f[f["title"].str.contains(query, case=False, regex=False)
          | f["summary"].str.contains(query, case=False, regex=False)]

st.sidebar.caption(f"**{len(f):,}** of {len(df):,} cases match")

# ------------------------------------------------------------------ header
st.title("FIMI Case Classifier")
st.caption(
    "9,071 pro-Kremlin disinformation cases from the EUvsDisinfo database (2015–2020), classified by "
    "narrative and target country, with suggested DISARM techniques and STIX 2.1 export. "
    "Labels are machine-generated triage aids that need analyst review."
)

classified = f[f["narrative_ids"].map(lambda ns: ns != ["N00"])]
k = st.columns(5)
k[0].metric("Cases", f"{len(f):,}")
k[1].metric("With a narrative", f"{len(classified) / len(f):.0%}" if len(f) else "–")
k[2].metric("Narratives seen", len({n for ns in f["narrative_ids"] for n in ns if n != "N00"}))
k[3].metric("Target countries", len({c for cs in f["target_countries"] for c in cs}))
k[4].metric("DISARM techniques", len({t for ts in f["disarm_ids"] for t in ts}))

if f.empty:
    st.warning("No cases match the current filters.")
    st.stop()

tab_narr, tab_ctry, tab_disarm, tab_cases, tab_model, tab_stix = st.tabs(
    ["Narratives", "Countries", "DISARM", "Case explorer", "Model quality", "STIX export"]
)

narr = f[["case_id", "date", "narrative_ids"]].explode("narrative_ids").rename(columns={"narrative_ids": "nid"})
narr = narr[narr["nid"].notna() & (narr["nid"] != "N00")]
narr["narrative"] = narr["nid"].map(SHORT)
ctry = f[["case_id", "target_country_names"]].explode("target_country_names").dropna()
ctry = ctry.rename(columns={"target_country_names": "country"})

# ------------------------------------------------------------------ narratives
with tab_narr:
    counts = narr["narrative"].value_counts()
    st.subheader("Which stories are pushed most?")
    st.caption("Cases per narrative. A case can carry up to three narratives.")
    if counts.empty:
        st.info("No classified cases under the current filters.")
    else:
        st.plotly_chart(hbar(counts.index.tolist(), counts.values.tolist()), use_container_width=True)

        st.subheader("How the top narratives evolve")
        top = counts.index[:5].tolist()
        picked = st.multiselect("Narratives to compare", counts.index.tolist(), default=top, max_selections=6)
        t = narr[narr["narrative"].isin(picked)].dropna(subset=["date"]).copy()
        t["quarter"] = t["date"].dt.to_period("Q")
        last_full = (df["date"].max() + pd.Timedelta(days=1)).to_period("Q") - 1  # drop the partial last quarter
        t = t[t["quarter"] <= last_full]
        ts = (t.groupby(["quarter", "narrative"]).size().unstack(fill_value=0)
              .reindex(pd.period_range(t["quarter"].min(), last_full, freq="Q"), fill_value=0)
              if not t.empty else pd.DataFrame())
        if ts.empty:
            st.info("Pick at least one narrative.")
        else:
            fig = go.Figure()
            for i, name in enumerate(n for n in picked if n in ts.columns):
                fig.add_trace(go.Scatter(
                    x=ts.index.to_timestamp(), y=ts[name], name=name, mode="lines",
                    line=dict(width=2.2, color=CATEGORICAL[i], shape="spline", smoothing=0.6),
                    hovertemplate=f"{name}: %{{y}} cases<extra></extra>",
                ))
            style(fig, 420, legend=True)
            fig.update_layout(hovermode="x unified", legend=dict(y=-0.1))
            fig.update_xaxes(dtick="M12", tickformat="%Y")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Cases per quarter. The incomplete last quarter (July 2020) is left out. "
                       "The 2020 jump in biolab & health stories is the COVID-19 pandemic.")

    with st.expander("Emergent topics (NMF) – themes the taxonomy may be missing"):
        topics = pd.DataFrame(report.get("topics", []))
        if not topics.empty:
            topics["top terms"] = topics["top_terms"].map(lambda ts_: ", ".join(ts_[:8]))
            st.dataframe(topics[["n_cases", "top terms"]].rename(columns={"n_cases": "cases"})
                         .sort_values("cases", ascending=False), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ countries
with tab_ctry:
    cc = ctry["country"].value_counts().rename_axis("country").reset_index(name="cases")
    name_to_iso3 = {v[0]: v[1] for v in GAZETTEER.values()}
    cc["iso3"] = cc["country"].map(name_to_iso3)

    st.subheader("Who is targeted?")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.caption("Top 15 target countries (Russia excluded: it is the narrator)")
        top15 = cc.head(15)
        st.plotly_chart(hbar(top15["country"].tolist(), top15["cases"].tolist()), use_container_width=True)
    with right:
        st.caption("All target countries · colour capped so smaller targets stay visible; hover for exact counts")
        m = cc.dropna(subset=["iso3"])
        cap = max(1, int(m["cases"].quantile(0.9))) if len(m) else 1
        fig = go.Figure(go.Choropleth(
            locations=m["iso3"], z=m["cases"].clip(upper=cap), text=m["country"],
            customdata=m["cases"], colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
            marker_line_color="white", marker_line_width=0.5, zmin=0, zmax=cap,
            hovertemplate="<b>%{text}</b><br>%{customdata:,} cases<extra></extra>",
            colorbar=dict(title=dict(text="cases", font=dict(size=12)), thickness=10, len=0.6,
                          tickfont=dict(size=11, color=MUTED)),
        ))
        fig.update_geos(showframe=False, showcoastlines=False, showcountries=True, countrycolor="#e4e3dd",
                        showland=True, landcolor="#f4f3ef", projection_type="natural earth",
                        lataxis_range=[-10, 75], lonaxis_range=[-110, 120], bgcolor="rgba(0,0,0,0)")
        style(fig, 440)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which story for which country?")
    st.caption("Share of each country's cases that carry a narrative (top 10 countries × top 10 narratives)")
    top_c = cc["country"].head(10).tolist()
    top_n = narr["narrative"].value_counts().index[:10].tolist()
    mm = narr.merge(ctry, on="case_id")
    mm = mm[mm["country"].isin(top_c) & mm["narrative"].isin(top_n)]
    if not mm.empty:
        heat = mm.pivot_table(index="narrative", columns="country", values="case_id",
                              aggfunc="nunique", fill_value=0)
        per_country = ctry[ctry["country"].isin(top_c)].groupby("country")["case_id"].nunique()
        heat = heat.reindex(index=top_n, columns=top_c, fill_value=0)
        share = heat.div(per_country.reindex(heat.columns), axis=1).fillna(0)
        fig = go.Figure(go.Heatmap(
            z=share.values, x=share.columns, y=share.index, customdata=heat.values,
            colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)], zmin=0,
            text=[[f"{v:.0%}" if v >= 0.05 else "" for v in row] for row in share.values],
            texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2, showscale=False,
            hovertemplate="<b>%{y}</b> · %{x}<br>%{customdata} cases (%{z:.0%} of the country's cases)<extra></extra>",
        ))
        style(fig, 40 * len(top_n) + 60)
        fig.update_xaxes(side="top", tickfont=dict(color=INK))
        fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK), showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    orgs = f["target_orgs"].explode().dropna().value_counts()
    if not orgs.empty:
        st.caption("Targeted organisations: " + " · ".join(f"**{k_}** {v:,} cases" for k_, v in orgs.items()))

# ------------------------------------------------------------------ DISARM
with tab_disarm:
    rows = [{"case_id": r.case_id, "id": t["id"], "tactic": t["tactic"], "confidence": t["confidence"]}
            for r in f.itertuples() for t in r.disarm_techniques]
    tech = pd.DataFrame(rows)
    st.subheader("Which manipulation techniques show up?")
    st.caption("Suggested DISARM techniques, coloured by tactic. Two techniques are suggested for every case "
               "(T0023 Distort Facts and T0002 Facilitate State Propaganda) and are left out here.")
    tc = (tech[~tech["id"].isin(BASELINE)].groupby(["id", "tactic"])["case_id"].nunique()
          .reset_index(name="cases").sort_values("cases", ascending=False))
    if not tc.empty:
        labels = [f"{i} · {TECHNIQUES[i].name}" for i in tc["id"]]
        fig = hbar(labels, tc["cases"].tolist(), colors=[TACTIC_COLOR.get(t, GREY) for t in tc["tactic"]])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("  ".join(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{TACTIC_COLOR[t]};"
            f"margin-right:4px'></span><span style='color:{MUTED};font-size:0.85rem;margin-right:14px'>{t}</span>"
            for t in TACTIC_ORDER if t in set(tc["tactic"])), unsafe_allow_html=True)

    st.subheader("Technique profile of each narrative")
    st.caption("Share of a narrative's cases for which each technique is suggested")
    nt = narr.merge(tech[~tech["id"].isin(BASELINE)], on="case_id")
    top_n = narr["narrative"].value_counts().index[:12].tolist()
    nt = nt[nt["narrative"].isin(top_n)]
    if not nt.empty:
        hm = nt.pivot_table(index="narrative", columns="id", values="case_id", aggfunc="nunique", fill_value=0)
        cols = [c for c in tc["id"] if c in hm.columns]
        hm = hm.reindex(index=top_n, columns=cols, fill_value=0)
        hm = hm.div(narr["narrative"].value_counts().reindex(hm.index), axis=0).fillna(0)
        fig = go.Figure(go.Heatmap(
            z=hm.values, x=hm.columns, y=hm.index, zmin=0, zmax=1, xgap=2, ygap=2, showscale=False,
            colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
            text=[[f"{v:.0%}" if v >= 0.1 else "" for v in row] for row in hm.values],
            texttemplate="%{text}", textfont=dict(size=11),
            customdata=[[TECHNIQUES[c].name for c in hm.columns] for _ in hm.index],
            hovertemplate="<b>%{y}</b><br>%{x} %{customdata}<br>%{z:.0%} of cases<extra></extra>",
        ))
        style(fig, 36 * len(top_n) + 60)
        fig.update_xaxes(side="top", tickfont=dict(color=INK))
        fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK), showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Each case lists the exact reasons behind every suggestion in the Case explorer.")

# ------------------------------------------------------------------ case explorer
with tab_cases:
    st.subheader("Browse cases")
    table = f[["date", "title", "narrative_ids", "target_country_names", "disarm_ids"]].copy()
    table["date"] = table["date"].dt.date
    table["narrative"] = table["narrative_ids"].map(lambda ns: ", ".join(SHORT.get(n, n) for n in ns))
    table["targets"] = table["target_country_names"].map(", ".join)
    table["DISARM"] = table["disarm_ids"].map(lambda ids: ", ".join(i for i in ids if i not in BASELINE))
    table = table.sort_values("date", ascending=False)
    st.dataframe(table[["date", "title", "narrative", "targets", "DISARM"]], hide_index=True,
                 use_container_width=True, height=320,
                 column_config={"title": st.column_config.TextColumn(width="large")})

    options = f.sort_values("date", ascending=False)
    pick = st.selectbox(
        "Open a case", options.index,
        format_func=lambda i: (f"{options.at[i, 'date']:%Y-%m-%d} · " if pd.notna(options.at[i, "date"]) else "")
        + options.at[i, "title"][:110],
    )
    case = f.loc[pick]
    with st.container(border=True):
        st.markdown(f"#### {case['title']}")
        meta = [f"{case['date']:%d %b %Y}" if pd.notna(case["date"]) else "", ", ".join(case["outlets"]),
                ", ".join(case["languages"])]
        st.caption(" · ".join(m for m in meta if m))
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**What the outlet claimed**")
            st.write(case["summary"] or "–")
            st.markdown("**EUvsDisinfo disproof**")
            st.write(case["disproof"] or "–")
            if case["url"]:
                st.markdown(f"[Open the case on EUvsDisinfo ↗]({case['url']})")
        with b:
            st.markdown("**Narratives**")
            ev_map = case["narrative_evidence"] if isinstance(case["narrative_evidence"], dict) else {}
            for nid, conf in zip(case["narrative_ids"], case["narrative_confidence"]):
                ev = ", ".join(ev_map.get(nid, []))
                st.markdown(f"- {SHORT.get(nid, nid)} · confidence **{conf:.2f}**"
                            + (f"  \n  <span style='color:{MUTED};font-size:0.85rem'>evidence: {ev}</span>" if ev else ""),
                            unsafe_allow_html=True)
            st.markdown("**Target countries:** " + (", ".join(case["target_country_names"]) or "Unspecified"))
            if case["target_orgs"]:
                st.markdown("**Targeted organisations:** " + ", ".join(case["target_orgs"]))
            st.markdown("**Suggested DISARM techniques**")
            st.dataframe(pd.DataFrame([{"ID": t["id"], "Technique": t["name"], "Confidence": t["confidence"],
                                        "Why": "; ".join(t["reasons"])} for t in case["disarm_techniques"]]),
                         hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ model quality
with tab_model:
    st.subheader("How reliable are the narrative labels?")
    st.markdown(
        "The classifier never sees EUvsDisinfo's analyst **keyword tags**, so they work as an independent check. "
        "**Precision** = of the cases the model labels with a narrative, how many analysts tagged the same way. "
        "**Recall** = of the cases analysts tagged, how many the model finds. Tags are incomplete, so precision "
        "is a lower bound."
    )
    if evaluation is not None:
        ev = evaluation[~evaluation["narrative_id"].isin(["macro", "weighted"])].copy()
        ev["narrative"] = ev["narrative_id"].map(SHORT)
        w = evaluation[evaluation["narrative_id"] == "weighted"].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{w.weak_precision:.0%}")
        m2.metric("Recall", f"{w.weak_recall:.0%}")
        m3.metric("F1 (no labels)", f"{w.weak_f1:.2f}")
        m4.metric("F1 with analyst labels", f"{w.supervised_f1:.2f}", help="Reference model trained on the tags, 5-fold CV")

        ev = ev.sort_values("weak_precision", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ev["narrative"], x=ev["weak_precision"], name="Precision", orientation="h",
                             marker=dict(color=BLUE, cornerradius=3),
                             hovertemplate="%{y}<br>precision %{x:.0%}<extra></extra>"))
        fig.add_trace(go.Bar(y=ev["narrative"], x=ev["weak_recall"], name="Recall", orientation="h",
                             marker=dict(color=GREY, cornerradius=3),
                             hovertemplate="%{y}<br>recall %{x:.0%}<extra></extra>"))
        style(fig, 34 * len(ev) + 80, legend=True)
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, margin=dict(t=40),
                          legend=dict(y=1.02, yanchor="bottom"))
        fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK), showgrid=False)
        fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full table"):
            st.dataframe(evaluation, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ STIX
with tab_stix:
    st.subheader("Share the results as STIX 2.1")
    st.markdown(
        "Every case becomes a STIX `report` that references `attack-pattern` objects (DISARM techniques), "
        "`location` objects (target countries), outlet `identity` objects and the case `url`. A shared "
        "`threat-actor` *uses* the techniques and *targets* the countries."
    )
    limit = st.number_input("Cases to export (most recent first, current filters)", 1, len(f), min(len(f), 500))
    if st.button("Build STIX bundle", type="primary"):
        sub = f.sort_values("date", ascending=False).head(int(limit))
        bundle = to_bundle(build_objects(sub))
        payload = json.dumps(bundle, indent=2, ensure_ascii=False)
        types = pd.Series([o["type"] for o in bundle["objects"]]).value_counts()
        st.success(f"Bundle with {len(bundle['objects']):,} objects: " + ", ".join(f"{v} {k_}" for k_, v in types.items()))
        st.download_button("Download fimi_stix_bundle.json", payload, file_name="fimi_stix_bundle.json",
                           mime="application/json")
        st.code(json.dumps(next(o for o in bundle["objects"] if o["type"] == "report"), indent=2)[:4000],
                language="json")
