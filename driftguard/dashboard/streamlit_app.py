"""DriftGuard dashboard — watch a model drift and self-heal.

Run:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driftguard.config import Settings  # noqa: E402
from driftguard.monitor import run_loop  # noqa: E402

st.set_page_config(page_title="DriftGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ DriftGuard — Self-Healing MLOps")
st.caption("Monitor → retrain → validate → canary-promote / rollback. No human in the loop.")

with st.sidebar:
    st.header("Scenario")
    drift_at = st.slider("Inject drift at step", 4, 18, 10)
    n_batches = st.slider("Total steps", 16, 30, 24)
    tol = st.slider("Performance tolerance (×baseline)", 1.05, 1.6, 1.25, 0.05)
    chaos = st.button("💥 Inject drift & run", type="primary")

if chaos or "summary" not in st.session_state:
    s = Settings(drift_at=drift_at, n_batches=n_batches, perf_tolerance=tol)
    with st.spinner("Streaming, detecting drift, and self-healing…"):
        st.session_state.summary = run_loop(s)

summary = st.session_state.summary
d = asdict(summary)

c = st.columns(5)
c[0].metric("Detection lead time", f"{d['detection_lead_time']} steps")
c[1].metric("Recovery time", f"{d['recovery_time']} steps")
c[2].metric("Perf retained", f"{d['performance_retained_pct']}%")
c[3].metric("Model versions", d["n_versions"])
c[4].metric("Human interventions", d["human_interventions"])

# Build a per-step RMSE series from the event log.
rows = []
for e in d["events"]:
    det = e["detail"]
    rmse = det.get("window_rmse", det.get("rmse"))
    if rmse is not None and e["kind"] in {"ok", "drift_detected"}:
        rows.append({"step": e["step"], "rmse": rmse, "state": e["kind"]})
df = pd.DataFrame(rows)

st.subheader("Production RMSE over time")
if not df.empty:
    base = d["baseline_rmse"]
    line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="step:Q",
            y="rmse:Q",
            color=alt.Color(
                "state:N",
                scale=alt.Scale(domain=["ok", "drift_detected"], range=["#2ca02c", "#d62728"]),
            ),
            tooltip=["step", "rmse", "state"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [base * tol]}))
        .mark_rule(strokeDash=[5, 5], color="orange")
        .encode(y="y:Q")
    )
    drift_line = (
        alt.Chart(pd.DataFrame({"x": [d["drift_injected_step"]]}))
        .mark_rule(color="red")
        .encode(x="x:Q")
    )
    st.altair_chart((line + rule + drift_line).properties(height=340), use_container_width=True)
    st.caption("Red line = drift injected · orange dashed = drift threshold.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Event log")
    st.dataframe(pd.DataFrame(d["events"])[["step", "kind"]], height=320, use_container_width=True)
with col2:
    st.subheader("Healing decisions")
    decisions = [
        e for e in d["events"] if e["kind"] in {"drift_detected", "promoted", "rolled_back"}
    ]
    st.dataframe(
        pd.DataFrame([{"step": e["step"], "event": e["kind"], **e["detail"]} for e in decisions]),
        height=320,
        use_container_width=True,
    )
