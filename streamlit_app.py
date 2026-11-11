"""Epoch Research Dashboard: explain and inspect saved experiment results."""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_utils import (
    feature_shift,
    published_batch_check,
    published_day_check,
    published_primary,
    published_report,
    window_metrics,
)

ROOT = Path(__file__).resolve().parent
ORDER = ["Static", "Periodic", "Online", "Adwin"]
METRICS = {
    "Accuracy": "accuracy",
    "Balanced Accuracy": "balanced_accuracy",
    "F1 (UP)": "f1_up",
}

st.set_page_config(page_title="Epoch | Research Dashboard", page_icon="📈", layout="wide")

st.title("Epoch")
st.caption("A Study of Temporal Data Shift and Model Adaptation")
st.write(
    "A read-only research dashboard comparing four ways to adapt a classifier "
    "as new observations arrive. The dashboard shows measured results; "
    "opening it does not train a model or change the dataset."
)

try:
    primary, source, summary = published_primary(ROOT)
    day = published_day_check(ROOT)
    batches = published_batch_check(ROOT)
    report = published_report(ROOT)
except (OSError, KeyError, ValueError, TypeError) as error:
    st.error(f"Could not load research results: {error}")
    st.stop()

with st.sidebar:
    st.header("About this study")
    st.markdown(
        "**Dataset:** OpenML Electricity (ID 151)\n\n"
        "**Task:** Predict the UP or DOWN category\n\n"
        "**Primary inputs:** period, nswdemand, vicdemand, transfer\n\n"
        "**Evaluation:** Test each ordered batch before learning its labels"
    )
    st.caption("Not a real-time electricity price forecasting system.")
    st.divider()
    st.markdown("[GitHub repository](https://github.com/riri242/Epoch)")
    st.markdown("[Research methodology](https://github.com/riri242/Epoch/blob/main/docs/methodology.md)")

if source == "published":
    st.info(
        "Showing the reviewed results published in docs/research_report.md. "
        "To unlock per-batch and feature-shift charts locally, run "
        "`python scripts/run_all.py` from the project root."
    )
else:
    st.success("Showing the most recent locally generated experiment results.")

main_tab, timeline_tab, shift_tab, checks_tab, report_tab = st.tabs(
    ["Overview", "Performance over time", "Data shift", "Robustness checks", "Research report"]
)

with main_tab:
    top = primary.sort_values("Accuracy", ascending=False).iloc[0]
    n_records = summary["dataset_rows"] if summary else None
    if n_records is None:
        match = re.search(r"([\d,]+) observations", report)
        n_records = match.group(1) if match else "45,312"
    left, middle, right = st.columns(3)
    left.metric("Dataset observations", f"{n_records:,}" if isinstance(n_records, int) else str(n_records))
    middle.metric("Evaluated records", f"{summary['evaluated_rows']:,}" if summary else "31,719")
    right.metric("Highest accuracy", f"{top['Accuracy']:.2f}%", top["Strategy"])
    st.subheader("Compare adaptation policies")
    measure = st.selectbox("Performance measure", list(METRICS), index=0)
    st.bar_chart(primary.set_index("Strategy")[[measure]], y_label=f"{measure} (%)")
    st.dataframe(
        primary.style.format({key: "{:.2f}%" for key in METRICS if key in primary.columns}),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "Export comparison (CSV)", primary.to_csv(index=False).encode("utf-8"),
        file_name="epoch_primary_metrics.csv", mime="text/csv",
    )
    st.caption(
        "Higher accuracy does not guarantee higher F1 for UP. "
        "All reported comparisons concern this historical benchmark."
    )

with timeline_tab:
    st.subheader("How does performance change across the ordered stream?")
    st.write(
        "Each block is evaluated before its labels are used for updates. "
        "The horizontal axis is the **block index**, not a verified calendar date."
    )
    windows = window_metrics(ROOT)
    if windows is None:
        st.info(
            "Detailed block data are not included in the published summary. "
            "Run `python scripts/run_all.py` on your computer and reopen the dashboard."
        )
    else:
        chosen = st.selectbox("Window metric", list(METRICS), key="window_metric")
        strategy_options = [name.lower() for name in ORDER]
        selection = st.multiselect(
            "Strategies", strategy_options, default=strategy_options, key="strategies"
        )
        if selection:
            visible = windows.loc[windows["strategy"].isin(selection)]
            series = visible.pivot(index="block", columns="strategy", values=METRICS[chosen])
            smoother = st.slider("Moving-average blocks", min_value=1, max_value=12, value=5)
            series = series.rolling(smoother, min_periods=1).mean() * 100
            st.line_chart(series, y_label=f"{chosen} (%)", x_label="Evaluation block")
            st.caption("Moving averages are descriptive; their values are not independent statistical tests.")
        else:
            st.warning("Select at least one strategy to view the chart.")

with shift_tab:
    st.subheader("Feature distribution shift")
    st.write(
        "Population Stability Index (PSI) compares the current batch with "
        "the initial training distribution of each input feature. "
        "A PSI increase is **not** proof of concept drift."
    )
    shifts = feature_shift(ROOT)
    if shifts is None:
        st.info(
            "For the full PSI chart, generate local results with "
            "`python scripts/run_all.py`. The published report discusses "
            "source date anomalies and the study's limitations."
        )
    else:
        available = list(shifts["feature"].unique())
        selected = st.multiselect("Input features", available, default=available)
        if selected:
            visible = shifts.loc[shifts["feature"].isin(selected)]
            series = visible.pivot(index="block", columns="feature", values="psi")
            st.line_chart(series, y_label="PSI", x_label="Evaluation block")
        else:
            st.warning("Select at least one input feature.")
    st.markdown(
        "[Read the data-integrity audit](https://github.com/riri242/Epoch/blob/main/docs/data_quality.md)"
    )

with checks_tab:
    st.subheader("Check A — Excluding the unreliable day field")
    st.write(
        "The two versions use the same data order, model family and evaluation "
        "settings. The primary experiment **does not use** the day field."
    )
    st.dataframe(
        day.style.format({"With day": "{:.2f}%", "Without day": "{:.2f}%",
                          "Difference (pp)": "{:+.3f}"}),
        use_container_width=True, hide_index=True,
    )
    st.subheader("Check B — Feedback batch size")
    batch_metric = st.selectbox(
        "Batch metric", ["Accuracy", "Balanced Accuracy", "F1 (UP)"], key="batch_metric"
    )
    if batch_metric not in batches.columns:
        st.info("The published summary contains only Accuracy. Full metrics are available in local generated results.")
        batch_metric = "Accuracy"
    table = batches.pivot(index="Batch size", columns="Strategy", values=batch_metric)
    st.line_chart(table, y_label=f"{batch_metric} (%)", x_label="Batch size")
    st.dataframe(table.style.format("{:.2f}%"), use_container_width=True)
    st.caption(
        "The batch-size check changes when labels become available, so it "
        "should not be interpreted as a purely isolated batch-size effect."
    )

with report_tab:
    st.subheader("Scientific report")
    st.caption("The report includes methodology, findings, interpretation and limitations.")
    st.download_button(
        "Export report (Markdown)", report.encode("utf-8"),
        file_name="epoch_research_report.md", mime="text/markdown",
    )
    st.markdown(report)
