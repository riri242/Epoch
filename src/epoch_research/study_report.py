"""Create a concise research report from measured experiment outputs."""

from pathlib import Path


def write_research_report(primary, day_check, batch_check, batch_rows, audit, output):
    """Use experiment objects, never manually entered performance figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    metrics = primary["strategies"]
    score_rows = [
        "| Strategy | Accuracy | Balanced accuracy | F1 (UP) | Model updates |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("static", "periodic", "online", "adwin"):
        item = metrics[name]
        score_rows.append(
            f"| {name.title()} | {item['accuracy']:.3f} | "
            f"{item['balanced_accuracy']:.3f} | {item['f1_up']:.3f} | "
            f"{item['model_updates']} |"
        )

    ablation = [
        "| Strategy | Accuracy with day | Accuracy without day | Difference (pp) |",
        "|---|---:|---:|---:|",
    ]
    for name, item in day_check["comparison"].items():
        ablation.append(
            f"| {name.title()} | {item['with_day']['accuracy']:.3f} | "
            f"{item['without_day']['accuracy']:.3f} | "
            f"{item['delta_percentage_points']['accuracy']:+.3f} |"
        )

    robustness = [
        "| Batch size | Static | Periodic | Online | ADWIN | Best accuracy |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for size in batch_check["batch_sizes"]:
        subset = batch_rows.loc[batch_rows["batch_size"].eq(size)]
        values = dict(zip(subset["strategy"], subset["accuracy"]))
        winner = max(values, key=values.get)
        robustness.append(
            f"| {size} | {values['static']:.3f} | "
            f"{values['periodic']:.3f} | {values['online']:.3f} | "
            f"{values['adwin']:.3f} | {winner.title()} |"
        )
    best = max(metrics, key=lambda name: metrics[name]["accuracy"])
    difference = 100 * (metrics[best]["accuracy"] - metrics["static"]["accuracy"])
    best_f1 = max(metrics, key=lambda name: metrics[name]["f1_up"])
    robust_winners = []
    for size in batch_check["batch_sizes"]:
        subset = batch_rows.loc[batch_rows["batch_size"].eq(size)]
        accuracies = dict(zip(subset["strategy"], subset["accuracy"]))
        robust_winners.append(max(accuracies, key=accuracies.get))
    if len(set(robust_winners)) == 1:
        batch_interpretation = f"{robust_winners[0].title()} was highest in accuracy at every pre-specified batch size."
    else:
        batch_interpretation = "The highest-accuracy strategy changed across the tested batch sizes."

    lines = [
        "# Epoch: A Study of Temporal Data Shift and Model Adaptation",
        "",
        "## Abstract",
        "",
        "This exploratory study compares four classifier-update policies on the ordered",
        "Electricity benchmark. A fixed initial training period is followed by a",
        "test-before-train evaluation of future observations. The primary analysis",
        "excludes the unreliable calendar fields `day` and `date`. Additional",
        "analyses examine sensitivity to reintroducing `day` and to changing the",
        "feedback batch size. Results describe this particular benchmark and do",
        "not establish a new adaptation algorithm or universal superiority.",
        "",
        "## 1. Introduction and research question",
        "",
        "**Question:** Do periodic retraining, incremental online updates, or",
        "error-triggered retraining outperform a classifier that is never updated",
        "on a changing, ordered data stream?",
        "",
        "## 2. Data and methods",
        "",
        f"- Source: [OpenML Electricity, dataset 151](https://www.openml.org/d/151); {primary['dataset_rows']:,} observations.",
        f"- Initial training (first 30%): {primary['initial_training_rows']:,} records.",
        f"- Held-forward test-before-train evaluation: {primary['evaluated_rows']:,} records.",
        "- Outcome: electricity price movement class `UP` or `DOWN`.",
        f"- Primary predictors: {', '.join(primary['features'])}.",
        "- The contemporaneous price fields and suspect calendar fields are not predictors.",
        "- Model: standardized SGD logistic classifier, one copy per policy.",
        "- Preprocessing: scaler fitted on warm-up data only; no future-data scaling.",
        "- Update policies: Static (no update); Periodic (recent 6,000 rows every eight",
        "  256-row blocks); Online (partial fit after each block); ADWIN",
        "  (error-stream alerts trigger recent-window retraining with cooldown).",
        "- All methods evaluate an entire incoming batch **before** seeing its labels;",
        "  the protocol assumes immediate label availability after that batch.",
        "- Metrics: accuracy, balanced accuracy, and F1 for the `UP` class.",
        "- Feature shift: population stability index (PSI), which is descriptive",
        "  and does not on its own identify concept drift.",
        "- The original source row order is preserved; the `date` column is not",
        "  treated as a verified clock or used to reorder records.",
        "",
        "## 3. Results",
        "",
        "### 3.1 Primary experiment (without day)",
        "",
        *score_rows,
        "",
        f"The highest accuracy was obtained by **{best.title()}**: "
        f"{metrics[best]['accuracy']:.3f}, "
        f"{difference:+.2f} percentage points relative to Static. "
        f"The highest F1 (UP) came from **{best_f1.title()}** "
        f"({metrics[best_f1]['f1_up']:.3f}); thus one method does not lead on every metric.",
        "",
        "### 3.2 Sensitivity to the unreliable day predictor",
        "",
        "Only the inclusion of `day` changes; the evaluation rows, ordering,",
        "seed, model family, update settings and training split stay fixed.",
        "Differences are without-day minus with-day, expressed in percentage points.",
        "",
        *ablation,
        "",
        "### 3.3 Batch-size sensitivity",
        "",
        f"The study tests {', '.join(map(str, batch_check['batch_sizes']))} observations per batch. Periodic",
        "retraining and ADWIN cooldown intervals are kept approximately fixed",
        "in numbers of observations.",
        "",
        *robustness,
        "",
        batch_interpretation,
        "Batch size also alters when labels become available and when adaptation",
        "decisions occur. It is therefore **not** an isolated batch-size effect.",
        "",
        "### 3.4 Data-quality audit",
        "",
        f"- Backward jumps in normalized `date`: {audit['date_decreases']}",
        f"  (at zero-based rows {', '.join(map(str, audit['date_decrease_rows']))}).",
        "- A suspected repeated 48-row segment has matching NSW demand values,",
        f"  but only {audit.get('matching_entire_rows', 'unknown')} fully identical rows.",
        "- No observations were dropped, reconstructed, or reordered.",
        "",
        "## 4. Discussion",
        "",
        "The results show that adaptation can be helpful but is not uniformly",
        "advantageous: accuracy gains, class-specific F1 and update costs may",
        "favor different policies. The day-field check examines dependence on",
        "a problematic source feature but does not repair the benchmark's calendar.",
        "PSI trends alone are not evidence that measured performance changes were",
        "caused by concept drift.",
        "",
        "## 5. Limitations",
        "",
        "- One historical dataset and one model architecture.",
        "- Fixed hyperparameters and one seed; no confidence interval or formal",
        "  significance test was estimated.",
        "- Source-calendar anomalies mean ordered-stream position is not",
        "  independently verified wall-clock time.",
        "- Labels are available after each batch; deployment latency may differ.",
        "- Policies perform different numbers of updates and use different",
        "  amounts of recent data; this is a policy comparison, not a controlled",
        "  causal study of a single adaptation mechanism.",
        "- Batch-size tests also alter feedback and update timing.",
        "",
        "## 6. Conclusion",
        "",
        "Epoch provides a reproducible, controlled benchmark of four adaptation",
        "policies with a documented data-quality audit and two small robustness",
        "checks. Findings should be interpreted as exploratory evidence specific",
        "to the chosen Electricity stream. A second reliably timestamped dataset",
        "would be the next substantive step toward stronger general claims.",
        "",
        "## Reproducibility and references",
        "",
        "- Run `python scripts/run_all.py` at the project root; generated",
        "  metrics, visualizations, audits and supplementary analyses appear in `results/`.",
        "- [Electricity dataset (OpenML ID 151)](https://www.openml.org/d/151)",
        "- [Electricity data-quality analysis](https://el-hult.github.io/2023/12/21/elec2.html)",
        "- [River ADWIN documentation](https://riverml.xyz/latest/api/drift/ADWIN/)",
        "",
    ]
    path = output / "research_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
