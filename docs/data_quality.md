# Electricity data integrity

Epoch evaluates the original OpenML Electricity (ELEC2, data ID 151) row sequence. Its normalized `date` field is **not** a dependable timestamp. In the 45,312-row download examined for this project, backward jumps occurred at row indices 25488, 34896, 35232, 36240 and 40704 (zero-based).

A published inspection of ELEC2 also reports a repeated NSW demand day around rows 17376–17423, and warns that the `day` field may be offset following that duplication. This report **does not prove** that the other measurements or target labels are duplicated, so the project must not silently remove rows based on this observation alone.

The default benchmark therefore:
- Preserves the original source row order and does not sort on `date`.
- Excludes `date` from model predictors.
- Treats all findings as *ordered-stream* results, not independently verified real-time forecasts.
- Keeps the downloaded data unmodified for reproducibility.

**Research follow-up:** audit entire candidate repeated days across all measurement columns and labels; run a sensitivity experiment without `day` and report any impact on accuracy. The five backward jumps are recorded as source data-quality caveats, not interpreted as genuine distribution drift.

References:
- [OpenML Electricity dataset](https://www.openml.org/d/151)
- [Ludvig Hult, *Real World Data Sets are Messy: Case Study Elec2* (2023)](https://el-hult.github.io/2023/12/21/elec2.html)
