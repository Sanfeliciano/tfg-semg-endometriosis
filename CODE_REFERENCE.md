# Code Reference — TFG EMG Signal Analysis

This document explains every file, function, and method in the project, in the order they are called during a typical pipeline run.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [main.py — Pipeline entry point](#2-mainpy--pipeline-entry-point)
3. [src/data/loaders.py — MATLAB data loading](#3-srcdataloaderspy--matlab-data-loading)
4. [src/signal/features.py — Signal feature extraction](#4-srcsignalfeaturespy--signal-feature-extraction)
5. [src/signal/utils.py — Shared utilities](#5-srcsignalutilspy--shared-utilities)
6. [src/signal/process_all.py — Full processing loop](#6-srcsignalprocess_allpy--full-processing-loop)
7. [src/stats/statistics_.py — Statistical tests](#7-srcstatsstatistics_py--statistical-tests)
8. [src/plots/visualization.py — Plots and arrow tables](#8-srcplotsvisualizationpy--plots-and-arrow-tables)
9. [src/plots/evolution_plot.py — Evolution figure](#9-srcplotsevolution_plotpy--evolution-figure)

---

## 1. Project overview

The pipeline studies EMG (electromyography) signals from patients with **endometriosis** and a **healthy control group**, recorded at four time points (week 1, 8, 12, 24) during pelvic floor contractions and relaxation phases.

The main steps are:

```
MATLAB .mat files
    └── src/data/loaders.py          reads signals and segmentation limits
    └── src/signal/features.py       extracts RMS, FM, DI, SampEn, MSCOH, ICOH per segment
    └── src/signal/process_all.py    loops over all patients / weeks / phases, collapses contractions
    └── → saved to Excel (results/)
src/stats/statistics_.py             runs Mann-Whitney / T-test / Wilcoxon / GLM
src/plots/visualization.py          produces boxplots, heatmaps, forest plots, arrow tables
src/plots/evolution_plot.py         standalone evolution figure (run separately)
```

**Channels:** 8 surface EMG channels — M1, M2, M3, M4 (muscle), B1, B2 (superficial), S1, S2 (deep).  
**Channel pairs:** B1-B2, S1-S2, M1-M3, M2-M4, M3-M4 — used for coherence features.  
**Phases per recording:** PRE (rest/relaxation) + C1 to C5 (five contractions).

---

## 2. `main.py` — Pipeline entry point

Top-level script that orchestrates the full analysis. Run with `python main.py` from the project root. Contains no classes — only configuration, sequential steps, and one helper function.

### `PATHS` dict (module level)

Holds every input and output path in one place so they are easy to change. Directories for results are created automatically with `os.makedirs`.

| Key | What it points to |
|---|---|
| `segments_ctrl` / `segments_endo` | `.mat` files with contraction/relaxation segment limits |
| `signals_ctrl` / `signals_endo` | `.mat` files with raw EMG waveforms |
| `clinical_endo` / `clinical_ctrl` | `.mat` files with age, BMI, deliveries, etc. |
| `xlsx_ctrl_final` / `xlsx_endo_final` | Cached Excel outputs — if they exist the MATLAB step is skipped |
| `out_plots` | `results/plots` — all figures |
| `out_stats` | `results/tables` — all Excel tables |

### `clean_patient_id(val) -> str`

Converts heterogeneous patient ID strings (e.g. `"NHC_001"`, `"Endo01"`) to a plain integer string `"1"`. This is needed because the MATLAB signal files and the clinical data files use different naming conventions, and the GLM merge would silently drop rows if the IDs did not match.

**Steps 1–5** in `main.py` correspond to the five pipeline stages printed to the console:

| Step | What it does |
|---|---|
| `[1/5]` | Loads (or reloads from Excel cache) the signal features for both groups |
| `[2/5]` | Runs independent and paired statistical tests |
| `[3/5]` | Loads clinical data, runs the adjusted GLM, exports SPSS dataset as CSV |
| `[4/5]` | Generates boxplots for every feature × phase combination |
| `[5/5]` | Generates heatmaps, effect-size chart, and all arrow summary tables |

### `_print_significant_details_console(df_indep, df_pair, df_glm, alpha, max_rows)`

Prints a formatted console summary of all statistically significant results after the pipeline finishes. Useful for a quick sanity check without opening any Excel file. It handles four sub-tables: Independent, Paired, GLM (Endometriosis term), GLM (Age term), and GLM (Births term). Rows with `p < alpha` (default 0.05) are selected; if a table has more than `max_rows` rows only the first `max_rows` are printed.

---

## 3. `src/data/loaders.py` — MATLAB data loading

Reads the `.mat` data files produced by MATLAB and converts them into Python dictionaries and DataFrames.

### `load_matlab_signals(path_signals) -> dict`

Loads raw EMG waveforms from a `.mat` struct.

**Input:** path to a `.mat` file whose root variable is `data_struct`.

**Output:** nested dict `signals[patient_key][week_key][channel]` → 1-D NumPy array.

- Patient keys start with `"NHC_"` — other struct fields are ignored.
- Week keys are whatever MATLAB named them (e.g. `"week_1"`, `"week_8"`, …).
- The eight channels loaded per week are `M1, M2, M3, M4, B1, B2, S1, S2`.

### `load_matlab_segments(path_segments, fs=1000) -> dict`

Loads segment limits (start/end times) and channel quality flags from a `.mat` struct.

**Input:** path to a `.mat` file whose root variable is `datas_segmentation`.

**Output:** nested dict `segments[patient_key][week_key][phase]` → `{"samples": np.array, "aptitude": np.array}`.

- Only the phases `PRE, C1, C2, C3, C4, C5` are kept.
- `limits_event` values in the MATLAB file are in **seconds**; they are multiplied by `fs` (1000 Hz) and cast to int to get sample indices.
- `aptitude_channels` is a boolean-like array with one flag per channel. A `0` means that channel was deemed unreliable for that recording and its features will be set to `NaN` in `process_all.py`.

### `load_one_clinical_file(file_path, group_label) -> pd.DataFrame`

Loads clinical parameters for one group (Endometriosis or Control) from a `.mat` file.

**Output:** one row per patient with columns: `patient, group, age, vaginal_deliveries, bmi, years_pain, low_back_pain, musculoskeletal_alter, smoking`.

- Tries `datas_parametrization` as the root key; falls back to the first non-dunder key found.
- Missing fields are read as `np.nan` via `getattr(..., default)`.
- Returns an empty DataFrame on any loading error so the rest of the pipeline can continue.

---

## 4. `src/signal/features.py` — Signal feature extraction

Computes the six signal features used in the analysis. Each function works on one segment (a contiguous slice of a waveform).

### `single_signal_parameters_segments(signal, fs, feature, segments) -> float`

Computes one of four single-channel features from the first valid segment found.

**Arguments:**
- `signal` — full waveform (1-D NumPy array).
- `fs` — sampling frequency (1000 Hz throughout the project).
- `feature` — one of `"RMS"`, `"FM"`, `"DI"`, `"SAMPEN"`.
- `segments` — 2-D array of `[start_sample, end_sample]` pairs; only the first row is used.

Returns `np.nan` if no segment exists.

The PSD (power spectral density) is estimated using Welch's method with a **1-second window** and **50 % overlap** for all frequency-domain features.

| Feature | Formula / method | What it measures |
|---|---|---|
| **RMS** | √(mean(x²)) | Muscle activation intensity |
| **FM** (median frequency) | Interpolate the cumulative PSD to find the frequency where 50 % of spectral energy lies | Spectral centroid; drops with muscle fatigue |
| **DI** (fatigue index) | Σ(PSD / f) / Σ(PSD × f⁵) | Energy at low frequencies relative to high; increases with fatigue |
| **SampEn** (sample entropy) | `EntropyHub.SampEn` with m=2, r=0.15 on a z-scored segment, returns the m=2 value | Signal complexity / regularity |

### `two_signal_parameters_segments(signal_1, signal_2, fs, feature, segments) -> float`

Computes one of two coherence features between a **pair** of channels.

**Arguments:** same as above but with two signals and `feature` is `"MSCOH"` or `"ICOH"`.

Requires the segment to be **at least 4.1 seconds** long (otherwise skipped). The envelopes of both signals are extracted first via `envelope_lowpass` (see `utils.py`). The Welch window is **4 seconds** with **3.9 s overlap** — this gives good frequency resolution in the 0–10 Hz band where pelvic floor EMG synchrony is expected.

| Feature | Formula / method | What it measures |
|---|---|---|
| **MSCOH** (magnitude-squared coherence) | `scipy.signal.coherence`, averaged over 0–10 Hz | Linear synchrony between two channels |
| **ICOH** (imaginary coherence) | Im(Pxy) / √(Pxx × Pyy), absolute value averaged over 0–10 Hz | Non-zero-lag synchrony; robust to volume conduction artefacts |

---

## 5. `src/signal/utils.py` — Shared utilities

Small helper functions used by other modules.

### `envelope_lowpass(signal, fs) -> np.array`

Extracts the amplitude envelope of an EMG signal.

1. Full-wave rectification: `|signal|`
2. 4th-order zero-phase Butterworth low-pass filter at **10 Hz** cutoff (`scipy.signal.filtfilt`)

Used by `two_signal_parameters_segments` before computing coherence, because coherence between raw EMG signals would be dominated by high-frequency motor unit firing and not reflect the muscle-level coordination of interest.

### `get_data_for_statistic_analysis(df_combined, feature, channel, phase, week, group) -> pd.Series`

Filters `df_combined` to a specific combination of `(feature, channel/channel_pair, phase, week, group)` and returns the `value` column with NaNs dropped. Determines automatically whether to filter on `channel` or `channel_pair` based on whether the feature is `MSCOH` or `ICOH`.

*Note: this function is defined here but not currently called from `main.py`; it may be useful for exploratory analysis in a notebook.*

---

## 6. `src/signal/process_all.py` — Full processing loop

### `process_all(signals, segments) -> pd.DataFrame`

The main computation function. Iterates over every combination of patient × week × phase × channel (or channel pair) × feature and computes the corresponding value.

**Inputs:** the dicts returned by `load_matlab_signals` and `load_matlab_segments`.

**Output:** one long-format DataFrame with columns `patient, week, phase, channel (or channel_pair), feature, value`.

#### Channel aptitude check

Before calling any feature function, it checks the `aptitude` flag for that channel. If the flag is `0` for that recording the value is set directly to `NaN` — no computation is attempted.

The index mapping for the aptitude array is:

| Channel | Index |
|---|---|
| M1 | 0 |
| M2 | 1 |
| M3 | 2 |
| M4 | 3 |
| B1 | 4 |
| B2 | 5 |
| S1 | 8 |
| S2 | 9 |

#### Progress tracking

Counts the total number of (patient × week × phase × channel × feature) operations before starting and prints a running percentage and estimated time remaining on each iteration.

#### Phase collapsing

After computing all values, the raw DataFrame still has phases `PRE, C1, C2, C3, C4, C5`. The function collapses them to two final phases:

- **`pre`** — kept as-is (one row per recording).
- **`contraction`** — median of C1 through C5 per `(patient, week, channel/channel_pair, feature)`. This reduces variability across the five contractions and gives one representative value per session.

The final DataFrame is sorted so `pre` rows come before `contraction` rows within each patient/week/channel group.

---

## 7. `src/stats/statistics_.py` — Statistical tests

### `compute_full_statistics(df_endo, df_control) -> (pd.DataFrame, pd.DataFrame)`

Runs **independent** and **paired** hypothesis tests for every combination of `(feature, phase, channel)`.

Returns two DataFrames: `(df_indep, df_pair)`.

#### Independent tests (Control pooled vs Endometriosis per week)

For each `(feature, phase, channel, week)` combination:
1. Normality is tested with `pingouin.normality` on both groups.
2. If both are normal → **Levene's test** for equal variances.
   - Equal variances: **Student's T-test** (`pingouin.ttest`, `correction=False`)
   - Unequal variances: **Welch's T-test** (`pingouin.ttest`, `correction=True`)
3. If either group is non-normal → **Mann-Whitney U** (`pingouin.mwu`)

Output columns: `Feature, Phase, Channel, Comparison, p_val, Test_Used, Effect_Size (RBC)`.

MSCOH and ICOH values at the ceiling (≥ 0.999) are excluded before testing — these are saturated coherence values that would distort the distributions.

#### Paired tests (Endometriosis week_1 vs week_8/12/24)

For each `(feature, phase, channel, follow-up week)` combination:
1. Patients are matched by ID using `pivot` — only patients with data at both weeks are included.
2. Normality is tested on the **difference** (x − y).
3. Normal difference → **Paired T-test** (`pingouin.ttest`, `paired=True`)
4. Non-normal difference → **Wilcoxon signed-rank** (`pingouin.wilcoxon`)

Requires at least 3 matched pairs to run.

---

### `fit_adjusted_glm(df_endo, df_control, df_clinical, debug=True, min_n_per_group=3) -> pd.DataFrame`

Fits a **General Linear Model** (Gaussian family, identity link) for every `(feature, phase, channel, week)` combination.

**Formula:** `value ~ group_binary + age + C(vaginal_births)`

- `group_binary` — 1 = Endometriosis, 0 = Control (the main comparison of interest).
- `age` — continuous covariate.
- `C(vaginal_births)` — binary factor (0 = never delivered vaginally, 1 = at least one).

Control subjects are pooled across all weeks and compared against endometriosis at each individual follow-up week, so `week` is a grouping variable rather than a model term.

**Output columns per row:** Feature, Phase, Channel_or_Pair, Week, and for each of the three terms (Endo, Age, Births): coefficient, SE, 95% CI lower, 95% CI upper, p-value. Also N, N_Control, N_Endo.

Rows are skipped if:
- The control subset for that `(feature, phase, channel)` is empty.
- After dropping NaNs in the required columns fewer than `min_n_per_group` observations remain in either group.
- Both groups are not present in the data (no contrast to estimate).
- The model fitting raises any exception (logged to the `glm_fail` counter).

The `debug` flag enables verbose counters at the end showing how many combinations were skipped and why.

#### Inner helper `dprint(*args)`

Wrapper around `print` that only outputs when `debug=True`. Used for all diagnostic messages inside the GLM function.

---

## 8. `src/plots/visualization.py` — Plots and arrow tables

Contains all plotting and table-export functions. No classes.

---

### Private helpers (prefix `_`)

These are internal utilities not called from `main.py` directly.

#### `_ensure_dir(path)`
Creates a directory (and any parent directories) if it does not exist. Wraps `os.makedirs(..., exist_ok=True)`.

#### `_norm_str(x) -> str`
Strips and lowercases a value to a string. Used for robust phase-label matching.

#### `_p_to_count(p) -> int`
Maps a p-value to an arrow count:
- `p < 0.001` → 3
- `p < 0.01` → 2
- `p < 0.05` → 1
- `p ≥ 0.05` or NaN → 0

#### `_coef_to_dir(coef) -> str`
Returns `"↑"` if the coefficient is positive, `"↓"` if negative, `""` if NaN.

#### `_cell_arrows(p, coef=None) -> str`
Combines `_p_to_count` and `_coef_to_dir` into a single cell string. If no direction is available (coef is None/NaN) uses bullet characters `"•"` instead of arrows. For example: `p=0.003, coef<0` → `"↓↓"`.

#### `_style_arrow_cell(val) -> str`
Returns a CSS style string for Excel cell styling based on how many arrows/bullets are in the cell:
- 3 → dark grey background (`#7A7A7A`), white text
- 2 → medium grey (`#B0B0B0`)
- 1 → light grey (`#D9D9D9`)
- empty → white

#### `_phase_to_visual(ph) -> str`
Converts any phase label variant (`"pre"`, `"rest"`, `"reposo"`, etc.) to `"Relaxation"` or `"Contraction"` for display in tables.

#### `_col_channel_for_feature(feature) -> str`
Returns `"channel_pair"` for coherence features (MSCOH, ICOH), `"channel"` for all others.

#### `_normalize_phase2(x) -> str`
Reduces any phase label to exactly `"pre"` or `"contraction"` for two-level grouping in summary tables. Looks for `"pre"` or `"relax"` as substrings; everything else maps to `"contraction"`.

#### `_build_glm_arrow_table(df_glm, p_col, coef_col, output_path)`
Internal function that builds and saves one GLM arrow table to Excel. Called three times by `generate_glm_arrow_tables` — once for each covariate term. Uses `pivot_table` to reshape the long-format GLM results into a Feature × (Phase | Week) matrix, then applies `_style_arrow_cell` via `Styler.applymap`.

#### `_prepare_data(df_eff, coef_col, p_col)` (nested inside `export_forestplots_covariates_summary`)
Renames feature labels for publication (`FM` → `MDF`, `SAMPEN` → `SampEn`, `ICOH` → `iCOH`), builds a display label string `Feature | Channel | Week`, sorts by p-value or absolute coefficient (controlled by `sort_by`), and returns the top `top_n` rows.

#### `_draw_forest_plot(df_plot, coef_col, lo_col, hi_col, p_col, title, output_path)` (nested inside `export_forestplots_covariates_summary`)
Draws a horizontal forest plot: one row per significant result, showing the estimated coefficient as a dot and the 95 % CI as a horizontal line. Annotates each row with `β=…, p=…`. Saves to PNG at 300 dpi.

---

### Public functions

#### `combine_dataframes_from_excel(endo_path, control_path) -> pd.DataFrame`
Reads two Excel files (endometriosis and control results), adds a `"group"` column to each, and concatenates them into one long DataFrame. *Not called from `main.py`* — used when starting from cached Excel files in exploratory scripts.

---

#### `generate_separated_boxplots(df_combined, feature_name, momento, output_dir)`

Generates a grid of boxplots — one subplot per channel or channel pair — comparing the two groups.

- `momento="PRE"` → filters phase `"pre"` (rest).
- Any other value for `momento` → filters phase `"contraction"`.
- White boxes with black outlines; individual data points overlaid as a strip plot (black = Endometriosis, grey = Control).
- Saved to `output_dir/BOXPLOT_{MOMENTO}_{FEATURE}.png` at 300 dpi.
- Called from `main.py` for every feature × {PRE, GLOBAL} combination.

---

#### `plot_pvalue_distribution(df_indep, df_pair)`

Draws two histograms of p-value distributions (independent tests on the left, paired on the right) with a dashed red line at p=0.05. Useful to visually assess whether the tests are well-calibrated. *Not called from `main.py`* — used for exploratory analysis.

---

#### `plot_pvalue_heatmap(df_results, titulo, output_dir, filename)`

Generates a heatmap where:
- Rows = `Feature (Channel)` labels.
- Columns = comparison names (e.g. `"Control vs week_8"`).
- Non-significant cells (p ≥ 0.05) are shown in light blue.
- Significant cells (p < 0.05) are colour-coded `yellow → orange → red` and annotated with the exact p-value.

Called twice from `main.py`: once for independent results, once for paired results.

---

#### `plot_effect_sizes(df_results, titulo, output_dir, filename)`

Horizontal bar chart of effect sizes for significant results only. Bars pointing right are teal (positive effect), left are salmon (negative). Reference lines at |0.1|, |0.3|, |0.5| mark small/medium/large thresholds. Limited to the top 30 results by absolute effect size if there are more than 30. Called once from `main.py` for the independent results.

---

#### `generate_summary_arrow_table(df_stats, df_endo, df_ctrl, output_dir)`

Arrow table for **independent** comparisons (Control vs Endometriosis).

- Rows = significant `(Feature, Channel)` pairs only.
- Columns = `Contraction | week_X` and `Relaxation | week_X` for X ∈ {1, 8, 12, 24}.
- Cell content: arrows pointing up (↑) or down (↓) relative to the pooled control median, repeated 1–3 times by significance level. Empty if not significant.
- Direction check uses actual medians from the data, not the test statistic, to avoid sign confusion.
- Saved as `SUMMARY_ARROW_TABLE.xlsx` with grey-shaded cells.

---

#### `generate_paired_arrow_table(df_stats_pair, df_endo, output_dir)`

Arrow table for **paired** comparisons (Endometriosis week_1 vs later weeks).

- Same structure as above but columns only cover weeks 8, 12, 24 (week 1 is the baseline).
- Direction = median of the follow-up week vs median of week 1 (within Endometriosis patients).
- Saved as `PAIRED_ARROW_TABLE.xlsx`.

---

#### `generate_glm_arrow_tables(df_glm, output_dir)`

Generates three separate GLM arrow tables by calling `_build_glm_arrow_table` three times, one per covariate term:

| File | Covariate |
|---|---|
| `GLM_ARROW_ENDO.xlsx` | Group effect (Endometriosis vs Control) |
| `GLM_ARROW_AGE.xlsx` | Age |
| `GLM_ARROW_BIRTHS.xlsx` | Vaginal deliveries (binary) |

---

#### `export_forestplots_covariates_summary(df_glm, effects, alpha, phase_pre, phase_contra, sort_by, top_n, output_dir)`

Generates forest plots for the clinical covariates (Age and Vaginal Deliveries) restricted to B and S channels/pairs.

**Parameters:**
- `effects` — which covariates to plot; default `("Age", "Births")`.
- `alpha` — significance threshold for inclusion; default 0.05.
- `sort_by` — `"p"` (ascending p-value) or `"abscoef"` (descending |coefficient|).
- `top_n` — maximum rows per plot; default 50.
- `output_dir` — where to save figures.

Produces up to four PNG files (contraction + relaxation × age + deliveries):
```
FOREST_Age_Contraction_BS.png
FOREST_Age_Relaxation_BS.png
FOREST_VaginalDelivery_Contraction_BS.png
FOREST_VaginalDelivery_Relaxation_BS.png
```

Only significant results are plotted. The function auto-detects which string the contraction phase was labelled with in `df_glm`.

---

## 9. `src/plots/evolution_plot.py` — Evolution figure

Standalone script (run with `python src/plots/evolution_plot.py`). Reads the SPSS-format dataset exported by `main.py` and produces a 4 × 2 grid figure showing the **longitudinal evolution** of the four single-channel features across weeks, split by phase (Contraction left, Relaxation right).

### Configuration constants (module level)

| Constant | Description |
|---|---|
| `CHANNEL` | Which channel to plot; `"B2"` by default (switch to `"S2"` for deep layer) |
| `OUTPUT_NAME` | Output path: `results/plots/FIGURE_FINAL.png` |
| `DATASET_FILE` | Input: `results/tables/spss_dataset.csv` |
| `FEATURES` | `["RMS", "FM", "DI", "SAMPEN"]` |
| `WEEKS` / `WEEK_LABELS` | Internal identifiers vs. x-axis tick values (0, 8, 12, 24) |
| `LABELS` | Display axis labels per feature (e.g. `"MDF (Hz)"`) |

### `get_phase_subset(dataframe, feature, mode) -> pd.DataFrame`

Filters the dataset to one feature and one mode:
- `mode="contraction"` → all phases except `"pre"`.
- `mode="relaxation"` → only `"pre"`.

Also filters to the configured `CHANNEL`.

### `summarize(vals) -> (median, Q1, Q3)`

Returns the median, 25th, and 75th percentile of a Series after dropping NaNs. Returns `(pd.NA, pd.NA, pd.NA)` if the input is empty.

### `get_stats(dataframe, feature, mode) -> pd.DataFrame`

Calls `get_phase_subset` and then computes, for each of the four weeks:
- The per-week median + IQR of the **Endometriosis** group.
- The pooled median + IQR of the **Control** group (same values repeated for all weeks, as controls are a baseline reference rather than a longitudinal cohort).

Returns a DataFrame with one row per week and columns `med_endo, q1_endo, q3_endo, med_ctrl, q1_ctrl, q3_ctrl`.

### `draw(ax, stats, ylabel)`

Plots median and IQR lines for both groups onto a Matplotlib axis:
- **Black** lines: Endometriosis (solid = median, dashed = Q1/Q3).
- **Grey** lines: Healthy control (same style).
- Adds a small vertical margin (8 % of range) above and below the data range.

### `if __name__ == "__main__":` block

Entry point. Reads the dataset, normalises string columns (strip + lower/upper), applies unit scaling (RMS × 100 to express in cV, DI × 10⁶ for readability), calls `get_stats` + `draw` for each feature × phase, adds a legend, adjusts subplot spacing, saves to `results/plots/FIGURE_FINAL.png` at 300 dpi, and calls `plt.show()`.
