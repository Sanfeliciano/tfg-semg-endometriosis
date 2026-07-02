import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==============================================================================
# Utils
# ==============================================================================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _norm_str(x):
    return str(x).strip().lower()


def _p_to_count(p):
    """Map p-value to arrow count: 0/1/2/3 based on significance thresholds."""
    if pd.isna(p):
        return 0
    if p < 0.001:
        return 3
    if p < 0.01:
        return 2
    if p < 0.05:
        return 1
    return 0


def _coef_to_dir(coef):
    """Return ↑ if coef > 0, ↓ if coef < 0, '' if NaN."""
    if pd.isna(coef):
        return ""
    return "↑" if coef > 0 else "↓"


def _cell_arrows(p, coef=None):
    """
    Return arrows scaled by significance.
    If coef is None or NaN, use bullets (no direction).
    """
    n = _p_to_count(p)
    if n == 0:
        return ""
    if coef is None or pd.isna(coef):
        return "•" * n
    return _coef_to_dir(coef) * n


def _style_arrow_cell(val):
    """
    Background intensity based on number of arrows/bullets:
      3 -> dark
      2 -> medium
      1 -> light
    """
    base = "text-align: center; border: 1px solid black;"
    if not isinstance(val, str) or val == "":
        return f"background-color: white; {base}"

    n = val.count("↑") + val.count("↓") + val.count("•")
    if n >= 3:
        return f"background-color: #7A7A7A; color: white; {base}"
    if n == 2:
        return f"background-color: #B0B0B0; color: black; {base}"
    return f"background-color: #D9D9D9; color: black; {base}"


def _phase_to_visual(ph):
    ph = _norm_str(ph)
    if ph in ["pre", "rest", "reposo", "relaxation"]:
        return "Relaxation"
    if ph in ["contraction", "contraction_global", "global", "contra"]:
        return "Contraction"
    return ph


def _col_channel_for_feature(feature: str) -> str:
    return "channel_pair" if feature in ["MSCOH", "ICOH"] else "channel"


# ==============================================================================
# Combine
# ==============================================================================
def combine_dataframes_from_excel(endo_path, control_path):
    df_endo = pd.read_excel(endo_path)
    df_control = pd.read_excel(control_path)
    df_endo["group"] = "Endometriosis"
    df_control["group"] = "Control"
    return pd.concat([df_endo, df_control], ignore_index=True)


# ==============================================================================
# Boxplots
# ==============================================================================
def generate_separated_boxplots(df_combined, feature_name, momento, output_dir="results"):
    """
    momento='PRE'  -> phase='pre'
    momento!='PRE' -> phase='contraction'
    """
    if df_combined is None or df_combined.empty:
        return

    df_feat = df_combined[df_combined["feature"] == feature_name].copy()
    if df_feat.empty:
        return

    df_feat["phase"] = df_feat["phase"].astype(str).str.lower().str.strip()

    if str(momento).upper() == "PRE":
        df_plot = df_feat[df_feat["phase"] == "pre"].copy()
        titulo_momento = "REST (PRE)"
    else:
        df_plot = df_feat[df_feat["phase"].isin(["contraction", "contraction_global", "global"])].copy()
        titulo_momento = "CONTRACTION"

    if df_plot.empty:
        return

    col_name = "channel" if feature_name in ["RMS", "FM", "DI", "SAMPEN"] else "channel_pair"

    sns.set_style("ticks")
    g = sns.catplot(
        data=df_plot,
        x="group",
        y="value",
        hue="group",
        col=col_name,
        kind="box",
        sharey=False,
        col_wrap=4,
        palette=["#FFFFFF", "#FFFFFF"],
        height=4.5, aspect=0.8,
        linewidth=1.5,
        fliersize=0,
        legend=False
    )

    for ax in g.axes.flat:
        for patch in ax.patches:
            patch.set_edgecolor("black")

    g.map_dataframe(
        sns.stripplot,
        x="group",
        y="value",
        hue="group",
        palette=["black", "gray"],
        alpha=0.5,
        jitter=True,
        size=4
    )

    g.fig.suptitle(f"{feature_name} - {titulo_momento}", y=1.05, fontweight="bold", fontsize=14)
    g.set_titles("{col_name}", fontweight="bold")
    g.set_axis_labels("", f"Value {feature_name}")

    _ensure_dir(output_dir)
    file_name = f"BOXPLOT_{str(momento).upper()}_{feature_name}.png"
    g.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches="tight")
    plt.close(g.fig)


# ==============================================================================
# P-value distribution
# ==============================================================================
def plot_pvalue_distribution(df_indep, df_pair):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if df_indep is not None and not df_indep.empty:
        sns.histplot(df_indep["p_val"], bins=20, kde=True, ax=axes[0], color="skyblue")
        axes[0].axvline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
        axes[0].set_title("P-value Distribution: Healthy vs Endometriosis")
        axes[0].set_xlabel("P-value")
        axes[0].set_ylabel("Frequency (number of tests)")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No independent data", ha="center")

    if df_pair is not None and not df_pair.empty:
        sns.histplot(df_pair["p_val"], bins=20, kde=True, ax=axes[1], color="lightgreen")
        axes[1].axvline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05")
        axes[1].set_title("P-value Distribution: Patient Evolution")
        axes[1].set_xlabel("P-value")
        axes[1].set_ylabel("Frequency (number of tests)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No paired data", ha="center")

    plt.tight_layout()


# ==============================================================================
# Heatmap p-values
# ==============================================================================
def plot_pvalue_heatmap(df_results, titulo, output_dir="results/plots", filename="heatmap.png"):
    if df_results is None or df_results.empty:
        print(f"No data for heatmap: {titulo}")
        return

    _ensure_dir(output_dir)

    df_plot = df_results.copy()
    df_plot["Label_Y"] = df_plot["Feature"].astype(str) + " (" + df_plot["Channel"].astype(str) + ")"

    try:
        heatmap_data = df_plot.pivot_table(index="Label_Y", columns="Comparison", values="p_val")
    except Exception as e:
        print(f"Error building heatmap pivot for {titulo}: {e}")
        return

    plt.figure(figsize=(12, max(6, len(heatmap_data) * 0.4)))

    mask_nonsig = heatmap_data > 0.05

    sns.heatmap(
        heatmap_data, cmap="Blues", cbar=False, mask=~mask_nonsig, vmin=0, vmax=1,
        linewidths=0.5, linecolor="lightgrey"
    )

    sns.heatmap(
        heatmap_data, cmap="YlOrRd_r", mask=mask_nonsig, vmin=0, vmax=0.05,
        annot=True, fmt=".3f",
        cbar_kws={"label": "P-value (only < 0.05)"},
        linewidths=0.5, linecolor="lightgrey"
    )

    plt.title(f"Significance Heatmap: {titulo}")
    plt.ylabel("Parameter (Channel)")
    plt.xlabel("Comparison")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    ruta = os.path.join(output_dir, filename)
    plt.savefig(ruta, dpi=300)
    print(f"Heatmap saved: {ruta}")
    plt.close()


# ==============================================================================
# Effect sizes
# ==============================================================================
def plot_effect_sizes(df_results, titulo, output_dir="results/plots", filename="effect_sizes.png"):
    if df_results is None or df_results.empty:
        return

    df_sig = df_results[(df_results["p_val"] < 0.05) & (df_results["Effect_Size"].notnull())].copy()
    if df_sig.empty:
        print(f"No significant results with effect size for: {titulo}")
        return

    _ensure_dir(output_dir)

    df_sig["Label"] = df_sig["Comparison"].astype(str) + " | " + df_sig["Feature"].astype(str) + " | " + df_sig["Channel"].astype(str)
    df_sig["Abs_Effect"] = df_sig["Effect_Size"].abs()
    df_sig = df_sig.sort_values(by="Abs_Effect", ascending=False)

    if len(df_sig) > 30:
        df_sig = df_sig.head(30)
        titulo += " (Top 30)"

    plt.figure(figsize=(10, max(6, len(df_sig) * 0.4)))

    colores = ["teal" if x > 0 else "salmon" for x in df_sig["Effect_Size"]]
    sns.barplot(data=df_sig, x="Effect_Size", y="Label", palette=colores, hue="Label", legend=False)

    plt.axvline(x=0.1, color="grey", linestyle=":", label="Small (0.1)")
    plt.axvline(x=0.3, color="grey", linestyle="--", label="Medium (0.3)")
    plt.axvline(x=0.5, color="grey", linestyle="-", label="Large (0.5)")
    plt.axvline(x=-0.1, color="grey", linestyle=":")
    plt.axvline(x=-0.3, color="grey", linestyle="--")
    plt.axvline(x=-0.5, color="grey", linestyle="-")

    plt.title(f"Effect Size for Significant Results: {titulo}")
    plt.xlabel("Effect Size")
    plt.ylabel("Comparison | Feature | Channel")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    ruta = os.path.join(output_dir, filename)
    plt.savefig(ruta, dpi=300)
    print(f"Effect size plot saved: {ruta}")
    plt.close()


def _order_arrow_columns(df: pd.DataFrame, weeks: list) -> pd.DataFrame:
    """Reorder columns: fixed identifiers | Contraction weeks | Relaxation weeks."""
    cols_fixed = ["Parameter", "Channel"]
    cols_contra = [f"Contraction | {w}" for w in weeks if f"Contraction | {w}" in df.columns]
    cols_relax = [f"Relaxation | {w}" for w in weeks if f"Relaxation | {w}" in df.columns]
    other_cols = [c for c in df.columns if c not in cols_fixed + cols_contra + cols_relax]
    return df[cols_fixed + cols_contra + cols_relax + other_cols]


def _drop_empty_arrow_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where all arrow cells are empty."""
    arrow_cols = df.columns[2:]
    df = df.copy()
    df[arrow_cols] = df[arrow_cols].replace("", np.nan)
    return df.dropna(subset=arrow_cols, how="all")


def _normalize_phase2(x: str) -> str:
    """
    Normalise phase to exactly two levels for summary tables:
      - 'pre'         -> Relaxation
      - 'contraction' -> Contraction
    Accepts variants: PRE, pre, Relaxation, GLOBAL, contraction, etc.
    """
    s = str(x).lower().strip()
    if "pre" in s or "relax" in s:
        return "pre"
    return "contraction"


# ==============================================================================
# Arrow tables (independent) - ROBUST (NO FALSE POSITIVES)
# ==============================================================================
def generate_summary_arrow_table(df_stats, df_endo, df_ctrl, output_dir="results/tables"):
    """
    Arrow table: Independent (Control vs Endo per week).
    - Direction: median Endo (week) vs median Control (pooled).
    - Intensity: p<0.05 (1), p<0.01 (2), p<0.001 (3).
    - Exact comparison match to avoid false positives.
      Comparison string must be: "Control vs week_8".
    """
    if df_stats is None or df_stats.empty:
        return

    print("   -> Generating Arrow Table (Independent)...")

    weeks = ["week_1", "week_8", "week_12", "week_24"]

    stats = df_stats.copy()
    stats["p_val"] = pd.to_numeric(stats["p_val"], errors="coerce")
    stats["Feature"] = stats["Feature"].astype(str).str.strip()
    stats["Channel"] = stats["Channel"].astype(str).str.strip()
    stats["Comparison"] = stats["Comparison"].astype(str).str.strip()
    stats["Phase2"] = stats["Phase"].apply(_normalize_phase2)

    sig_rows = stats[stats["p_val"] < 0.05]
    if sig_rows.empty:
        print("      No significant results to generate the table.")
        return

    relevant_items = sig_rows[["Feature", "Channel"]].drop_duplicates().values

    df_endo2 = df_endo.copy()
    df_ctrl2 = df_ctrl.copy()
    df_endo2["phase2"] = df_endo2["phase"].apply(_normalize_phase2)
    df_ctrl2["phase2"] = df_ctrl2["phase"].apply(_normalize_phase2)

    data_rows = []

    for feat, ch in relevant_items:
        row = {"Parameter": feat, "Channel": ch}
        col_ch = _col_channel_for_feature(feat)

        for phase_code, phase_label in [("pre", "Relaxation"), ("contraction", "Contraction")]:
            df_e_val = df_endo2[(df_endo2["feature"] == feat) & (df_endo2["phase2"] == phase_code)]
            df_c_val = df_ctrl2[(df_ctrl2["feature"] == feat) & (df_ctrl2["phase2"] == phase_code)]

            val_ctrl = df_c_val[df_c_val[col_ch] == ch]["value"].median()

            for wk in weeks:
                comp = f"Control vs {wk}"
                stats_subset = stats[
                    (stats["Feature"] == feat) &
                    (stats["Channel"] == ch) &
                    (stats["Phase2"] == phase_code) &
                    (stats["Comparison"] == comp)
                ]

                cell = ""
                if not stats_subset.empty:
                    p = stats_subset["p_val"].min()
                    if pd.notna(p) and p < 0.05:
                        val_endo = df_e_val[
                            (df_e_val[col_ch] == ch) & (df_e_val["week"].astype(str).str.strip() == wk)
                        ]["value"].median()

                        if pd.notna(val_endo) and pd.notna(val_ctrl):
                            direction = "↑" if val_endo > val_ctrl else "↓"
                            cell = direction * _p_to_count(p)

                row[f"{phase_label} | {wk}"] = cell

        data_rows.append(row)

    df_final = pd.DataFrame(data_rows)
    df_final = _order_arrow_columns(df_final, weeks)
    df_final = _drop_empty_arrow_rows(df_final)

    if df_final.empty:
        print("All rows were empty (no arrows). Table not generated.")
        return

    _ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "SUMMARY_ARROW_TABLE.xlsx")

    styler = df_final.style.applymap(_style_arrow_cell, subset=df_final.columns[2:])
    styler.set_properties(subset=["Parameter", "Channel"], **{"text-align": "left", "font-weight": "bold"})
    styler.to_excel(output_path, index=False)
    print(f"Arrow Table (Independent) saved: {output_path}")


# ==============================================================================
# Arrow tables (paired) - ROBUST (NO FALSE POSITIVES)
# ==============================================================================
def generate_paired_arrow_table(df_stats_pair, df_endo, output_dir="results/tables"):
    """
    Arrow table: Paired (Endo week_1 vs week_8/12/24).
    - Direction: median weekX vs median week_1.
    - Intensity: p<0.05 (1), p<0.01 (2), p<0.001 (3).
    - Exact comparison match to avoid false positives.
      Comparison string must be: "week_1 vs week_12".
    """
    if df_stats_pair is None or df_stats_pair.empty:
        return

    print("   -> Generating Arrow Table (Paired Evolution)...")

    base_week = "week_1"
    compare_weeks = ["week_8", "week_12", "week_24"]

    stats = df_stats_pair.copy()
    stats["p_val"] = pd.to_numeric(stats["p_val"], errors="coerce")
    stats["Feature"] = stats["Feature"].astype(str).str.strip()
    stats["Channel"] = stats["Channel"].astype(str).str.strip()
    stats["Comparison"] = stats["Comparison"].astype(str).str.strip()
    stats["Phase2"] = stats["Phase"].apply(_normalize_phase2)

    sig_rows = stats[stats["p_val"] < 0.05]
    if sig_rows.empty:
        print("      No significant paired results.")
        return

    relevant_items = sig_rows[["Feature", "Channel"]].drop_duplicates().values

    df_endo2 = df_endo.copy()
    df_endo2["phase2"] = df_endo2["phase"].apply(_normalize_phase2)
    df_endo2["week"] = df_endo2["week"].astype(str).str.strip()

    data_rows = []

    for feat, ch in relevant_items:
        row = {"Parameter": feat, "Channel": ch}
        col_ch = _col_channel_for_feature(feat)

        for phase_code, phase_label in [("pre", "Relaxation"), ("contraction", "Contraction")]:
            df_raw = df_endo2[(df_endo2["feature"] == feat) & (df_endo2["phase2"] == phase_code)]

            val_base = df_raw[(df_raw[col_ch] == ch) & (df_raw["week"] == base_week)]["value"].median()
            if pd.isna(val_base):
                continue

            for wk in compare_weeks:
                comp = f"{base_week} vs {wk}"
                stats_subset = stats[
                    (stats["Feature"] == feat) &
                    (stats["Channel"] == ch) &
                    (stats["Phase2"] == phase_code) &
                    (stats["Comparison"] == comp)
                ]

                cell = ""
                if not stats_subset.empty:
                    p = stats_subset["p_val"].min()
                    if pd.notna(p) and p < 0.05:
                        val_post = df_raw[(df_raw[col_ch] == ch) & (df_raw["week"] == wk)]["value"].median()
                        if pd.notna(val_post) and pd.notna(val_base):
                            direction = "↑" if val_post > val_base else "↓"
                            cell = direction * _p_to_count(p)

                row[f"{phase_label} | {wk}"] = cell

        data_rows.append(row)

    df_final = pd.DataFrame(data_rows)
    df_final = _order_arrow_columns(df_final, compare_weeks)
    df_final = _drop_empty_arrow_rows(df_final)

    if df_final.empty:
        print("All rows were empty (no arrows). Table not generated.")
        return

    _ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "PAIRED_ARROW_TABLE.xlsx")

    styler = df_final.style.applymap(_style_arrow_cell, subset=df_final.columns[2:])
    styler.set_properties(subset=["Parameter", "Channel"], **{"text-align": "left", "font-weight": "bold"})
    styler.to_excel(output_path, index=False)
    print(f"Paired Arrow Table saved: {output_path}")


# ==============================================================================
# Arrow tables (GLM): Endo / Age / Births
# ==============================================================================
def _build_glm_arrow_table(df_glm, p_col, coef_col, output_path):
    """
    Arrow table for df_glm.
    Minimum expected columns: Feature, Phase, Channel_or_Pair, Week, p_col.
    Direction is taken from coef_col when available; otherwise bullets are used.
    """
    if df_glm is None or df_glm.empty:
        print(f"df_glm is empty: {os.path.basename(output_path)} not generated")
        return

    df = df_glm.copy()

    needed = {"Feature", "Phase", "Channel_or_Pair", "Week", p_col}
    if not needed.issubset(df.columns):
        print(f"Missing columns in df_glm for {output_path}: {sorted(needed)}")
        return

    if coef_col not in df.columns:
        df[coef_col] = np.nan

    df["Phase_Vis"] = df["Phase"].apply(_phase_to_visual)

    week_order = ["week_1", "week_8", "week_12", "week_24"]
    weeks_present = [w for w in week_order if w in df["Week"].astype(str).unique()]
    if not weeks_present:
        weeks_present = sorted(df["Week"].astype(str).unique())

    df["cell"] = df.apply(lambda r: _cell_arrows(r[p_col], r[coef_col]), axis=1)
    df["col"] = df["Phase_Vis"].astype(str) + " | " + df["Week"].astype(str)

    df_final = (
        df.pivot_table(
            index=["Feature", "Channel_or_Pair"],
            columns="col",
            values="cell",
            aggfunc=lambda x: "" if x.dropna().empty else x.dropna().iloc[0]
        )
        .reset_index()
        .rename(columns={"Feature": "Parameter", "Channel_or_Pair": "Channel"})
    )

    df_final = _order_arrow_columns(df_final, weeks_present)
    df_final = _drop_empty_arrow_rows(df_final)

    if df_final.empty:
        print(f"All rows were empty. {os.path.basename(output_path)} not generated.")
        return

    _ensure_dir(os.path.dirname(output_path))

    styler = df_final.style.applymap(_style_arrow_cell, subset=df_final.columns[2:])
    styler.set_properties(subset=["Parameter", "Channel"], **{"text-align": "left", "font-weight": "bold"})
    styler.to_excel(output_path, index=False)
    print(f"GLM Arrow Table saved: {output_path}")


def generate_glm_arrow_tables(df_glm, output_dir="results/tables"):
    """Generate 3 GLM arrow tables: Endometriosis, Age, and Births."""
    _ensure_dir(output_dir)

    _build_glm_arrow_table(
        df_glm,
        p_col="P_Val_Endo",
        coef_col="Coef_Endo",
        output_path=os.path.join(output_dir, "GLM_ARROW_ENDO.xlsx")
    )

    _build_glm_arrow_table(
        df_glm,
        p_col="P_Val_Age",
        coef_col="Coef_Age",
        output_path=os.path.join(output_dir, "GLM_ARROW_AGE.xlsx")
    )

    _build_glm_arrow_table(
        df_glm,
        p_col="P_Val_Births",
        coef_col="Coef_Births",
        output_path=os.path.join(output_dir, "GLM_ARROW_BIRTHS.xlsx")
    )


def export_forestplots_covariates_summary(
    df_glm,
    effects=("Age", "Births"),
    alpha=0.05,
    phase_pre="pre",
    phase_contra=None,
    sort_by="p",
    top_n=50,
    output_dir="results/plots"
):
    """
    Generate separate forest plots for clinical covariates.

    Output figures:
      - FOREST_Age_Contraction_BS.png
      - FOREST_Age_Relaxation_BS.png
      - FOREST_VaginalDelivery_Contraction_BS.png
      - FOREST_VaginalDelivery_Relaxation_BS.png

    Only significant results are plotted.
    Only B and S channels/pairs are included.
    """

    os.makedirs(output_dir, exist_ok=True)

    if df_glm is None or df_glm.empty:
        print("df_glm is empty. No forest plots generated.")
        return

    df = df_glm.copy()

    df["Feature"] = df["Feature"].astype(str)
    df["Phase"] = df["Phase"].astype(str).str.lower().str.strip()
    df["Week"] = df["Week"].astype(str).str.strip()
    df["Channel_or_Pair"] = df["Channel_or_Pair"].astype(str)

    # filter to B and S channels/pairs only
    df = df[df["Channel_or_Pair"].str.startswith(("B", "S"))].copy()

    if df.empty:
        print("No B or S channels/pairs found. No forest plots generated.")
        return

    # detect contraction phase label
    if phase_contra is None:
        phases = set(df["Phase"].unique())
        candidates = ["contraction_global", "contraction", "global", "contra"]
        phase_contra = next((c for c in candidates if c in phases), None)

        if phase_contra is None:
            print("Contraction phase could not be detected.")
            print("Available phases:", sorted(phases))
            return

    phase_pre = str(phase_pre).lower().strip()
    phase_contra = str(phase_contra).lower().strip()

    phase_map = {
        "Contraction": phase_contra,
        "Relaxation": phase_pre
    }

    effect_config = {
        "Age": {
            "coef": "Coef_Age",
            "ci_low": "CI_Low_Age",
            "ci_high": "CI_High_Age",
            "pval": "P_Val_Age",
            "title": "Effect of age",
            "filename": "Age"
        },
        "Births": {
            "coef": "Coef_Births",
            "ci_low": "CI_Low_Births",
            "ci_high": "CI_High_Births",
            "pval": "P_Val_Births",
            "title": "Effect of previous vaginal delivery",
            "filename": "VaginalDelivery"
        }
    }

    def _prepare_data(df_eff, coef_col, p_col):
        df_eff = df_eff.copy()

        df_eff["Feature_plot"] = df_eff["Feature"].replace({
            "FM": "MDF",
            "SAMPEN": "SampEn",
            "ICOH": "iCOH"
        })

        df_eff["Label"] = (
            df_eff["Feature_plot"].astype(str)
            + " | "
            + df_eff["Channel_or_Pair"].astype(str)
            + " | "
            + df_eff["Week"].astype(str)
        )

        if sort_by == "abscoef":
            df_eff["_ord"] = df_eff[coef_col].abs()
            df_eff = df_eff.sort_values("_ord", ascending=False)
        else:
            df_eff = df_eff.sort_values(p_col, ascending=True)

        return df_eff.head(top_n)

    def _draw_forest_plot(df_plot, coef_col, lo_col, hi_col, p_col, title, output_path):

        coefs = df_plot[coef_col].values
        lows = df_plot[lo_col].values
        highs = df_plot[hi_col].values
        pvals = df_plot[p_col].values
        labels = df_plot["Label"].values

        y = np.arange(len(df_plot))

        fig_h = max(5, 0.45 * len(df_plot) + 2)
        fig, ax = plt.subplots(figsize=(11, fig_h))

        for i in range(len(df_plot)):
            ax.plot([lows[i], highs[i]], [y[i], y[i]], color="black", linewidth=1.3)
            ax.plot(coefs[i], y[i], "o", color="black", markersize=6)

        ax.axvline(0, linestyle="--", color="gray", linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Estimated coefficient (95% CI)", fontsize=11)

        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.yaxis.grid(False)

        x_min = np.nanmin(lows)
        x_max = np.nanmax(highs)
        span = x_max - x_min if x_max > x_min else 1.0

        ax.set_xlim(x_min - 0.2 * span, x_max + 0.65 * span)

        text_x = x_max + 0.06 * span

        for i in range(len(df_plot)):
            ax.text(
                text_x, y[i],
                f"β={coefs[i]:.3f}, p={pvals[i]:.3g}",
                va="center", fontsize=8, color="#333333"
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")

    for effect in effects:

        if effect not in effect_config:
            print(f"Effect '{effect}' is not configured. Skipping.")
            continue

        cfg = effect_config[effect]

        coef_col = cfg["coef"]
        lo_col = cfg["ci_low"]
        hi_col = cfg["ci_high"]
        p_col = cfg["pval"]

        needed = {"Feature", "Phase", "Channel_or_Pair", "Week", coef_col, lo_col, hi_col, p_col}
        missing = [c for c in needed if c not in df.columns]

        if missing:
            print(f"Missing columns for {effect}: {missing}")
            continue

        df_eff = df.dropna(subset=[coef_col, lo_col, hi_col, p_col]).copy()
        df_eff = df_eff[df_eff[p_col] < alpha].copy()

        if df_eff.empty:
            print(f"No significant results for {effect}.")
            continue

        for phase_label, phase_code in phase_map.items():

            df_phase = df_eff[df_eff["Phase"] == phase_code].copy()

            if df_phase.empty:
                print(f"No significant {effect} effects during {phase_label.lower()}.")
                continue

            df_phase = _prepare_data(df_phase, coef_col=coef_col, p_col=p_col)

            plot_title = f"{cfg['title']} during {phase_label.lower()}"
            filename = f"FOREST_{cfg['filename']}_{phase_label}_BS.png"
            output_path = os.path.join(output_dir, filename)

            _draw_forest_plot(
                df_plot=df_phase,
                coef_col=coef_col,
                lo_col=lo_col,
                hi_col=hi_col,
                p_col=p_col,
                title=plot_title,
                output_path=output_path
            )

    print("\n Forest plots for age and previous vaginal delivery generated.")
