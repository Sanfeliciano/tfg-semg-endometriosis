import os
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from src.data.loaders import load_matlab_signals, load_matlab_segments, load_one_clinical_file
from src.signal.process_all import process_all
from src.stats.statistics_ import compute_full_statistics, fit_adjusted_glm
from src.plots.visualization import (
    generate_separated_boxplots,
    plot_pvalue_heatmap,
    plot_effect_sizes,
    generate_summary_arrow_table,
    generate_paired_arrow_table,
    generate_glm_arrow_tables,
    export_forestplots_covariates_summary
)

# ---------------------------
# PATHS
# ---------------------------
PATHS = {
    "segments_ctrl": "data/Segmentations/segmentations_controls_v2_with_probe.mat",
    "segments_endo": "data/Segmentations/segmentations_endometriosis_with_probe.mat",
    "signals_ctrl": "data/Signals/complete_signals_controls_v2_TFG_Alejandro_withB3andB4.mat",
    "signals_endo": "data/Signals/complete_signals_endometriosis_TFG_Alejandro_withB3andB4.mat",
    "clinical_endo": "data/Clinical/clinicalParameters_endometriosis.mat",
    "clinical_ctrl": "data/Clinical/clinicalParameters_controls_v2.mat",
    "xlsx_ctrl_final": "results/results_control_final.xlsx",
    "xlsx_endo_final": "results/results_endometriosis_final.xlsx",
    "out_plots": "results/plots",
    "out_stats": "results/tables"
}
os.makedirs(PATHS["out_plots"], exist_ok=True)
os.makedirs(PATHS["out_stats"], exist_ok=True)


def clean_patient_id(val) -> str:
    """Convert 'NHC_001' / 'Endo01' to '1' for consistent merging."""
    nums = re.findall(r"\d+", str(val))
    return str(int(nums[0])) if nums else str(val)


# ============================================================
# 1) LOAD SIGNALS
# ============================================================
print("[1/5] Signal data management...")

if os.path.exists(PATHS["xlsx_endo_final"]) and os.path.exists(PATHS["xlsx_ctrl_final"]):
    df_endo = pd.read_excel(PATHS["xlsx_endo_final"])
    df_control = pd.read_excel(PATHS["xlsx_ctrl_final"])
else:
    print("   -> Processing from Matlab (this may take a while)...")
    df_endo = process_all(load_matlab_signals(PATHS["signals_endo"]), load_matlab_segments(PATHS["segments_endo"]))
    df_control = process_all(load_matlab_signals(PATHS["signals_ctrl"]), load_matlab_segments(PATHS["segments_ctrl"]))
    df_endo.to_excel(PATHS["xlsx_endo_final"], index=False)
    df_control.to_excel(PATHS["xlsx_ctrl_final"], index=False)

# minimal normalisation
df_endo["week"] = df_endo["week"].astype(str).str.strip()
df_control["week"] = df_control["week"].astype(str).str.strip()
df_endo["phase"] = df_endo["phase"].astype(str).str.lower().str.strip()
df_control["phase"] = df_control["phase"].astype(str).str.lower().str.strip()

# consistent IDs
df_endo["patient"] = df_endo["patient"].apply(clean_patient_id)
df_control["patient"] = df_control["patient"].apply(clean_patient_id)


# ============================================================
# 2) BASIC STATISTICS
# ============================================================
print("[2/5] Basic Statistics (Independent + Paired)...")
df_indep, df_pair = compute_full_statistics(df_endo, df_control)

df_indep.to_excel(os.path.join(PATHS["out_stats"], "stats_independent.xlsx"), index=False)
df_pair.to_excel(os.path.join(PATHS["out_stats"], "stats_paired.xlsx"), index=False)


# ============================================================
# 3) ADJUSTED GLM (AGE + DELIVERIES) + SPSS EXPORT
# ============================================================
print("[3/5] Adjusted GLM model (Age + Vaginal Deliveries)...")

df_c1 = load_one_clinical_file(PATHS["clinical_endo"], "Endometriosis")
df_c2 = load_one_clinical_file(PATHS["clinical_ctrl"], "Control")
df_clin = pd.concat([df_c1, df_c2], ignore_index=True)

# unify IDs
df_endo["patient"] = df_endo["patient"].apply(clean_patient_id)
df_control["patient"] = df_control["patient"].apply(clean_patient_id)
df_clin["patient"] = df_clin["patient"].apply(clean_patient_id)

df_clin = df_clin.drop_duplicates("patient")

# keep only required columns
df_clin = df_clin[["patient", "age", "vaginal_deliveries"]].copy()
df_clin["age"] = pd.to_numeric(df_clin["age"], errors="coerce")
df_clin["vaginal_deliveries"] = pd.to_numeric(df_clin["vaginal_deliveries"], errors="coerce").fillna(0)
df_clin["vaginal_deliveries"] = (df_clin["vaginal_deliveries"] > 0).astype(int)  # binarise to 0/1

# run GLM
df_glm = fit_adjusted_glm(df_endo, df_control, df_clin)
ruta_glm = os.path.join(PATHS["out_stats"], "stats_glm_adjusted.xlsx")
df_glm.to_excel(ruta_glm, index=False)
print(f"   GLM results saved: {ruta_glm}")

# export combined dataset for SPSS as CSV
df_data = pd.concat(
    [df_endo.assign(group_binary=1), df_control.assign(group_binary=0)],
    ignore_index=True
)

if "channel_pair" in df_data.columns:
    df_data["location"] = df_data["channel"].combine_first(df_data["channel_pair"])
else:
    df_data["location"] = df_data["channel"]

df_spss = df_data.merge(df_clin, on="patient", how="inner")
ruta_spss = os.path.join(PATHS["out_stats"], "spss_dataset.csv")
df_spss.to_csv(ruta_spss, index=False, encoding="utf-8")
print(f"   SPSS dataset saved: {ruta_spss}")


# ============================================================
# 4) BOXPLOTS
# ============================================================
print("[4/5] Generating Global Boxplots...")
df_viz = pd.concat(
    [df_endo.assign(group="Endometriosis"), df_control.assign(group="Control")],
    ignore_index=True
)

for feat in df_viz["feature"].dropna().unique():
    generate_separated_boxplots(df_viz, feat, "PRE", PATHS["out_plots"])
    generate_separated_boxplots(df_viz, feat, "GLOBAL", PATHS["out_plots"])


# ============================================================
# 5) HEATMAPS + SUMMARY TABLES
# ============================================================
print("[5/5] Generating Heatmaps...")
plot_pvalue_heatmap(df_indep, "Healthy vs Endometriosis", PATHS["out_plots"], "heatmap_indep.png")
plot_pvalue_heatmap(df_pair, "Evolution", PATHS["out_plots"], "heatmap_pareado.png")
plot_effect_sizes(df_indep, "Independent Effect", PATHS["out_plots"], "efectos_indep.png")

generate_summary_arrow_table(df_indep, df_endo, df_control, output_dir=PATHS["out_stats"])
generate_paired_arrow_table(df_pair, df_endo, output_dir=PATHS["out_stats"])
generate_glm_arrow_tables(df_glm, output_dir=PATHS["out_stats"])

export_forestplots_covariates_summary(df_glm)
print("\nProcess complete.")


# ============================================================
# CONSOLE SUMMARY: SIGNIFICANT RESULTS (with weeks and values)
# ============================================================
def _print_significant_details_console(df_indep, df_pair, df_glm, alpha=0.05, max_rows=200):
    import pandas as pd
    import numpy as np

    def _as_str_table(df, cols, title):
        if df is None or df.empty:
            print(f"\n  {title}: no data")
            return

        df2 = df.copy()
        cols2 = [c for c in cols if c in df2.columns]
        df2 = df2[cols2].copy()

        if df2.empty:
            print(f"\n  {title}: expected columns not found")
            return

        if len(df2) > max_rows:
            print(f"\n  {title}: {len(df2)} rows (showing {max_rows})")
            print(df2.head(max_rows).to_string(index=False))
            return

        print(f"\n  {title}: {len(df2)} rows")
        print(df2.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"SIGNIFICANT RESULTS SUMMARY (p < {alpha})")
    print("=" * 72)

    # --- INDEPENDENT ---
    if df_indep is not None and not df_indep.empty and "p_val" in df_indep.columns:
        indep_sig = df_indep[pd.to_numeric(df_indep["p_val"], errors="coerce") < alpha].copy()
        if "Comparison" in indep_sig.columns:
            indep_sig["Week"] = indep_sig["Comparison"].astype(str)
        _as_str_table(
            indep_sig.sort_values(["Feature", "Channel", "Phase", "Week"], kind="stable"),
            cols=["Feature", "Channel", "Phase", "Week", "p_val", "Test_Used", "Effect_Size"],
            title="Independent"
        )
    else:
        print("\n  Independent: no data or missing p_val column")

    # --- PAIRED ---
    if df_pair is not None and not df_pair.empty and "p_val" in df_pair.columns:
        pair_sig = df_pair[pd.to_numeric(df_pair["p_val"], errors="coerce") < alpha].copy()
        if "Comparison" in pair_sig.columns:
            pair_sig["Week"] = pair_sig["Comparison"].astype(str)
        _as_str_table(
            pair_sig.sort_values(["Feature", "Channel", "Phase", "Week"], kind="stable"),
            cols=["Feature", "Channel", "Phase", "Week", "p_val", "Test_Used", "Effect_Size"],
            title="Paired"
        )
    else:
        print("\n  Paired: no data or missing p_val column")

    # --- GLM - ENDO ---
    if df_glm is not None and not df_glm.empty and "P_Val_Endo" in df_glm.columns:
        glm_endo = df_glm[pd.to_numeric(df_glm["P_Val_Endo"], errors="coerce") < alpha].copy()
        _as_str_table(
            glm_endo.sort_values(["Feature", "Channel_or_Pair", "Phase", "Week"], kind="stable"),
            cols=["Feature", "Channel_or_Pair", "Phase", "Week", "Coef_Endo", "CI_Low_Endo", "CI_High_Endo", "P_Val_Endo", "N"],
            title="GLM (Endometriosis)"
        )
    else:
        print("\n  GLM (Endometriosis): no data or missing P_Val_Endo column")

    # --- GLM - AGE ---
    if df_glm is not None and not df_glm.empty and "P_Val_Age" in df_glm.columns:
        glm_age = df_glm[pd.to_numeric(df_glm["P_Val_Age"], errors="coerce") < alpha].copy()
        _as_str_table(
            glm_age.sort_values(["Feature", "Channel_or_Pair", "Phase", "Week"], kind="stable"),
            cols=["Feature", "Channel_or_Pair", "Phase", "Week", "Coef_Age", "CI_Low_Age", "CI_High_Age", "P_Val_Age", "N"],
            title="GLM (Age)"
        )
    else:
        print("\n  GLM (Age): no data or missing P_Val_Age column")

    # --- GLM - BIRTHS ---
    if df_glm is not None and not df_glm.empty and "P_Val_Births" in df_glm.columns:
        glm_births = df_glm[pd.to_numeric(df_glm["P_Val_Births"], errors="coerce") < alpha].copy()
        _as_str_table(
            glm_births.sort_values(["Feature", "Channel_or_Pair", "Phase", "Week"], kind="stable"),
            cols=["Feature", "Channel_or_Pair", "Phase", "Week", "Coef_Births", "CI_Low_Births", "CI_High_Births", "P_Val_Births", "N"],
            title="GLM (Births)"
        )
    else:
        print("\n  GLM (Births): no data or missing P_Val_Births column")

    print("\n" + "=" * 72 + "\n")


_print_significant_details_console(df_indep, df_pair, df_glm, alpha=0.05, max_rows=200)
