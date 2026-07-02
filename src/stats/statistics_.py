import pandas as pd
import pingouin as pg
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_full_statistics(df_endo, df_control):
    print("--- Running Basic Statistical Tests ---")
    features = df_endo['feature'].unique()
    phases = df_endo['phase'].unique()
    base_week = 'week_1'
    followup_weeks = ['week_8', 'week_12', 'week_24']
    all_weeks = [base_week] + followup_weeks

    res_indep = []
    res_pair = []

    for feature in features:
        for phase in phases:
            channels = list(
                df_endo[(df_endo['feature'] == feature) & (df_endo['phase'] == phase)]['channel'].dropna().unique())
            if 'channel_pair' in df_endo.columns:
                pairs = list(df_endo[(df_endo['feature'] == feature) & (df_endo['phase'] == phase)][
                                 'channel_pair'].dropna().unique())
                channels += pairs

            for channel in channels:
                col_ch = 'channel' if channel in df_endo['channel'].values else 'channel_pair'

                # --- A) INDEPENDENT TEST ---
                d_ctrl = df_control[(df_control['feature'] == feature) & (df_control['phase'] == phase) & (
                            df_control[col_ch] == channel)]['value'].values

                # filter coherence values at ceiling
                if feature in ['MSCOH', 'ICOH']: d_ctrl = d_ctrl[d_ctrl < 0.999]

                for week in all_weeks:
                    d_endo = df_endo[
                        (df_endo['feature'] == feature) & (df_endo['phase'] == phase) & (df_endo['week'] == week) & (
                                    df_endo[col_ch] == channel)]['value'].values
                    if feature in ['MSCOH', 'ICOH']: d_endo = d_endo[d_endo < 0.999]

                    if len(d_ctrl) >= 3 and len(d_endo) >= 3:
                        try:
                            norm = (pg.normality(d_ctrl).iloc[0]['pval'] > 0.05) and (
                                        pg.normality(d_endo).iloc[0]['pval'] > 0.05)
                        except:
                            norm = False

                        if norm:
                            eq_var = pg.homoscedasticity([d_ctrl, d_endo]).iloc[0]['pval'] > 0.05
                            res = pg.ttest(d_endo, d_ctrl, correction=not eq_var)
                            test_name = "T-Test" if eq_var else "Welch"
                        else:
                            res = pg.mwu(d_endo, d_ctrl)
                            test_name = "Mann-Whitney"

                        res_indep.append({
                            'Feature': feature, 'Phase': phase, 'Channel': channel,
                            'Comparison': f"Control vs {week}",
                            'p_val': res['p_val'].values[0],
                            'Test_Used': test_name,
                            'Effect_Size': res['RBC'].values[0] if 'RBC' in res.columns else np.nan
                        })

                # --- B) PAIRED TEST ---
                subset = df_endo[
                    (df_endo['feature'] == feature) & (df_endo['phase'] == phase) & (df_endo[col_ch] == channel)]
                if feature in ['MSCOH', 'ICOH']: subset = subset[subset['value'] < 0.999]

                for wk in followup_weeks:
                    pair_data = subset[subset['week'].isin([base_week, wk])]
                    wide = pair_data.pivot(index='patient', columns='week', values='value').dropna()

                    if base_week in wide and wk in wide and len(wide) >= 3:
                        x, y = wide[base_week].values, wide[wk].values
                        try:
                            norm_diff = pg.normality(x - y).iloc[0]['pval'] > 0.05
                        except:
                            norm_diff = False

                        if norm_diff:
                            res = pg.ttest(x, y, paired=True)
                            test_name = "Paired T-Test"
                        else:
                            res = pg.wilcoxon(x, y)
                            test_name = "Wilcoxon"

                        res_pair.append({
                            'Feature': feature, 'Phase': phase, 'Channel': channel,
                            'Comparison': f"{base_week} vs {wk}",
                            'p_val': res['p_val'].values[0],
                            'Test_Used': test_name
                        })

    return pd.DataFrame(res_indep), pd.DataFrame(res_pair)


def fit_adjusted_glm(df_endo, df_control, df_clinical, debug=True, min_n_per_group=3):
    """
    GLM: Control (pooled) vs Endometriosis (per week),
    adjusted for age and deliveries (binary 0/1 as fixed factor).
    """

    def dprint(*args):
        if debug:
            print(*args)

    dprint("   -> Running adjusted GLM...")

    for name, df in [("df_endo", df_endo), ("df_control", df_control), ("df_clinical", df_clinical)]:
        if df is None or df.empty:
            dprint(f"{name} is empty -> cannot model.")
            return pd.DataFrame()

    required_endo = {"patient", "feature", "phase", "week", "value"}
    required_ctrl = {"patient", "feature", "phase", "value"}
    if not required_endo.issubset(df_endo.columns):
        dprint(f"Missing columns in df_endo: {sorted(required_endo - set(df_endo.columns))}")
        return pd.DataFrame()
    if not required_ctrl.issubset(df_control.columns):
        dprint(f"Missing columns in df_control: {sorted(required_ctrl - set(df_control.columns))}")
        return pd.DataFrame()

    # -------------------------
    # 1) Prepare group data
    # -------------------------
    endo = df_endo.copy()
    endo["group_binary"] = 1

    ctrl = df_control.copy()
    ctrl["group_binary"] = 0

    endo["phase"] = endo["phase"].astype(str).str.lower().str.strip()
    ctrl["phase"] = ctrl["phase"].astype(str).str.lower().str.strip()

    # unify channel / channel_pair into a single location column
    endo["location"] = endo.get("channel")
    if "channel_pair" in endo.columns:
        endo["location"] = endo["location"].combine_first(endo.get("channel_pair"))

    ctrl["location"] = ctrl.get("channel")
    if "channel_pair" in ctrl.columns:
        ctrl["location"] = ctrl["location"].combine_first(ctrl.get("channel_pair"))

    endo = endo.dropna(subset=["location"])
    ctrl = ctrl.dropna(subset=["location"])

    dprint(f"Endo rows after location filter: {len(endo)} | Control rows: {len(ctrl)}")

    # -------------------------
    # 2) Merge clinical data
    # -------------------------
    clin = df_clinical.copy()

    births_col = "vaginal_births" if "vaginal_births" in clin.columns else "vaginal_deliveries"
    if births_col not in clin.columns or "age" not in clin.columns or "patient" not in clin.columns:
        dprint("df_clinical is missing required columns: patient, age, vaginal_births/vaginal_deliveries")
        dprint("   Clinical columns found:", list(clin.columns))
        return pd.DataFrame()

    clin = clin[["patient", "age", births_col]].rename(columns={births_col: "vaginal_births"})
    clin["patient"] = clin["patient"].astype(str).str.strip()
    endo["patient"] = endo["patient"].astype(str).str.strip()
    ctrl["patient"] = ctrl["patient"].astype(str).str.strip()

    clin["vaginal_births"] = pd.to_numeric(clin["vaginal_births"], errors="coerce")

    endo_before = len(endo)
    ctrl_before = len(ctrl)

    endo = endo.merge(clin, on="patient", how="inner")
    ctrl = ctrl.merge(clin, on="patient", how="inner")

    dprint(f"After clinical merge -> Endo: {endo_before}→{len(endo)} | Control: {ctrl_before}→{len(ctrl)}")

    if endo.empty or ctrl.empty:
        dprint("No data left after clinical merge.")
        dprint("   -> This usually means patient IDs do not match between signals and clinical data.")
        return pd.DataFrame()

    # -------------------------
    # 3) GLM per comparison
    # -------------------------
    results = []
    counters = {
        "ctrl_empty": 0,
        "after_dropna_empty": 0,
        "no_both_groups": 0,
        "min_n_fail": 0,
        "glm_fail": 0,
        "ok": 0,
    }

    base_groups = endo.groupby(["feature", "phase", "location"])

    for (feat, ph, loc), endo_sub_allweeks in base_groups:
        ctrl_sub = ctrl[(ctrl["feature"] == feat) & (ctrl["phase"] == ph) & (ctrl["location"] == loc)].copy()
        if ctrl_sub.empty:
            counters["ctrl_empty"] += 1
            continue

        if feat in ["MSCOH", "ICOH"]:
            ctrl_sub = ctrl_sub[ctrl_sub["value"] < 0.999]

        for wk, endo_sub in endo_sub_allweeks.groupby("week"):
            sub = pd.concat([ctrl_sub, endo_sub], ignore_index=True)

            needed = ["value", "group_binary", "age", "vaginal_births"]
            sub = sub.dropna(subset=needed)
            if sub.empty:
                counters["after_dropna_empty"] += 1
                continue

            if feat in ["MSCOH", "ICOH"]:
                sub = sub[sub["value"] < 0.999]
                if sub.empty:
                    counters["after_dropna_empty"] += 1
                    continue

            if sub["group_binary"].nunique() < 2:
                counters["no_both_groups"] += 1
                continue

            n0 = (sub["group_binary"] == 0).sum()
            n1 = (sub["group_binary"] == 1).sum()
            if n0 < min_n_per_group or n1 < min_n_per_group:
                counters["min_n_fail"] += 1
                continue

            try:
                formula = "value ~ group_binary + age + C(vaginal_births)"
                model = smf.glm(formula, data=sub, family=sm.families.Gaussian())
                res = model.fit()

                ci = res.conf_int()
                se = res.bse

                birth_terms = [t for t in res.params.index if t.startswith("C(vaginal_births)[T.")]
                birth_term = "C(vaginal_births)[T.1]" if "C(vaginal_births)[T.1]" in res.params.index else (
                    birth_terms[0] if birth_terms else None
                )

                def grab(term):
                    if term is None:
                        return np.nan, np.nan, np.nan, np.nan, np.nan
                    lo, hi = (ci.loc[term, 0], ci.loc[term, 1]) if term in ci.index else (np.nan, np.nan)
                    return (
                        res.params.get(term, np.nan),
                        se.get(term, np.nan),
                        lo,
                        hi,
                        res.pvalues.get(term, np.nan),
                    )

                coef_g, se_g, lo_g, hi_g, p_g = grab("group_binary")
                coef_a, se_a, lo_a, hi_a, p_a = grab("age")
                coef_b, se_b, lo_b, hi_b, p_b = grab(birth_term)

                results.append({
                    "Feature": feat,
                    "Phase": ph,
                    "Channel_or_Pair": loc,
                    "Week": wk,

                    "Coef_Endo": coef_g, "SE_Endo": se_g, "CI_Low_Endo": lo_g, "CI_High_Endo": hi_g, "P_Val_Endo": p_g,
                    "Coef_Age": coef_a, "SE_Age": se_a, "CI_Low_Age": lo_a, "CI_High_Age": hi_a, "P_Val_Age": p_a,

                    "Births_Term": birth_term if birth_term else "",
                    "Coef_Births": coef_b, "SE_Births": se_b, "CI_Low_Births": lo_b, "CI_High_Births": hi_b, "P_Val_Births": p_b,

                    "N": int(len(sub)),
                    "N_Control": int(n0),
                    "N_Endo": int(n1),
                })

                counters["ok"] += 1

            except Exception as e:
                counters["glm_fail"] += 1
                continue

    dprint("---- DEBUG SUMMARY ----")
    for k, v in counters.items():
        dprint(f"{k}: {v}")

    out = pd.DataFrame(results)
    dprint(f"Final rows in results: {len(out)}")

    return out
