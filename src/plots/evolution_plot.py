import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================================
# CONFIGURATION
# ==========================================================
CHANNEL = "B2"          # "S2" for deep, "B2" for superficial
OUTPUT_NAME = "results/plots/FIGURE_FINAL.png"
DATASET_FILE = "results/tables/spss_dataset.csv"

FEATURES = ["RMS", "FM", "DI", "SAMPEN"]

WEEKS = ["week_1", "week_8", "week_12", "week_24"]
WEEK_LABELS = [0, 8, 12, 24]

LABELS = {
    "RMS": "RMS (cV)",
    "FM": "MDF (Hz)",
    "DI": "DI (×10⁻⁶)",
    "SAMPEN": "SampEn",
}


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def get_phase_subset(dataframe, feature, mode):
    """
    mode:
      - 'contraction' -> everything except PRE
      - 'relaxation'  -> PRE only
    """
    sub = dataframe[
        (dataframe["feature"] == feature) &
        (dataframe["channel"] == CHANNEL)
    ].copy()

    if mode == "contraction":
        sub = sub[sub["phase"] != "pre"]
    elif mode == "relaxation":
        sub = sub[sub["phase"] == "pre"]
    else:
        raise ValueError("mode must be 'contraction' or 'relaxation'")

    return sub


def summarize(vals):
    vals = vals.dropna()

    if len(vals) == 0:
        return pd.NA, pd.NA, pd.NA

    return vals.median(), vals.quantile(0.25), vals.quantile(0.75)


def get_stats(dataframe, feature, mode):
    """Return median and percentiles for Endometriosis (per week) and Healthy (pooled baseline)."""
    sub = get_phase_subset(dataframe, feature, mode)

    ctrl_vals = sub[sub["group_binary"] == 0]["value"]
    med_c, q1_c, q3_c = summarize(ctrl_vals)

    rows = []

    for wk in WEEKS:
        wk_vals = sub[
            (sub["group_binary"] == 1) &
            (sub["week"] == wk)
        ]["value"]

        med_e, q1_e, q3_e = summarize(wk_vals)

        rows.append({
            "week": wk,
            "med_endo": med_e,
            "q1_endo": q1_e,
            "q3_endo": q3_e,
            "med_ctrl": med_c,
            "q1_ctrl": q1_c,
            "q3_ctrl": q3_c,
        })

    return pd.DataFrame(rows)


def draw(ax, stats, ylabel):
    # --- ENDOMETRIOSIS ---
    ax.plot(WEEK_LABELS, stats["med_endo"], color="black", linewidth=2.4)
    ax.plot(WEEK_LABELS, stats["q1_endo"], color="black", linestyle="--", linewidth=1.3)
    ax.plot(WEEK_LABELS, stats["q3_endo"], color="black", linestyle="--", linewidth=1.3)

    # --- HEALTHY ---
    ax.plot(WEEK_LABELS, stats["med_ctrl"], color="gray", linewidth=2.4)
    ax.plot(WEEK_LABELS, stats["q1_ctrl"], color="gray", linestyle="--", linewidth=1.3)
    ax.plot(WEEK_LABELS, stats["q3_ctrl"], color="gray", linestyle="--", linewidth=1.3)

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(WEEK_LABELS)
    ax.grid(False)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(ymin - 0.08 * yrange, ymax + 0.08 * yrange)

    ax.tick_params(axis="both", labelsize=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_NAME), exist_ok=True)
    df = pd.read_csv(DATASET_FILE)

    df["feature"] = df["feature"].astype(str).str.strip().str.upper()
    df["phase"] = df["phase"].astype(str).str.strip().str.lower()
    df["channel"] = df["channel"].astype(str).str.strip().str.upper()
    df["week"] = df["week"].astype(str).str.strip().str.lower()

    df.loc[df["feature"] == "RMS", "value"] *= 1e2
    df.loc[df["feature"] == "DI", "value"] *= 1e6

    fig, axes = plt.subplots(4, 2, figsize=(14, 8), sharex=True)

    axes[0, 0].set_title("CONTRACTION", fontsize=18, fontweight="bold", pad=6)
    axes[0, 1].set_title("RELAXATION", fontsize=18, fontweight="bold", pad=6)

    for i, feat in enumerate(FEATURES):
        stats_con = get_stats(df, feat, "contraction")
        stats_rel = get_stats(df, feat, "relaxation")

        draw(axes[i, 0], stats_con, LABELS[feat])
        draw(axes[i, 1], stats_rel, LABELS[feat])

    # x-axis labels on bottom row only
    axes[3, 0].set_xlabel("Week", fontsize=13)
    axes[3, 1].set_xlabel("Week", fontsize=13)

    # ==========================================================
    # LEGEND
    # ==========================================================
    legend_elements = [
        Line2D([0], [0], color="black", lw=2.4, label="Endometriosis"),
        Line2D([0], [0], color="gray", lw=2.4, label="Healthy"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.015)
    )

    plt.subplots_adjust(
        left=0.07, right=0.98, top=0.92, bottom=0.12,
        hspace=0.28, wspace=0.25
    )

    plt.savefig(OUTPUT_NAME, dpi=300, bbox_inches="tight")
    plt.show()
