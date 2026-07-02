import numpy as np
import pandas as pd
import time
from .features import (
    single_signal_parameters_segments,
    two_signal_parameters_segments
)


def _log_progress(count, total, start_time):
    elapsed = time.time() - start_time
    remaining = (elapsed / max(count, 1)) * (total - count)
    print(f"Progress: {count}/{total} ({count/total*100:.1f}%), ETA: {remaining:.1f}s", flush=True)


def process_all(signals, segments):
    all_results = []
    channels_single = ['M1', 'M2', 'M3', 'M4', 'B1', 'B2', 'S1', 'S2']
    channels_pairs = [['B1', 'B2'], ['S1', 'S2'], ['M1', 'M3'], ['M2', 'M4'], ['M3', 'M4']]
    feature_single = ['RMS', 'FM', 'DI', 'SAMPEN']
    feature_pairs = ['MSCOH', 'ICOH']

    aptitude_channels = {"M1": 0, "M2": 1, "M3": 2, "M4": 3, "B1": 4, "B2": 5, "S1": 8, "S2": 9}

    # count real operations for progress tracking
    total_iterations = 0
    for patient in signals.keys():
        for week in signals[patient].keys():
            for phase in segments[patient][week].keys():
                total_iterations += len(channels_single) * len(feature_single) \
                                    + len(channels_pairs) * len(feature_pairs)

    count = 0
    start_time = time.time()

    for patient in signals.keys():
        for week in signals[patient].keys():
            for phase in segments[patient][week].keys():
                segment_data = segments[patient][week][phase]
                segment_samples = segment_data["samples"]

                # --------- Single channels ----------
                for channel in channels_single:
                    signal = signals[patient][week][channel]
                    index_aptitude = aptitude_channels[channel]

                    for feature in feature_single:
                        if segment_samples.size == 0 or segment_data["aptitude"][index_aptitude] == 0:
                            value = np.nan
                        else:
                            value = single_signal_parameters_segments(
                                signal, fs=1000, feature=feature, segments=segment_samples
                            )

                        all_results.append({
                            "patient": patient,
                            "week": week,
                            "phase": phase,
                            "channel": channel,
                            "feature": feature,
                            "value": value
                        })

                        count += 1
                        _log_progress(count, total_iterations, start_time)

                # --------- Channel pairs ----------
                for channel_1, channel_2 in channels_pairs:
                    signal_1 = signals[patient][week][channel_1]
                    signal_2 = signals[patient][week][channel_2]

                    index_aptitude_1 = aptitude_channels[channel_1]
                    index_aptitude_2 = aptitude_channels[channel_2]

                    for feature in feature_pairs:
                        if (segment_samples.size == 0 or
                            segment_data["aptitude"][index_aptitude_1] == 0 or
                            segment_data["aptitude"][index_aptitude_2] == 0):
                            value = np.nan
                        else:
                            value = two_signal_parameters_segments(
                                signal_1, signal_2, fs=1000, feature=feature, segments=segment_samples
                            )

                        all_results.append({
                            "patient": patient,
                            "week": week,
                            "phase": phase,
                            "channel_pair": f"{channel_1}_{channel_2}",
                            "feature": feature,
                            "value": value
                        })

                        count += 1
                        _log_progress(count, total_iterations, start_time)

    print("\nProcessing complete")

    df_results = pd.DataFrame(all_results)

    # ============================================================
    # COLLAPSE: keep only phase=pre and contraction_global
    # contraction_global = median(c1..c5)
    # ============================================================
    CONTRACTION_PHASES = {"c1", "c2", "c3", "c4", "c5"}
    PHASE_PRE = "pre"
    PHASE_GLOBAL = "contraction"

    df_results["phase"] = df_results["phase"].astype(str).str.lower().str.strip()

    # 1) PRE: keep as-is (including NaNs)
    df_pre_single = df_results[(df_results["phase"] == PHASE_PRE) & (df_results["channel"].notna())].copy()
    df_pre_pairs  = df_results[(df_results["phase"] == PHASE_PRE) & (df_results["channel_pair"].notna())].copy()

    # 2) Contractions: median per (patient, week, channel, feature) and per (patient, week, channel_pair, feature)
    df_con_single = df_results[df_results["phase"].isin(CONTRACTION_PHASES) & (df_results["channel"].notna())].copy()
    if not df_con_single.empty:
        df_glob_single = (
            df_con_single
            .groupby(["patient", "week", "channel", "feature"], as_index=False)["value"]
            .median()
        )
        df_glob_single["phase"] = PHASE_GLOBAL
    else:
        df_glob_single = pd.DataFrame(columns=["patient", "week", "channel", "feature", "value", "phase"])

    df_con_pairs = df_results[df_results["phase"].isin(CONTRACTION_PHASES) & (df_results["channel_pair"].notna())].copy()
    if not df_con_pairs.empty:
        df_glob_pairs = (
            df_con_pairs
            .groupby(["patient", "week", "channel_pair", "feature"], as_index=False)["value"]
            .median()
        )
        df_glob_pairs["phase"] = PHASE_GLOBAL
    else:
        df_glob_pairs = pd.DataFrame(columns=["patient", "week", "channel_pair", "feature", "value", "phase"])

    # 3) Concatenate PRE + GLOBAL and return
    df_final = pd.concat([df_pre_single, df_glob_single, df_pre_pairs, df_glob_pairs], ignore_index=True)

    # sort: pre first
    phase_order = {PHASE_PRE: 0, PHASE_GLOBAL: 1}
    df_final["_phase_order"] = df_final["phase"].map(phase_order).fillna(99)

    sort_cols = ["patient", "week", "_phase_order", "feature"]
    if "channel" in df_final.columns:
        sort_cols = ["patient", "week", "channel", "_phase_order", "feature"]

    df_final = df_final.sort_values(sort_cols).drop(columns="_phase_order")

    return df_final
