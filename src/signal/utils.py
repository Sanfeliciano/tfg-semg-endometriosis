import numpy as np
import pandas as pd
import scipy.signal as sig

def envelope_lowpass(signal, fs):
    rectified = np.abs(signal)
    b, a = sig.butter(4, 10 / (fs / 2), btype='low')  # 4th-order Butterworth, cutoff 10 Hz
    envelope = sig.filtfilt(b, a, rectified)
    return envelope

def get_data_for_statistic_analysis(df_combined, feature, channel, phase, week, group):
    colum_name = 'channel_pair' if feature in ['MSCOH', 'ICOH'] else 'channel'

    df_filtered = df_combined[
        (df_combined['feature'] == feature) &
        (df_combined['phase'] == phase) &
        (df_combined['week'] == week) &
        (df_combined['group'] == group) &
        (df_combined[colum_name] == channel)
    ]
    return df_filtered['value'].copy().dropna()
