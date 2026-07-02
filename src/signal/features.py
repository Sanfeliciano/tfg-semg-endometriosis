import numpy as np
import scipy.signal as sig
import EntropyHub as entropy
from .utils import envelope_lowpass


def single_signal_parameters_segments(signal, fs, feature, segments):
    segments = np.atleast_2d(segments)
    for start, end in segments:
        seg = signal[start:end]

        nperseg = int(fs * 1.0)  # one-second window
        noverlap = nperseg // 2  # 50% overlap
        f, psd = sig.welch(seg, fs=fs, nperseg=nperseg, noverlap=noverlap)
        psd_norm = psd / np.sum(psd)

        match feature.upper():

            case 'RMS':  # reflects muscle activation intensity
                rms = np.sqrt(np.mean(seg ** 2))
                return rms

            case 'FM':  # median frequency: 50% of spectral energy lies below this point
                cumsum = np.cumsum(psd_norm)
                f50 = np.interp(0.5, cumsum, f)
                return f50

            case 'DI':  # fatigue index: higher DI means more energy at low frequencies
                f_safe = np.where(f == 0, 1e-10, f)
                numerator = np.sum(psd * (1 / f_safe))
                denominator = np.sum(psd * (f_safe ** 5))
                di = numerator / denominator
                return di

            case 'SAMPEN':  # sample entropy with m=2 and r=0.15
                std_seg = np.std(seg)
                seg_z = (seg - np.mean(seg)) / std_seg
                sampen = entropy.SampEn(seg_z, m=2, r=0.15)
                value = sampen[0][2]  # take the m=2 value from the returned array
                return value

    return np.nan


def two_signal_parameters_segments(signal_1, signal_2, fs, feature, segments):
    segments = np.atleast_2d(segments)

    for start, end in segments:
        seg_1 = signal_1[start:end]
        seg_2 = signal_2[start:end]

        # minimum duration check: 4.1 seconds
        duration_seconds = len(seg_1) / fs
        if duration_seconds < 4.1:
            continue

        env_1 = envelope_lowpass(seg_1, fs)
        env_2 = envelope_lowpass(seg_2, fs)

        # window configuration: at fs=1000 Hz, nperseg=4000 (4 seconds)
        nperseg = int(4 * fs)
        noverlap = int(3.9 * fs)  # 3.9 s overlap

        match feature.upper():
            case 'MSCOH':
                f, Cxy = sig.coherence(env_1, env_2, fs=fs, nperseg=nperseg, noverlap=noverlap)
                freq_range = (f >= 0) & (f <= 10)
                return np.mean(Cxy[freq_range])

            case 'ICOH':
                f, Pxx = sig.welch(env_1, fs=fs, nperseg=nperseg, noverlap=noverlap)
                f, Pyy = sig.welch(env_2, fs=fs, nperseg=nperseg, noverlap=noverlap)
                f, Pxy = sig.csd(env_1, env_2, fs=fs, nperseg=nperseg, noverlap=noverlap)

                iCOH = np.imag(Pxy) / np.sqrt(Pxx * Pyy)
                freq_range = (f >= 0) & (f <= 10)
                return np.mean(np.abs(iCOH[freq_range]))

    return np.nan
