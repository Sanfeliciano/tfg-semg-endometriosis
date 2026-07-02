import numpy as np
import scipy.io as sio
import pandas as pd

def load_matlab_signals(path_signals):
    # signals[patient][week]["M1"|"M2"|"M3"|"M4"|"B1"|"B2"|"S1"|"S2"] -> np.array
    mat = sio.loadmat(path_signals, squeeze_me=True, struct_as_record=False)
    root = mat['data_struct']

    data = {}
    # iterate only over patient keys
    for patient_key in [k for k in root._fieldnames if k.startswith("NHC_")]:
        patient_struct = getattr(root, patient_key)
        patient_data = {}

        for week_key in patient_struct._fieldnames:
            week_struct = getattr(patient_struct, week_key)
            if hasattr(week_struct, "signals"):
                signals_struct = week_struct.signals

                week_data = {
                    "M1": np.array(signals_struct.M1).squeeze(),
                    "M2": np.array(signals_struct.M2).squeeze(),
                    "M3": np.array(signals_struct.M3).squeeze(),
                    "M4": np.array(signals_struct.M4).squeeze(),
                    "B1": np.array(signals_struct.B1).squeeze(),
                    "B2": np.array(signals_struct.B2).squeeze(),
                    "S1": np.array(signals_struct.S1).squeeze(),
                    "S2": np.array(signals_struct.S2).squeeze(),
                }
                patient_data[week_key] = week_data

        data[patient_key] = patient_data

    return data


def load_matlab_segments(path_segments, fs=1000):
    # segments[patient][week][phase]
    valid_segments = ['PRE', 'C1', 'C2', 'C3', 'C4', 'C5']
    mat = sio.loadmat(path_segments, squeeze_me=True, struct_as_record=False)
    root = mat["datas_segmentation"]

    data = {}
    for patient_key in [k for k in root._fieldnames if k.startswith("NHC_")]:
        patient_struct = getattr(root, patient_key)
        patient_data = {}

        for week_key in patient_struct._fieldnames:
            week_struct = getattr(patient_struct, week_key)
            week_segments = {}

            for phase_key in week_struct._fieldnames:
                if phase_key not in valid_segments:  # skip unexpected phases
                    continue

                phase_obj = getattr(week_struct, phase_key)

                if hasattr(phase_obj, "limits_event"):  # get segment limits
                    seg = np.array(phase_obj.limits_event)
                    if seg.size > 0 and not np.all(seg == 0):
                        seg_samples = (seg * fs).astype(int)  # convert from seconds to samples
                    if hasattr(phase_obj, "aptitude_channels"):  # read channel aptitude flags
                        aptitude = np.array(phase_obj.aptitude_channels)

                # empty or all-zero limits_event leaves seg_samples empty
                week_segments[phase_key] = {
                    "samples": seg_samples,
                    "aptitude": aptitude
                }

            patient_data[week_key] = week_segments
        data[patient_key] = patient_data

    return data


def load_one_clinical_file(file_path, group_label):
    """Load a single .mat clinical data file. Includes smoking, BMI, deliveries, etc."""
    data_list = []

    try:
        mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

        # locate root structure
        if 'datas_parametrization' in mat:
            root = mat['datas_parametrization']
        else:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if not keys: return pd.DataFrame()
            root = mat[keys[0]]

        # identify patients
        patient_keys = [k for k in root._fieldnames if k.startswith("NHC_")]

        for pat_key in patient_keys:
            pat_data = getattr(root, pat_key)

            age = getattr(pat_data, 'age', np.nan)
            births = getattr(pat_data, 'vaginal_deliveries', np.nan)
            bmi = getattr(pat_data, 'bmi', np.nan)
            years_pain = getattr(pat_data, 'years_pain', np.nan)
            lbp = getattr(pat_data, 'low_back_pain', np.nan)
            musc_alt = getattr(pat_data, 'musculoskeletal_alter', np.nan)
            smoke = getattr(pat_data, 'smoking', np.nan)  # smoking: 0=No, 1=Yes

            data_list.append({
                'patient': pat_key,
                'group': group_label,
                'age': float(age),
                'vaginal_deliveries': float(births),
                'bmi': float(bmi),
                'years_pain': float(years_pain),
                'low_back_pain': float(lbp),
                'musculoskeletal_alter': float(musc_alt),
                'smoking': float(smoke)
            })

        print(f"Loaded: {group_label} -> {len(data_list)} patients.")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data_list)
