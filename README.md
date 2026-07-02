# TFG - Pelvic floor sEMG analysis

This repository contains the Python code used to process pelvic floor surface EMG recordings, extract signal features and perform the statistical analysis for the Bachelor Thesis.

## Repository structure

```text
.
├── main.py                     # Full analysis pipeline
├── requirements.txt
├── src/
│   ├── data/
│   │   └── loaders.py          # MATLAB .mat file loaders
│   ├── signal/
│   │   ├── features.py         # RMS, median frequency, DI, SampEn, MSCOH, iCOH
│   │   ├── process_all.py      # Feature extraction pipeline over all patients/weeks
│   │   └── utils.py            # Envelope and filtering helpers
│   ├── stats/
│   │   └── statistics_.py      # Statistical tests and adjusted GLM
│   └── plots/
│       ├── visualization.py    # Boxplots, heatmaps, arrow tables, forest plots
│       └── evolution_plot.py   # Longitudinal evolution figure
├── data/                       # Local input data, not included in GitHub
│   ├── Signals/
│   ├── Segmentations/
│   └── Clinical/
├── results/                    # Generated outputs, not included in GitHub
│   ├── tables/
│   └── plots/
└── docs/
    └── figures/                # Example output figures
```

## Data files

The original `.mat` files are not included in the repository because they contain clinical research data. To run the pipeline, place the files locally using the paths expected in `main.py`:

```text
data/Segmentations/segmentations_controls_v2_with_probe.mat
data/Segmentations/segmentations_endometriosis_with_probe.mat
data/Signals/complete_signals_controls_v2_TFG_Alejandro_withB3andB4.mat
data/Signals/complete_signals_endometriosis_TFG_Alejandro_withB3andB4.mat
data/Clinical/clinicalParameters_endometriosis.mat
data/Clinical/clinicalParameters_controls_v2.mat
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the analysis

```bash
python main.py
```

The pipeline generates processed tables, statistical results and figures inside `results/`.

## Main analysis steps

1. Load MATLAB signal, segmentation and clinical files.
2. Extract single-channel features: RMS, median frequency, Dimitrov index and sample entropy.
3. Extract channel-pair connectivity features: magnitude-squared coherence and imaginary coherence.
4. Run independent and paired statistical tests.
5. Run adjusted GLM models controlling for age and vaginal deliveries.
6. Export result tables and plots.

## Privacy note

Input data and generated result files are intentionally excluded from version control. Only reproducible source code and documentation should be uploaded to GitHub.
