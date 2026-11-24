# KISR wrist-PPG sensor-sizing study - data & code package

Accompanies the manuscript "Sensor-Channel and Front-End Sizing Evidence for
Wrist-PPG Heart-Rate Monitoring: A Cross-Dataset Controlled Study" submitted to
IEEE Sensors Journal (Regular Paper).

## Contents
- code/      analysis, experiment and figure-generation scripts (Python 3.12;
             numpy, scipy, matplotlib, wfdb, neurokit2, python-docx, Pillow)
- results/   per-window error caches and all summary JSONs for the three
             cohorts (Wrist 19 records / 8 subjects / 1664 windows; BIDMC 53
             subjects / 6306 windows; PPG-DaLiA 15 subjects / 2825 windows),
             plus equivalence (TOST), energy-model and estimator-probe outputs
- figures/   the ten manuscript figures at print size (3.4 in wide, 600 dpi)
             and the two author-supplied illustration panels
- checksums.sha256

## Raw data (not redistributed; public sources)
- Wrist: https://physionet.org/content/wrist/1.0.0/
- BIDMC: https://physionet.org/physiobank/database/bidmc/
- PPG-DaLiA: https://archive.ics.uci.edu/dataset/495/ppg+dalia
  (official page https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html;
  archive sha256 recorded in results/dalia_acquisition.json)

## Reproduction
1. pip install numpy scipy matplotlib wfdb neurokit2 python-docx Pillow
2. Download the three datasets into data/wrist, data/bidmc, data/dalia
   (code/download_*.py automate BIDMC and DaLiA).
3. python code/run_wrist_expanded.py etc. regenerate results/*.json;
   python code/render_col*.py regenerate figures/*.png;
   python code/build_8p.py rebuilds the manuscript docx from the JSEN template.
All statistics are subject-level (Wilcoxon signed-rank, subject-clustered
bootstrap 4000 resamples); equivalence uses TOST with pre-specified margins
(+/-2 bpm Wrist/DaLiA, +/-0.5 bpm BIDMC).

## Licence and contact
Code: MIT. Results/figures: CC-BY 4.0. Raw data retain their original licences.
Corresponding author: Peilin Han (4124640044@email.sdfmu.edu.cn).
