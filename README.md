# Weighted observation channels split coordinated attacks into detectability regimes in crowd-sourced continuous control

Reproducibility package for the manuscript:
*"Weighted observation channels split coordinated attacks into detectability regimes in crowd-sourced continuous control."*
Single author: BongKeun Song (Institute of Fluid Mechanics / LSTM, FAU Erlangen-Nürnberg).

This archive contains everything needed to reproduce the numerical results and figures.
Python + NumPy only for the core; Matplotlib is used for figures. No additional data
collection is required — the Monte Carlo table is included.

## Contents

```
code/
  topology_boundary_pilot_v5.py   MAIN simulator + experiment (7 topologies, all figures/tables)
  variance_channel_verify.py      Theorem B verification (pre-registered PV1–PV3, lightweight)
  make_figures.py                 regenerates all four manuscript figures from data/
  topology_boundary_pilot_v2.py   earlier pilot (grid-censoring diagnosis; provenance)
  topology_boundary_pilot_v3.py   earlier pilot (grid opened to 0.45; reversal first observed)
  topology_boundary_pilot_v4.py   earlier pilot (grid to 0.50, c-sweep, CSV export)
data/
  topology_v5_results.csv         Monte Carlo results (MC=200): p*, status, RMSE per cell
figures/
  fig1_pstar_reversal.png         detection boundary p*(c): the reversal
  fig2_variance_asymmetry.png     Var[gamma_C] sign asymmetry
  fig3_tradeoff.png               command–monitor trade-off
  fig4_heatmap.png                p* by topology × coordination
derivation/
  derivation_note.md              first-principles derivation (Prop. 1, Prop. 2, Var[theta])
README.md, LICENSE, requirements.txt, CITATION.cff
```

## Requirements

- Python 3.9+
- numpy
- matplotlib (only for make_figures.py)

Install: `pip install -r requirements.txt`

## Reproducing the results

1. **Main experiment (Table 1, Table 2, Figure 1, Figure 4 data).**
   Regenerates the full 7-topology × attack × coordination sweep at MC=200.
   ```
   cd code
   python topology_boundary_pilot_v5.py 200 1
   ```
   Runtime scales with MC and grid; use `python topology_boundary_pilot_v5.py 100 1`
   for a faster (~half-cost) preview. Output is written to `topology_v5_results.csv`
   in the working directory (identical in structure to `data/topology_v5_results.csv`).

2. **Theorem B verification (pre-registered predictions PV1–PV3).**
   ```
   cd code
   python variance_channel_verify.py 4000
   ```
   Prints, for p=0.4: the opposite/scatter variances across c, and PASS/FAIL for
   PV1 (scatter variance monotone increasing, superlinear),
   PV2 (opposite variance ∝ 1−c, R² > 0.9),
   PV3 (scatter/opposite variance ratio ≥ 50 near c=1).

3. **Figures.**
   ```
   cd code
   python make_figures.py
   ```
   Reads `data/topology_v5_results.csv` and the simulator, writes the four PNGs
   into `figures/`. **Note:** Figure 1, 3, and 4 are deterministic functions of the
   archived CSV and will match the manuscript exactly. Figure 2 (and any panel that
   redraws from the simulator rather than the CSV) depends on live Monte Carlo draws;
   re-running `make_figures.py` reproduces the same qualitative pattern (variance
   sign asymmetry, same order of magnitude) but not necessarily byte-identical pixels
   to the manuscript's Figure 2, since it is regenerated rather than read from the
   archived table. The exact PNGs used in the manuscript are included in `figures/`
   as shipped; treat `make_figures.py` as a regeneration/verification tool, not as
   the source of the shipped images.

## Notes on determinism

Random seeds are fixed per cell inside the scripts, so re-running reproduces the same
numbers up to the platform's NumPy RNG. Absolute detection boundaries p* are calibrated
quantities: they depend on the CUSUM constants defined at the top of the simulator
(threshold H, slack, warm-up length, adaptive rate, variance floor). Changing those
constants shifts the absolute p* values but not the qualitative results (the mean-channel
blindness of Proposition 1 and the variance sign-asymmetry of Proposition 2), which are
argued in the manuscript to be independent of these implementation constants.

## What is closed vs. semi-empirical (see derivation/derivation_note.md)

- Closed / implementation-independent: Proposition 1 (mean-channel blindness, an identity),
  Proposition 2 (variance sign asymmetry: opposite ∝ 1−c, scatter superlinear), and the
  closed form for the mean-direction angular variance Var[theta].
- Semi-empirical / unresolved: the absolute detection boundary p* (stopping-time
  approximation) and the strong-attack absolute variance (the mean-direction wobble
  couples to the weighted sum and does not separate as an independent term).
