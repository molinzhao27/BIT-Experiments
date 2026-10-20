Same 84 conditions, trials and nested 10×10 CV as Phase 1. Accuracy in %, chance 14.3%.
Δ = mean per-condition accuracy difference vs the baseline [95% bootstrap CI over conditions]; better/worse = conditions where McNemar q < 0.05 (BH within model); Wilcoxon = signed-rank test of the 84 paired differences.

| Model | Input | Mean acc. (84) | Paper's 9 arrays | Inner-speech conditions | Significant conditions | Δ vs baseline (pp) | Better / worse | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| Gaussian NB, pooled variance (paper) | 500 ms window mean | 36.7 | 43.6 | 28.7 | 54/84 | — | — | — |
| Shrinkage LDA (Ledoit-Wolf) | 500 ms window mean | 40.5 | 48.3 | 31.7 | 55/84 | +3.7 [+2.5, +5.0] | 8 / 0 | 1.4e-07 |
| Logistic regression (L2) | 500 ms window mean | 39.6 | 47.4 | 30.7 | 52/84 | +2.9 [+1.8, +4.0] | 4 / 1 | 2.4e-06 |
| Linear SVM (L2) | 500 ms window mean | 39.8 | 47.3 | 31.3 | 55/84 | +3.1 [+2.1, +4.2] | 4 / 0 | 3.5e-07 |
| 1D-CNN (time series) | binned time series | 42.2 | 50.5 | 32.9 | 56/84 | +5.5 [+3.8, +7.3] | 15 / 0 | 1.6e-07 |
| GRU (time series) | binned time series | 36.1 | 43.5 | 27.7 | 50/84 | -0.7 [-2.5, +1.1] | 3 / 5 | 0.77 |
