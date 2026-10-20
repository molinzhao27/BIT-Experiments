Δ = mean per-condition accuracy difference vs the paper's Gaussian NB (pp), 84 conditions = 12 arrays × 7 behaviors from 4 participants. 95% percentile bootstrap intervals (10000 resamples) over: conditions (treats them as independent, as model_comparison.md does), arrays (12), sessions (28 participant × behavior recordings, each with every array that recorded it), and arrays × behaviors resampled independently (two-way). The independent units are the 4 participants: with 4/4 improved, a two-sided sign test gives p = 0.125, its minimum (ties, i.e. Δ = 0 exactly, are dropped from the sign test).

| Model | Δ (pp) | by condition | by array | by session | array × behavior | participants improved | sign test p |
|---|---|---|---|---|---|---|---|
| Shrinkage LDA (Ledoit-Wolf) | +3.7 | [+2.6, +5.0] | [+1.9, +5.6] | [+2.1, +5.5] | [+1.5, +6.3] | 4/4 | 0.125 |
| Logistic regression (L2) | +2.9 | [+1.8, +4.0] | [+1.2, +4.6] | [+1.5, +4.3] | [+0.8, +5.3] | 3/4 (1 tied) | 0.25 |
| Linear SVM (L2) | +3.1 | [+2.1, +4.2] | [+1.3, +5.0] | [+1.7, +4.6] | [+1.0, +5.5] | 3/4 | 0.625 |
| 1D-CNN (time series) | +5.5 | [+3.7, +7.3] | [+3.0, +8.0] | [+3.3, +7.9] | [+2.0, +9.5] | 4/4 | 0.125 |
| GRU (time series) | -0.7 | [-2.5, +1.1] | [-2.6, +1.4] | [-3.2, +1.8] | [-3.7, +2.2] | 2/4 | 1 |

Mean Δ per participant (pp; arrays per participant: T12 2, T15 4, T16 3, T17 3):

| Model | T12 | T15 | T16 | T17 |
|---|---|---|---|---|
| Shrinkage LDA (Ledoit-Wolf) | +2.1 | +7.2 | +3.5 | +0.4 |
| Logistic regression (L2) | +1.4 | +5.6 | +3.0 | +0.0 |
| Linear SVM (L2) | -0.1 | +6.4 | +3.7 | +0.3 |
| 1D-CNN (time series) | +3.7 | +7.4 | +6.9 | +2.6 |
| GRU (time series) | -3.5 | +2.3 | +1.1 | -4.5 |
