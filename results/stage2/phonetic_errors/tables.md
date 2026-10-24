## Phonetic side (all pairs; excess = observed − length-matched null, 95% trial-bootstrap CI)

| Run | WER | subs | pairs (OOV) | phon obs / null | phon excess | p | feat excess | sound-alike obs / null | excess sound-alikes / 100 ref words | homophones obs / null | first / last phoneme excess | phon excess beyond spelling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | 25.2% | 1121 | 1093 (2.5%) | 0.288 / 0.096 | +0.192 [+0.178, +0.206] | <0.001 | +0.179 [+0.168, +0.190] | 23.6% / 2.7% | 4.8 | 1.8% / 0.0% | +0.305 / +0.229 | +0.107 [+0.097, +0.117] |
| Fast-dLLM v2 (diffusion) · null_shuffled | 114.7% | 3785 | 3760 (0.7%) | 0.090 / 0.090 | +0.000 [-0.004, +0.004] | 0.446 | -0.002 [-0.006, +0.002] | 3.0% / 3.3% | -0.2 | 0.0% / 0.0% | +0.000 / -0.003 | +0.004 [-0.000, +0.007] |
| Aero | 34.2% | 1374 | 1371 (0.2%) | 0.209 / 0.091 | +0.118 [+0.105, +0.131] | <0.001 | +0.106 [+0.096, +0.118] | 16.1% / 2.7% | 3.9 | 0.5% / 0.0% | +0.173 / +0.153 | +0.063 [+0.054, +0.073] |
| Aero · null_shuffled | 115.6% | 3801 | 3798 (0.1%) | 0.090 / 0.089 | +0.000 [-0.004, +0.005] | 0.416 | -0.001 [-0.005, +0.004] | 3.3% / 3.3% | -0.0 | 0.0% / 0.0% | +0.001 / -0.001 | +0.003 [-0.000, +0.007] |
| Qwen base, no prompt | 43.8% | 1641 | 1631 (0.6%) | 0.166 / 0.092 | +0.074 [+0.065, +0.084] | <0.001 | +0.066 [+0.058, +0.075] | 10.4% / 2.5% | 2.7 | 0.4% / 0.0% | +0.109 / +0.096 | +0.043 [+0.036, +0.050] |
| Qwen base, no prompt · null_shuffled | 113.2% | 3755 | 3746 (0.2%) | 0.086 / 0.088 | -0.002 [-0.007, +0.002] | 0.849 | -0.004 [-0.009, +0.000] | 3.2% / 3.3% | -0.1 | 0.0% / 0.0% | +0.000 / -0.005 | +0.001 [-0.003, +0.005] |
| Qwen Instruct | 55.7% | 2101 | 2096 (0.2%) | 0.137 / 0.089 | +0.048 [+0.040, +0.056] | <0.001 | +0.043 [+0.036, +0.050] | 6.9% / 2.5% | 1.9 | 0.1% / 0.0% | +0.075 / +0.047 | +0.027 [+0.021, +0.033] |
| Qwen Instruct · null_shuffled | 113.7% | 3729 | 3724 (0.1%) | 0.093 / 0.089 | +0.004 [-0.000, +0.008] | 0.058 | +0.002 [-0.002, +0.006] | 3.5% / 3.4% | 0.0 | 0.0% / 0.0% | +0.008 / +0.001 | +0.005 [+0.001, +0.008] |
| Aero, random-init encoder (4a) | 33.6% | 1376 | 1368 (0.6%) | 0.217 / 0.094 | +0.124 [+0.111, +0.137] | <0.001 | +0.111 [+0.100, +0.122] | 16.2% / 2.7% | 3.9 | 0.8% / 0.1% | +0.192 / +0.155 | +0.068 [+0.059, +0.077] |
| Aero, random-init encoder (4a) · null_shuffled | 115.3% | 3820 | 3811 (0.2%) | 0.089 / 0.089 | +0.000 [-0.004, +0.005] | 0.441 | +0.000 [-0.004, +0.004] | 3.2% / 3.2% | -0.0 | 0.1% / 0.0% | -0.000 / -0.003 | +0.003 [-0.000, +0.007] |

## Language side (all pairs)

| Run | spelling excess | meaning (sem) obs / null | sem excess | p | related obs / null | commoner obs / null | context fit (ctx, nats) excess | ctx p | fits context better than ref obs / null | ctx coverage | median zipf: correct / substituted ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | +0.160 [+0.147, +0.173] | 0.137 / 0.053 | +0.084 [+0.075, +0.094] | <0.001 | 9.7% / 2.9% | 68% / 64% | +1.90 [+1.71, +2.10] | <0.001 | 16% / 8% | 97% | 6.38 / 4.96 |
| Fast-dLLM v2 (diffusion) · null_shuffled | -0.003 [-0.007, +0.001] | 0.129 / 0.119 | +0.010 [+0.006, +0.014] | <0.001 | 7.4% / 6.8% | 54% / 53% | +0.19 [+0.11, +0.27] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Aero | +0.105 [+0.093, +0.117] | 0.122 / 0.060 | +0.062 [+0.052, +0.072] | <0.001 | 8.4% / 3.5% | 60% / 58% | +1.82 [+1.60, +2.05] | <0.001 | 15% / 8% | 95% | 6.41 / 5.19 |
| Aero · null_shuffled | -0.003 [-0.007, +0.001] | 0.124 / 0.117 | +0.007 [+0.003, +0.011] | <0.001 | 7.0% / 6.8% | 53% / 52% | +0.17 [+0.08, +0.26] | <0.001 | 5% / 5% | 96% | 7.09 / 6.00 |
| Qwen base, no prompt | +0.060 [+0.051, +0.070] | 0.121 / 0.069 | +0.052 [+0.045, +0.061] | <0.001 | 9.3% / 3.8% | 58% / 58% | +1.57 [+1.39, +1.75] | <0.001 | 13% / 8% | 97% | 6.46 / 5.37 |
| Qwen base, no prompt · null_shuffled | -0.004 [-0.008, -0.000] | 0.127 / 0.121 | +0.006 [+0.002, +0.010] | 0.002 | 6.7% / 6.8% | 54% / 53% | +0.13 [+0.05, +0.22] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Qwen Instruct | +0.039 [+0.032, +0.047] | 0.115 / 0.078 | +0.038 [+0.032, +0.044] | <0.001 | 7.7% / 4.8% | 53% / 53% | +1.20 [+1.05, +1.34] | <0.001 | 10% / 6% | 97% | 6.50 / 5.53 |
| Qwen Instruct · null_shuffled | +0.002 [-0.003, +0.006] | 0.124 / 0.116 | +0.008 [+0.004, +0.012] | <0.001 | 7.2% / 6.8% | 52% / 51% | +0.18 [+0.10, +0.26] | <0.001 | 5% / 5% | 96% | 7.09 / 6.00 |
| Aero, random-init encoder (4a) | +0.102 [+0.090, +0.115] | 0.124 / 0.061 | +0.063 [+0.054, +0.072] | <0.001 | 9.1% / 4.0% | 58% / 58% | +1.83 [+1.63, +2.03] | <0.001 | 16% / 8% | 95% | 6.41 / 5.21 |
| Aero, random-init encoder (4a) · null_shuffled | -0.005 [-0.009, -0.001] | 0.124 / 0.116 | +0.008 [+0.004, +0.012] | <0.001 | 7.0% / 6.7% | 52% / 51% | +0.20 [+0.11, +0.29] | <0.001 | 5% / 5% | 95% | 7.09 / 6.00 |

## Sound-alike (phon >= 0.5) x related in meaning (sem >= 0.4), share of pairs

| Run | alike + related | alike only | related only | neither |
|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | 4.0% | 19.6% | 5.7% | 70.7% |
| Fast-dLLM v2 (diffusion) · null_shuffled | 1.2% | 1.8% | 6.2% | 90.8% |
| Aero | 3.1% | 13.0% | 5.3% | 78.6% |
| Aero · null_shuffled | 1.2% | 2.1% | 5.8% | 90.9% |
| Qwen base, no prompt | 2.8% | 7.6% | 6.6% | 83.1% |
| Qwen base, no prompt · null_shuffled | 1.1% | 2.1% | 5.6% | 91.2% |
| Qwen Instruct | 1.9% | 5.0% | 5.9% | 87.3% |
| Qwen Instruct · null_shuffled | 1.2% | 2.3% | 5.9% | 90.6% |
| Aero, random-init encoder (4a) | 3.8% | 12.4% | 5.3% | 78.5% |
| Aero, random-init encoder (4a) · null_shuffled | 0.9% | 2.2% | 6.0% | 90.8% |

## Slices (excess with 95% CI; n pairs)

| Run | slice | n | phon | phon beyond spelling | sem | ctx (nats) |
|---|---|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | seen | 72 | +0.164 [+0.118, +0.218] | +0.095 [+0.062, +0.134] | +0.104 [+0.071, +0.147] | +1.97 [+1.14, +2.98] |
| Fast-dLLM v2 (diffusion) | unseen | 1021 | +0.194 [+0.180, +0.209] | +0.108 [+0.098, +0.118] | +0.083 [+0.073, +0.093] | +1.90 [+1.70, +2.11] |
| Fast-dLLM v2 (diffusion) | mostly_right | 739 | +0.206 [+0.188, +0.224] | +0.114 [+0.102, +0.127] | +0.094 [+0.083, +0.107] | +2.28 [+2.02, +2.53] |
| Fast-dLLM v2 (diffusion) | mostly_wrong | 354 | +0.164 [+0.140, +0.187] | +0.092 [+0.076, +0.108] | +0.063 [+0.049, +0.077] | +1.12 [+0.83, +1.41] |
| Fast-dLLM v2 (diffusion) | content | 756 | +0.184 [+0.169, +0.199] | +0.105 [+0.094, +0.116] | +0.074 [+0.064, +0.086] | +1.84 [+1.61, +2.09] |
| Fast-dLLM v2 (diffusion) | no_stem | 1070 | +0.184 [+0.171, +0.198] | +0.104 [+0.095, +0.114] | +0.079 [+0.070, +0.089] | +1.87 [+1.68, +2.08] |
| Fast-dLLM v2 (diffusion) | not_alike | 835 | +0.089 [+0.079, +0.098] | +0.046 [+0.038, +0.053] | +0.074 [+0.064, +0.084] | +1.85 [+1.62, +2.08] |
| Aero | seen | 90 | +0.081 [+0.042, +0.128] | +0.049 [+0.012, +0.093] | +0.040 [+0.012, +0.073] | +1.22 [+0.03, +2.71] |
| Aero | unseen | 1281 | +0.120 [+0.108, +0.134] | +0.064 [+0.055, +0.074] | +0.064 [+0.054, +0.074] | +1.86 [+1.65, +2.09] |
| Aero | mostly_right | 512 | +0.175 [+0.154, +0.196] | +0.094 [+0.080, +0.109] | +0.100 [+0.084, +0.117] | +3.16 [+2.85, +3.49] |
| Aero | mostly_wrong | 859 | +0.084 [+0.070, +0.099] | +0.045 [+0.034, +0.056] | +0.039 [+0.029, +0.050] | +1.03 [+0.78, +1.28] |
| Aero | content | 787 | +0.127 [+0.112, +0.143] | +0.067 [+0.056, +0.079] | +0.066 [+0.054, +0.079] | +2.04 [+1.78, +2.31] |
| Aero | no_stem | 1355 | +0.112 [+0.100, +0.125] | +0.061 [+0.052, +0.070] | +0.057 [+0.048, +0.066] | +1.79 [+1.57, +2.02] |
| Aero | not_alike | 1150 | +0.044 [+0.035, +0.052] | +0.017 [+0.010, +0.025] | +0.049 [+0.039, +0.058] | +1.68 [+1.46, +1.92] |
| Qwen base, no prompt | seen | 95 | +0.033 [+0.002, +0.065] | +0.014 [-0.003, +0.034] | +0.051 [+0.021, +0.089] | +1.70 [+1.05, +2.49] |
| Qwen base, no prompt | unseen | 1536 | +0.077 [+0.067, +0.087] | +0.045 [+0.037, +0.052] | +0.052 [+0.044, +0.061] | +1.57 [+1.38, +1.76] |
| Qwen base, no prompt | mostly_right | 487 | +0.107 [+0.089, +0.126] | +0.060 [+0.047, +0.073] | +0.080 [+0.066, +0.095] | +2.82 [+2.52, +3.12] |
| Qwen base, no prompt | mostly_wrong | 1144 | +0.060 [+0.048, +0.072] | +0.036 [+0.027, +0.045] | +0.040 [+0.031, +0.050] | +1.04 [+0.84, +1.24] |
| Qwen base, no prompt | content | 863 | +0.079 [+0.067, +0.091] | +0.044 [+0.035, +0.053] | +0.059 [+0.048, +0.071] | +1.79 [+1.55, +2.03] |
| Qwen base, no prompt | no_stem | 1615 | +0.070 [+0.060, +0.080] | +0.041 [+0.034, +0.048] | +0.049 [+0.041, +0.057] | +1.56 [+1.38, +1.74] |
| Qwen base, no prompt | not_alike | 1462 | +0.026 [+0.019, +0.032] | +0.015 [+0.009, +0.020] | +0.040 [+0.032, +0.048] | +1.46 [+1.27, +1.64] |
| Qwen Instruct | seen | 108 | +0.042 [+0.014, +0.075] | +0.028 [+0.006, +0.052] | +0.052 [+0.027, +0.089] | +0.92 [+0.32, +1.70] |
| Qwen Instruct | unseen | 1988 | +0.048 [+0.040, +0.056] | +0.026 [+0.020, +0.033] | +0.037 [+0.031, +0.043] | +1.21 [+1.07, +1.36] |
| Qwen Instruct | mostly_right | 361 | +0.069 [+0.052, +0.088] | +0.037 [+0.023, +0.050] | +0.087 [+0.070, +0.105] | +3.20 [+2.82, +3.59] |
| Qwen Instruct | mostly_wrong | 1735 | +0.043 [+0.035, +0.052] | +0.024 [+0.018, +0.031] | +0.027 [+0.021, +0.034] | +0.77 [+0.64, +0.91] |
| Qwen Instruct | content | 949 | +0.046 [+0.036, +0.056] | +0.023 [+0.016, +0.031] | +0.050 [+0.040, +0.060] | +1.49 [+1.28, +1.71] |
| Qwen Instruct | no_stem | 2077 | +0.043 [+0.036, +0.051] | +0.025 [+0.019, +0.031] | +0.035 [+0.029, +0.041] | +1.18 [+1.04, +1.32] |
| Qwen Instruct | not_alike | 1952 | +0.016 [+0.010, +0.021] | +0.008 [+0.004, +0.013] | +0.030 [+0.024, +0.036] | +1.12 [+0.98, +1.27] |
| Aero, random-init encoder (4a) | seen | 55 | +0.118 [+0.052, +0.204] | +0.075 [+0.034, +0.131] | +0.077 [+0.026, +0.165] | +1.53 [+0.63, +2.40] |
| Aero, random-init encoder (4a) | unseen | 1313 | +0.124 [+0.110, +0.138] | +0.068 [+0.059, +0.077] | +0.062 [+0.053, +0.071] | +1.84 [+1.64, +2.05] |
| Aero, random-init encoder (4a) | mostly_right | 453 | +0.175 [+0.153, +0.197] | +0.095 [+0.078, +0.112] | +0.107 [+0.090, +0.124] | +3.29 [+2.95, +3.65] |
| Aero, random-init encoder (4a) | mostly_wrong | 915 | +0.098 [+0.082, +0.115] | +0.055 [+0.045, +0.065] | +0.041 [+0.031, +0.051] | +1.11 [+0.91, +1.32] |
| Aero, random-init encoder (4a) | content | 802 | +0.126 [+0.112, +0.140] | +0.067 [+0.057, +0.077] | +0.072 [+0.060, +0.084] | +2.02 [+1.76, +2.28] |
| Aero, random-init encoder (4a) | no_stem | 1350 | +0.118 [+0.105, +0.131] | +0.066 [+0.057, +0.075] | +0.059 [+0.050, +0.068] | +1.80 [+1.61, +2.00] |
| Aero, random-init encoder (4a) | not_alike | 1147 | +0.050 [+0.041, +0.060] | +0.026 [+0.019, +0.033] | +0.050 [+0.041, +0.059] | +1.70 [+1.50, +1.91] |

## Slice differences within a run (a − b, 95% CI)

| Run | metric | a − b | diff |
|---|---|---|---|
| Fast-dLLM v2 (diffusion) | phon | seen − unseen | -0.030 [-0.078, +0.025] |
| Fast-dLLM v2 (diffusion) | phon | mostly_right − mostly_wrong | +0.042 [+0.013, +0.071] |
| Fast-dLLM v2 (diffusion) | sem | seen − unseen | +0.022 [-0.014, +0.066] |
| Fast-dLLM v2 (diffusion) | sem | mostly_right − mostly_wrong | +0.032 [+0.013, +0.050] |
| Fast-dLLM v2 (diffusion) | ctx | seen − unseen | +0.076 [-0.801, +1.118] |
| Fast-dLLM v2 (diffusion) | ctx | mostly_right − mostly_wrong | +1.161 [+0.775, +1.544] |
| Aero | phon | seen − unseen | -0.039 [-0.081, +0.009] |
| Aero | phon | mostly_right − mostly_wrong | +0.091 [+0.066, +0.116] |
| Aero | sem | seen − unseen | -0.024 [-0.054, +0.011] |
| Aero | sem | mostly_right − mostly_wrong | +0.061 [+0.041, +0.081] |
| Aero | ctx | seen − unseen | -0.642 [-1.846, +0.866] |
| Aero | ctx | mostly_right − mostly_wrong | +2.136 [+1.737, +2.545] |
| Qwen base, no prompt | phon | seen − unseen | -0.044 [-0.076, -0.011] |
| Qwen base, no prompt | phon | mostly_right − mostly_wrong | +0.047 [+0.025, +0.069] |
| Qwen base, no prompt | sem | seen − unseen | -0.001 [-0.032, +0.038] |
| Qwen base, no prompt | sem | mostly_right − mostly_wrong | +0.040 [+0.024, +0.057] |
| Qwen base, no prompt | ctx | seen − unseen | +0.129 [-0.534, +0.949] |
| Qwen base, no prompt | ctx | mostly_right − mostly_wrong | +1.777 [+1.427, +2.130] |
| Qwen Instruct | phon | seen − unseen | -0.006 [-0.036, +0.028] |
| Qwen Instruct | phon | mostly_right − mostly_wrong | +0.026 [+0.006, +0.047] |
| Qwen Instruct | sem | seen − unseen | +0.015 [-0.011, +0.053] |
| Qwen Instruct | sem | mostly_right − mostly_wrong | +0.060 [+0.041, +0.078] |
| Qwen Instruct | ctx | seen − unseen | -0.294 [-0.909, +0.490] |
| Qwen Instruct | ctx | mostly_right − mostly_wrong | +2.427 [+2.018, +2.835] |
| Aero, random-init encoder (4a) | phon | seen − unseen | -0.006 [-0.073, +0.080] |
| Aero, random-init encoder (4a) | phon | mostly_right − mostly_wrong | +0.076 [+0.048, +0.104] |
| Aero, random-init encoder (4a) | sem | seen − unseen | +0.015 [-0.037, +0.103] |
| Aero, random-init encoder (4a) | sem | mostly_right − mostly_wrong | +0.066 [+0.047, +0.085] |
| Aero, random-init encoder (4a) | ctx | seen − unseen | -0.309 [-1.241, +0.566] |
| Aero, random-init encoder (4a) | ctx | mostly_right − mostly_wrong | +2.182 [+1.792, +2.598] |

## Run contrasts (difference in mean excess, a − b; paired trial bootstrap)

| a | b | metric | slice | diff |
|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | Aero | phon | all | +0.074 [+0.058, +0.092] |
| Fast-dLLM v2 (diffusion) | Aero | phon | mostly_right | +0.031 [+0.006, +0.056] |
| Fast-dLLM v2 (diffusion) | Aero | phon|orth | all | +0.044 [+0.032, +0.057] |
| Fast-dLLM v2 (diffusion) | Aero | phon|orth | mostly_right | +0.020 [+0.002, +0.039] |
| Fast-dLLM v2 (diffusion) | Aero | sem | all | +0.022 [+0.010, +0.034] |
| Fast-dLLM v2 (diffusion) | Aero | sem | mostly_right | -0.006 [-0.025, +0.013] |
| Fast-dLLM v2 (diffusion) | Aero | ctx | all | +0.084 [-0.170, +0.343] |
| Fast-dLLM v2 (diffusion) | Aero | ctx | mostly_right | -0.883 [-1.257, -0.525] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon | all | +0.118 [+0.102, +0.134] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon | mostly_right | +0.098 [+0.074, +0.123] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon|orth | all | +0.064 [+0.053, +0.076] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon|orth | mostly_right | +0.054 [+0.037, +0.071] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | sem | all | +0.032 [+0.021, +0.043] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | sem | mostly_right | +0.014 [-0.004, +0.031] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | ctx | all | +0.330 [+0.079, +0.585] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | ctx | mostly_right | -0.538 [-0.930, -0.156] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | phon | all | +0.144 [+0.129, +0.160] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | phon | mostly_right | +0.136 [+0.111, +0.162] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | phon|orth | all | +0.081 [+0.070, +0.091] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | phon|orth | mostly_right | +0.078 [+0.059, +0.097] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | sem | all | +0.046 [+0.036, +0.057] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | sem | mostly_right | +0.007 [-0.013, +0.028] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | ctx | all | +0.709 [+0.471, +0.947] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct | ctx | mostly_right | -0.924 [-1.371, -0.480] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | phon | all | +0.068 [+0.051, +0.086] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | phon | mostly_right | +0.031 [+0.004, +0.058] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | phon|orth | all | +0.039 [+0.027, +0.052] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | phon|orth | mostly_right | +0.019 [-0.001, +0.039] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | sem | all | +0.021 [+0.009, +0.034] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | sem | mostly_right | -0.012 [-0.031, +0.007] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | ctx | all | +0.077 [-0.180, +0.339] |
| Fast-dLLM v2 (diffusion) | Aero, random-init encoder (4a) | ctx | mostly_right | -1.016 [-1.401, -0.638] |
| Aero | Qwen base, no prompt | phon | all | +0.043 [+0.029, +0.057] |
| Aero | Qwen base, no prompt | phon | mostly_right | +0.067 [+0.043, +0.093] |
| Aero | Qwen base, no prompt | phon|orth | all | +0.020 [+0.010, +0.031] |
| Aero | Qwen base, no prompt | phon|orth | mostly_right | +0.034 [+0.017, +0.052] |
| Aero | Qwen base, no prompt | sem | all | +0.010 [-0.001, +0.021] |
| Aero | Qwen base, no prompt | sem | mostly_right | +0.020 [-0.001, +0.040] |
| Aero | Qwen base, no prompt | ctx | all | +0.247 [+0.011, +0.493] |
| Aero | Qwen base, no prompt | ctx | mostly_right | +0.345 [-0.056, +0.751] |
| Aero | Qwen Instruct | phon | all | +0.070 [+0.056, +0.084] |
| Aero | Qwen Instruct | phon | mostly_right | +0.105 [+0.078, +0.133] |
| Aero | Qwen Instruct | phon|orth | all | +0.037 [+0.026, +0.047] |
| Aero | Qwen Instruct | phon|orth | mostly_right | +0.058 [+0.038, +0.078] |
| Aero | Qwen Instruct | sem | all | +0.024 [+0.014, +0.035] |
| Aero | Qwen Instruct | sem | mostly_right | +0.013 [-0.010, +0.035] |
| Aero | Qwen Instruct | ctx | all | +0.625 [+0.387, +0.871] |
| Aero | Qwen Instruct | ctx | mostly_right | -0.040 [-0.518, +0.440] |
| Aero | Aero, random-init encoder (4a) | phon | all | -0.006 [-0.021, +0.009] |
| Aero | Aero, random-init encoder (4a) | phon | mostly_right | -0.000 [-0.026, +0.027] |
| Aero | Aero, random-init encoder (4a) | phon|orth | all | -0.005 [-0.016, +0.006] |
| Aero | Aero, random-init encoder (4a) | phon|orth | mostly_right | -0.001 [-0.019, +0.018] |
| Aero | Aero, random-init encoder (4a) | sem | all | -0.001 [-0.013, +0.011] |
| Aero | Aero, random-init encoder (4a) | sem | mostly_right | -0.007 [-0.028, +0.015] |
| Aero | Aero, random-init encoder (4a) | ctx | all | -0.007 [-0.249, +0.237] |
| Aero | Aero, random-init encoder (4a) | ctx | mostly_right | -0.133 [-0.524, +0.264] |
| Qwen base, no prompt | Qwen Instruct | phon | all | +0.027 [+0.015, +0.038] |
| Qwen base, no prompt | Qwen Instruct | phon | mostly_right | +0.038 [+0.012, +0.062] |
| Qwen base, no prompt | Qwen Instruct | phon|orth | all | +0.017 [+0.008, +0.025] |
| Qwen base, no prompt | Qwen Instruct | phon|orth | mostly_right | +0.024 [+0.005, +0.042] |
| Qwen base, no prompt | Qwen Instruct | sem | all | +0.015 [+0.006, +0.024] |
| Qwen base, no prompt | Qwen Instruct | sem | mostly_right | -0.007 [-0.028, +0.014] |
| Qwen base, no prompt | Qwen Instruct | ctx | all | +0.379 [+0.184, +0.570] |
| Qwen base, no prompt | Qwen Instruct | ctx | mostly_right | -0.386 [-0.862, +0.075] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | phon | all | -0.049 [-0.064, -0.035] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | phon | mostly_right | -0.067 [-0.093, -0.041] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | phon|orth | all | -0.025 [-0.036, -0.015] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | phon|orth | mostly_right | -0.035 [-0.054, -0.015] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | sem | all | -0.010 [-0.021, +0.001] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | sem | mostly_right | -0.026 [-0.046, -0.007] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | ctx | all | -0.253 [-0.484, -0.016] |
| Qwen base, no prompt | Aero, random-init encoder (4a) | ctx | mostly_right | -0.478 [-0.897, -0.060] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon | all | -0.076 [-0.091, -0.062] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon | mostly_right | -0.105 [-0.133, -0.077] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon|orth | all | -0.042 [-0.052, -0.032] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon|orth | mostly_right | -0.058 [-0.080, -0.036] |
| Qwen Instruct | Aero, random-init encoder (4a) | sem | all | -0.025 [-0.036, -0.014] |
| Qwen Instruct | Aero, random-init encoder (4a) | sem | mostly_right | -0.019 [-0.042, +0.003] |
| Qwen Instruct | Aero, random-init encoder (4a) | ctx | all | -0.632 [-0.856, -0.397] |
| Qwen Instruct | Aero, random-init encoder (4a) | ctx | mostly_right | -0.093 [-0.577, +0.387] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | phon | all | +0.192 [+0.177, +0.207] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | phon|orth | all | +0.103 [+0.093, +0.114] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | sem | all | +0.074 [+0.064, +0.085] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | ctx | all | +1.716 [+1.504, +1.931] |
| Aero | Aero · null_shuffled | phon | all | +0.117 [+0.104, +0.131] |
| Aero | Aero · null_shuffled | phon|orth | all | +0.060 [+0.050, +0.070] |
| Aero | Aero · null_shuffled | sem | all | +0.055 [+0.044, +0.066] |
| Aero | Aero · null_shuffled | ctx | all | +1.650 [+1.409, +1.903] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | phon | all | +0.077 [+0.066, +0.088] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | phon|orth | all | +0.042 [+0.034, +0.050] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | sem | all | +0.046 [+0.038, +0.055] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | ctx | all | +1.439 [+1.236, +1.642] |
| Qwen Instruct | Qwen Instruct · null_shuffled | phon | all | +0.044 [+0.036, +0.053] |
| Qwen Instruct | Qwen Instruct · null_shuffled | phon|orth | all | +0.022 [+0.015, +0.029] |
| Qwen Instruct | Qwen Instruct · null_shuffled | sem | all | +0.030 [+0.023, +0.037] |
| Qwen Instruct | Qwen Instruct · null_shuffled | ctx | all | +1.020 [+0.859, +1.188] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | phon | all | +0.123 [+0.110, +0.137] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | phon|orth | all | +0.065 [+0.055, +0.075] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | sem | all | +0.055 [+0.045, +0.065] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | ctx | all | +1.626 [+1.408, +1.845] |

## 20 most common substitutions (reference → decoded, count)

| # | Fast-dLLM v2 (diffusion) | Aero | Qwen base, no prompt | Qwen Instruct | Aero, random-init encoder (4a) |
|---|---|---|---|---|---|
| 1 | has → as (4) | the → a (3) | in → the (2) | of → in (3) | in → and (3) |
| 2 | the → they (3) | to → the (3) | the → it's (2) | as → a (3) | for → your (3) |
| 3 | their → there (3) | he's → it's (3) | these → those (2) | in → and (3) | in → it (3) |
| 4 | such → just (3) | out → up (3) | on → and (2) | of → the (3) | to → do (3) |
| 5 | clothes → close (3) | the → that (2) | the → a (2) | a → and (3) | well → will (2) |
| 6 | in → and (3) | it → in (2) | a → to (2) | in → it (3) | has → as (2) |
| 7 | and → a (3) | the → do (2) | not → but (2) | in → of (3) | a → he (2) |
| 8 | in → it (3) | of → is (2) | to → the (2) | it → it's (3) | an → a (2) |
| 9 | anyway → anymore (2) | up → of (2) | to → a (2) | of → over (3) | in → is (2) |
| 10 | is → in (2) | such → just (2) | that → there (2) | in → the (2) | as → you (2) |
| 11 | it → he (2) | look → like (2) | the → go (2) | on → of (2) | how → i (2) |
| 12 | oh → who (2) | him → of (2) | of → do (2) | in → with (2) | are → a (2) |
| 13 | got → not (2) | year → here (2) | as → have (2) | in → a (2) | as → for (2) |
| 14 | its → it's (2) | a → he (2) | i → you (2) | the → little (2) | year → here (2) |
| 15 | has → is (2) | and → a (2) | for → of (2) | work → week (2) | the → they (2) |
| 16 | bay → pay (2) | of → the (2) | more → for (2) | in → to (2) | education → experience (2) |
| 17 | once → what's (2) | had → and (2) | money → morning (2) | for → of (2) | building → buying (2) |
| 18 | look → like (2) | new → you (2) | his → is (2) | up → it (2) | its → it's (2) |
| 19 | half → have (2) | they're → there (2) | run → one (2) | her → the (2) | high → i (2) |
| 20 | what's → what (2) | as → has (2) | have → i (2) | of → her (2) | and → at (2) |
