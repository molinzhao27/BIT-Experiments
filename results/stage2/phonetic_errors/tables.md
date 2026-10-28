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
| Qwen Instruct + diffusion | 23.6% | 1063 | 1029 (3.2%) | 0.298 / 0.095 | +0.203 [+0.189, +0.219] | <0.001 | +0.187 [+0.176, +0.199] | 25.6% / 2.7% | 4.9 | 1.6% / 0.0% | +0.330 / +0.260 | +0.112 [+0.102, +0.123] |
| Qwen Instruct + diffusion · null_shuffled | 115.0% | 3793 | 3763 (0.8%) | 0.088 / 0.089 | -0.001 [-0.005, +0.004] | 0.598 | -0.003 [-0.007, +0.002] | 3.2% / 3.3% | -0.1 | 0.0% / 0.0% | -0.000 / -0.005 | +0.002 [-0.001, +0.006] |
| Aero + diffusion | 21.8% | 964 | 930 (3.5%) | 0.295 / 0.095 | +0.200 [+0.185, +0.217] | <0.001 | +0.187 [+0.175, +0.200] | 25.9% / 2.7% | 4.5 | 1.3% / 0.0% | +0.309 / +0.248 | +0.107 [+0.096, +0.118] |
| Aero + diffusion · null_shuffled | 115.0% | 3790 | 3759 (0.8%) | 0.089 / 0.089 | -0.000 [-0.005, +0.004] | 0.522 | -0.002 [-0.006, +0.003] | 3.2% / 3.3% | -0.1 | 0.0% / 0.0% | +0.001 / -0.002 | +0.003 [-0.001, +0.007] |
| Arm A: phoneme CTC encoder + Aero | 15.5% | 687 | 663 (3.5%) | 0.391 / 0.098 | +0.293 [+0.271, +0.315] | <0.001 | +0.237 [+0.220, +0.254] | 41.9% / 3.2% | 5.4 | 3.2% / 0.1% | +0.408 / +0.374 | +0.157 [+0.142, +0.172] |
| Arm A: phoneme CTC encoder + Aero · null_shuffled | 115.2% | 3802 | 3779 (0.6%) | 0.087 / 0.089 | -0.001 [-0.005, +0.003] | 0.688 | -0.002 [-0.006, +0.003] | 3.0% / 3.2% | -0.2 | 0.0% / 0.0% | -0.002 / -0.003 | +0.002 [-0.002, +0.005] |
| Arm A + diffusion | 15.6% | 700 | 656 (6.3%) | 0.401 / 0.095 | +0.306 [+0.285, +0.328] | <0.001 | +0.254 [+0.238, +0.270] | 43.1% / 3.1% | 5.5 | 2.9% / 0.1% | +0.406 / +0.377 | +0.171 [+0.155, +0.187] |
| Arm A + diffusion · null_shuffled | 115.4% | 3798 | 3762 (0.9%) | 0.089 / 0.088 | +0.001 [-0.004, +0.005] | 0.355 | -0.001 [-0.005, +0.004] | 3.0% / 3.3% | -0.2 | 0.0% / 0.0% | -0.002 / +0.000 | +0.004 [+0.000, +0.008] |
| Arm C: contrastive (Aero L14) + Aero | 29.2% | 1231 | 1223 (0.6%) | 0.227 / 0.092 | +0.135 [+0.121, +0.149] | <0.001 | +0.124 [+0.112, +0.136] | 16.4% / 2.4% | 3.6 | 0.7% / 0.0% | +0.215 / +0.158 | +0.078 [+0.068, +0.088] |
| Arm C: contrastive (Aero L14) + Aero · null_shuffled | 115.0% | 3821 | 3813 (0.2%) | 0.088 / 0.089 | -0.001 [-0.005, +0.003] | 0.709 | -0.002 [-0.007, +0.002] | 3.2% / 3.3% | -0.1 | 0.0% / 0.0% | +0.000 / -0.006 | +0.002 [-0.001, +0.006] |
| Arm C + diffusion | 22.7% | 997 | 969 (2.8%) | 0.308 / 0.095 | +0.213 [+0.197, +0.228] | <0.001 | +0.188 [+0.175, +0.202] | 27.6% / 2.8% | 5.0 | 1.1% / 0.0% | +0.324 / +0.269 | +0.116 [+0.105, +0.126] |
| Arm C + diffusion · null_shuffled | 114.8% | 3776 | 3752 (0.6%) | 0.089 / 0.089 | +0.000 [-0.004, +0.005] | 0.468 | -0.002 [-0.006, +0.003] | 3.2% / 3.3% | -0.1 | 0.0% / 0.0% | -0.001 / -0.002 | +0.003 [-0.001, +0.007] |

## Language side (all pairs)

| Run | spelling excess | meaning (sem) obs / null | sem excess | p | related obs / null | commoner obs / null | context fit (ctx, nats) excess | ctx p | fits context better than ref obs / null | ctx coverage | median zipf: correct / substituted ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | +0.160 [+0.147, +0.173] | 0.143 / 0.059 | +0.084 [+0.074, +0.094] | <0.001 | 10.0% / 3.2% | 68% / 64% | +1.90 [+1.71, +2.10] | <0.001 | 16% / 8% | 97% | 6.38 / 4.96 |
| Fast-dLLM v2 (diffusion) · null_shuffled | -0.003 [-0.007, +0.001] | 0.139 / 0.130 | +0.010 [+0.006, +0.014] | <0.001 | 8.5% / 7.8% | 54% / 53% | +0.19 [+0.11, +0.27] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Aero | +0.105 [+0.093, +0.117] | 0.129 / 0.067 | +0.062 [+0.052, +0.071] | <0.001 | 9.1% / 3.9% | 60% / 58% | +1.82 [+1.60, +2.05] | <0.001 | 15% / 8% | 95% | 6.41 / 5.19 |
| Aero · null_shuffled | -0.003 [-0.007, +0.001] | 0.135 / 0.127 | +0.007 [+0.003, +0.011] | <0.001 | 8.1% / 7.7% | 53% / 52% | +0.17 [+0.08, +0.26] | <0.001 | 5% / 5% | 96% | 7.09 / 6.00 |
| Qwen base, no prompt | +0.060 [+0.051, +0.070] | 0.129 / 0.077 | +0.052 [+0.044, +0.060] | <0.001 | 10.1% / 4.1% | 58% / 58% | +1.57 [+1.39, +1.75] | <0.001 | 13% / 8% | 97% | 6.46 / 5.37 |
| Qwen base, no prompt · null_shuffled | -0.004 [-0.008, -0.000] | 0.137 / 0.131 | +0.006 [+0.002, +0.010] | 0.002 | 7.8% / 7.8% | 54% / 53% | +0.13 [+0.05, +0.22] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Qwen Instruct | +0.039 [+0.032, +0.047] | 0.124 / 0.087 | +0.037 [+0.031, +0.043] | <0.001 | 8.5% / 5.3% | 53% / 53% | +1.20 [+1.05, +1.34] | <0.001 | 10% / 6% | 97% | 6.50 / 5.53 |
| Qwen Instruct · null_shuffled | +0.002 [-0.003, +0.006] | 0.134 / 0.126 | +0.008 [+0.004, +0.011] | <0.001 | 8.3% / 7.8% | 52% / 51% | +0.18 [+0.10, +0.26] | <0.001 | 5% / 5% | 96% | 7.09 / 6.00 |
| Aero, random-init encoder (4a) | +0.102 [+0.090, +0.115] | 0.131 / 0.069 | +0.062 [+0.053, +0.071] | <0.001 | 9.4% / 4.3% | 58% / 58% | +1.83 [+1.63, +2.03] | <0.001 | 16% / 8% | 95% | 6.41 / 5.21 |
| Aero, random-init encoder (4a) · null_shuffled | -0.005 [-0.009, -0.001] | 0.134 / 0.126 | +0.008 [+0.004, +0.012] | <0.001 | 8.1% / 7.6% | 52% / 51% | +0.20 [+0.11, +0.29] | <0.001 | 5% / 5% | 95% | 7.09 / 6.00 |
| Qwen Instruct + diffusion | +0.174 [+0.160, +0.188] | 0.140 / 0.054 | +0.086 [+0.075, +0.097] | <0.001 | 9.9% / 2.8% | 68% / 65% | +2.02 [+1.81, +2.24] | <0.001 | 15% / 9% | 97% | 6.38 / 4.92 |
| Qwen Instruct + diffusion · null_shuffled | -0.004 [-0.008, +0.000] | 0.139 / 0.130 | +0.009 [+0.005, +0.013] | <0.001 | 8.7% / 7.9% | 54% / 52% | +0.20 [+0.12, +0.28] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Aero + diffusion | +0.173 [+0.159, +0.189] | 0.145 / 0.054 | +0.090 [+0.079, +0.102] | <0.001 | 9.6% / 2.5% | 66% / 64% | +2.22 [+2.00, +2.44] | <0.001 | 16% / 9% | 96% | 6.35 / 4.93 |
| Aero + diffusion · null_shuffled | -0.004 [-0.008, +0.000] | 0.137 / 0.129 | +0.008 [+0.004, +0.012] | <0.001 | 8.9% / 7.9% | 54% / 52% | +0.20 [+0.12, +0.29] | <0.001 | 6% / 5% | 96% | 7.09 / 6.01 |
| Arm A: phoneme CTC encoder + Aero | +0.248 [+0.228, +0.267] | 0.142 / 0.053 | +0.089 [+0.075, +0.102] | <0.001 | 10.9% / 3.4% | 59% / 58% | +2.20 [+1.93, +2.46] | <0.001 | 17% / 9% | 93% | 6.31 / 4.88 |
| Arm A: phoneme CTC encoder + Aero · null_shuffled | -0.003 [-0.007, +0.002] | 0.132 / 0.125 | +0.007 [+0.003, +0.011] | <0.001 | 8.5% / 7.8% | 53% / 51% | +0.14 [+0.06, +0.23] | <0.001 | 5% / 5% | 95% | 7.09 / 6.01 |
| Arm A + diffusion | +0.252 [+0.234, +0.271] | 0.132 / 0.050 | +0.082 [+0.069, +0.095] | <0.001 | 10.2% / 2.8% | 61% / 57% | +1.82 [+1.55, +2.09] | <0.001 | 14% / 9% | 96% | 6.31 / 4.83 |
| Arm A + diffusion · null_shuffled | -0.002 [-0.007, +0.002] | 0.133 / 0.126 | +0.007 [+0.003, +0.011] | <0.001 | 8.7% / 7.7% | 52% / 51% | +0.14 [+0.06, +0.23] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Arm C: contrastive (Aero L14) + Aero | +0.113 [+0.100, +0.126] | 0.126 / 0.062 | +0.064 [+0.055, +0.074] | <0.001 | 9.6% / 3.5% | 61% / 60% | +2.01 [+1.80, +2.24] | <0.001 | 17% / 9% | 97% | 6.38 / 5.10 |
| Arm C: contrastive (Aero L14) + Aero · null_shuffled | -0.004 [-0.008, +0.000] | 0.135 / 0.128 | +0.007 [+0.003, +0.011] | <0.001 | 8.3% / 7.8% | 53% / 52% | +0.16 [+0.07, +0.25] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |
| Arm C + diffusion | +0.176 [+0.162, +0.191] | 0.145 / 0.054 | +0.090 [+0.079, +0.102] | <0.001 | 11.0% / 2.8% | 67% / 64% | +2.00 [+1.79, +2.23] | <0.001 | 16% / 9% | 96% | 6.37 / 4.95 |
| Arm C + diffusion · null_shuffled | -0.003 [-0.007, +0.001] | 0.138 / 0.130 | +0.008 [+0.004, +0.012] | <0.001 | 9.0% / 7.9% | 53% / 53% | +0.20 [+0.11, +0.29] | <0.001 | 5% / 5% | 96% | 7.09 / 6.01 |

## Sound-alike (phon >= 0.5) x related in meaning (sem >= 0.4), share of pairs

| Run | alike + related | alike only | related only | neither |
|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | 4.2% | 19.4% | 5.8% | 70.6% |
| Fast-dLLM v2 (diffusion) · null_shuffled | 1.5% | 1.5% | 7.0% | 90.0% |
| Aero | 3.5% | 12.6% | 5.6% | 78.3% |
| Aero · null_shuffled | 1.4% | 1.8% | 6.6% | 90.1% |
| Qwen base, no prompt | 3.1% | 7.2% | 6.9% | 82.7% |
| Qwen base, no prompt · null_shuffled | 1.3% | 1.9% | 6.5% | 90.3% |
| Qwen Instruct | 2.1% | 4.7% | 6.4% | 86.7% |
| Qwen Instruct · null_shuffled | 1.5% | 2.0% | 6.9% | 89.6% |
| Aero, random-init encoder (4a) | 4.0% | 12.1% | 5.3% | 78.5% |
| Aero, random-init encoder (4a) · null_shuffled | 1.2% | 1.9% | 6.8% | 90.0% |
| Qwen Instruct + diffusion | 5.2% | 20.3% | 4.7% | 69.8% |
| Qwen Instruct + diffusion · null_shuffled | 1.5% | 1.7% | 7.2% | 89.6% |
| Aero + diffusion | 3.9% | 22.0% | 5.7% | 68.4% |
| Aero + diffusion · null_shuffled | 1.6% | 1.6% | 7.3% | 89.4% |
| Arm A: phoneme CTC encoder + Aero | 6.2% | 35.7% | 4.7% | 53.4% |
| Arm A: phoneme CTC encoder + Aero · null_shuffled | 1.4% | 1.6% | 7.1% | 89.9% |
| Arm A + diffusion | 5.6% | 37.5% | 4.6% | 52.3% |
| Arm A + diffusion · null_shuffled | 1.4% | 1.6% | 7.2% | 89.7% |
| Arm C: contrastive (Aero L14) + Aero | 3.1% | 13.2% | 6.5% | 77.2% |
| Arm C: contrastive (Aero L14) + Aero · null_shuffled | 1.5% | 1.7% | 6.8% | 90.0% |
| Arm C + diffusion | 5.7% | 21.9% | 5.4% | 67.1% |
| Arm C + diffusion · null_shuffled | 1.4% | 1.7% | 7.5% | 89.3% |

## Slices (excess with 95% CI; n pairs)

| Run | slice | n | phon | phon beyond spelling | sem | ctx (nats) |
|---|---|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | seen | 72 | +0.164 [+0.118, +0.218] | +0.095 [+0.062, +0.134] | +0.104 [+0.070, +0.147] | +1.97 [+1.14, +2.98] |
| Fast-dLLM v2 (diffusion) | unseen | 1021 | +0.194 [+0.180, +0.209] | +0.108 [+0.098, +0.118] | +0.082 [+0.072, +0.092] | +1.90 [+1.70, +2.11] |
| Fast-dLLM v2 (diffusion) | mostly_right | 739 | +0.206 [+0.188, +0.224] | +0.114 [+0.102, +0.127] | +0.094 [+0.082, +0.106] | +2.28 [+2.02, +2.53] |
| Fast-dLLM v2 (diffusion) | mostly_wrong | 354 | +0.164 [+0.140, +0.187] | +0.092 [+0.076, +0.108] | +0.062 [+0.049, +0.077] | +1.12 [+0.83, +1.41] |
| Fast-dLLM v2 (diffusion) | content | 756 | +0.184 [+0.169, +0.199] | +0.105 [+0.094, +0.116] | +0.074 [+0.063, +0.085] | +1.84 [+1.61, +2.09] |
| Fast-dLLM v2 (diffusion) | no_stem | 1070 | +0.184 [+0.171, +0.198] | +0.104 [+0.095, +0.114] | +0.079 [+0.070, +0.088] | +1.87 [+1.68, +2.08] |
| Fast-dLLM v2 (diffusion) | not_alike | 835 | +0.089 [+0.079, +0.098] | +0.046 [+0.038, +0.053] | +0.074 [+0.064, +0.084] | +1.85 [+1.62, +2.08] |
| Aero | seen | 90 | +0.081 [+0.042, +0.128] | +0.049 [+0.012, +0.093] | +0.039 [+0.011, +0.072] | +1.22 [+0.03, +2.71] |
| Aero | unseen | 1281 | +0.120 [+0.108, +0.134] | +0.064 [+0.055, +0.074] | +0.063 [+0.053, +0.073] | +1.86 [+1.65, +2.09] |
| Aero | mostly_right | 512 | +0.175 [+0.154, +0.196] | +0.094 [+0.080, +0.109] | +0.099 [+0.083, +0.116] | +3.16 [+2.85, +3.49] |
| Aero | mostly_wrong | 859 | +0.084 [+0.070, +0.099] | +0.045 [+0.034, +0.056] | +0.039 [+0.029, +0.050] | +1.03 [+0.78, +1.28] |
| Aero | content | 787 | +0.127 [+0.112, +0.143] | +0.067 [+0.056, +0.079] | +0.066 [+0.053, +0.078] | +2.04 [+1.78, +2.31] |
| Aero | no_stem | 1355 | +0.112 [+0.100, +0.125] | +0.061 [+0.052, +0.070] | +0.057 [+0.048, +0.066] | +1.79 [+1.57, +2.02] |
| Aero | not_alike | 1150 | +0.044 [+0.035, +0.052] | +0.017 [+0.010, +0.025] | +0.048 [+0.039, +0.057] | +1.68 [+1.46, +1.92] |
| Qwen base, no prompt | seen | 95 | +0.033 [+0.002, +0.065] | +0.014 [-0.003, +0.034] | +0.051 [+0.022, +0.089] | +1.70 [+1.05, +2.49] |
| Qwen base, no prompt | unseen | 1536 | +0.077 [+0.067, +0.087] | +0.045 [+0.037, +0.052] | +0.052 [+0.044, +0.060] | +1.57 [+1.38, +1.76] |
| Qwen base, no prompt | mostly_right | 487 | +0.107 [+0.089, +0.126] | +0.060 [+0.047, +0.073] | +0.079 [+0.065, +0.094] | +2.82 [+2.52, +3.12] |
| Qwen base, no prompt | mostly_wrong | 1144 | +0.060 [+0.048, +0.072] | +0.036 [+0.027, +0.045] | +0.040 [+0.031, +0.050] | +1.04 [+0.84, +1.24] |
| Qwen base, no prompt | content | 863 | +0.079 [+0.067, +0.091] | +0.044 [+0.035, +0.053] | +0.059 [+0.048, +0.071] | +1.79 [+1.55, +2.03] |
| Qwen base, no prompt | no_stem | 1615 | +0.070 [+0.060, +0.080] | +0.041 [+0.034, +0.048] | +0.049 [+0.041, +0.056] | +1.56 [+1.38, +1.74] |
| Qwen base, no prompt | not_alike | 1462 | +0.026 [+0.019, +0.032] | +0.015 [+0.009, +0.020] | +0.040 [+0.032, +0.047] | +1.46 [+1.27, +1.64] |
| Qwen Instruct | seen | 108 | +0.042 [+0.014, +0.075] | +0.028 [+0.006, +0.052] | +0.052 [+0.026, +0.089] | +0.92 [+0.32, +1.70] |
| Qwen Instruct | unseen | 1988 | +0.048 [+0.040, +0.056] | +0.026 [+0.020, +0.033] | +0.037 [+0.030, +0.043] | +1.21 [+1.07, +1.36] |
| Qwen Instruct | mostly_right | 361 | +0.069 [+0.052, +0.088] | +0.037 [+0.023, +0.050] | +0.087 [+0.070, +0.104] | +3.20 [+2.82, +3.59] |
| Qwen Instruct | mostly_wrong | 1735 | +0.043 [+0.035, +0.052] | +0.024 [+0.018, +0.031] | +0.027 [+0.021, +0.033] | +0.77 [+0.64, +0.91] |
| Qwen Instruct | content | 949 | +0.046 [+0.036, +0.056] | +0.023 [+0.016, +0.031] | +0.050 [+0.040, +0.060] | +1.49 [+1.28, +1.71] |
| Qwen Instruct | no_stem | 2077 | +0.043 [+0.036, +0.051] | +0.025 [+0.019, +0.031] | +0.034 [+0.028, +0.040] | +1.18 [+1.04, +1.32] |
| Qwen Instruct | not_alike | 1952 | +0.016 [+0.010, +0.021] | +0.008 [+0.004, +0.013] | +0.030 [+0.024, +0.036] | +1.12 [+0.98, +1.27] |
| Aero, random-init encoder (4a) | seen | 55 | +0.118 [+0.052, +0.204] | +0.075 [+0.034, +0.131] | +0.078 [+0.027, +0.165] | +1.53 [+0.63, +2.40] |
| Aero, random-init encoder (4a) | unseen | 1313 | +0.124 [+0.110, +0.138] | +0.068 [+0.059, +0.077] | +0.062 [+0.052, +0.071] | +1.84 [+1.64, +2.05] |
| Aero, random-init encoder (4a) | mostly_right | 453 | +0.175 [+0.153, +0.197] | +0.095 [+0.078, +0.112] | +0.106 [+0.089, +0.123] | +3.29 [+2.95, +3.65] |
| Aero, random-init encoder (4a) | mostly_wrong | 915 | +0.098 [+0.082, +0.115] | +0.055 [+0.045, +0.065] | +0.041 [+0.031, +0.051] | +1.11 [+0.91, +1.32] |
| Aero, random-init encoder (4a) | content | 802 | +0.126 [+0.112, +0.140] | +0.067 [+0.057, +0.077] | +0.071 [+0.059, +0.083] | +2.02 [+1.76, +2.28] |
| Aero, random-init encoder (4a) | no_stem | 1350 | +0.118 [+0.105, +0.131] | +0.066 [+0.057, +0.075] | +0.059 [+0.050, +0.068] | +1.80 [+1.61, +2.00] |
| Aero, random-init encoder (4a) | not_alike | 1147 | +0.050 [+0.041, +0.060] | +0.026 [+0.019, +0.033] | +0.049 [+0.040, +0.058] | +1.70 [+1.50, +1.91] |
| Qwen Instruct + diffusion | seen | 71 | +0.162 [+0.110, +0.226] | +0.096 [+0.064, +0.136] | +0.094 [+0.052, +0.143] | +2.86 [+1.84, +4.00] |
| Qwen Instruct + diffusion | unseen | 958 | +0.206 [+0.191, +0.222] | +0.114 [+0.103, +0.124] | +0.085 [+0.074, +0.096] | +1.96 [+1.75, +2.19] |
| Qwen Instruct + diffusion | mostly_right | 688 | +0.220 [+0.200, +0.239] | +0.123 [+0.110, +0.136] | +0.098 [+0.085, +0.112] | +2.50 [+2.26, +2.75] |
| Qwen Instruct + diffusion | mostly_wrong | 341 | +0.170 [+0.151, +0.189] | +0.091 [+0.077, +0.106] | +0.062 [+0.046, +0.079] | +1.07 [+0.70, +1.44] |
| Qwen Instruct + diffusion | content | 733 | +0.200 [+0.183, +0.217] | +0.110 [+0.099, +0.122] | +0.084 [+0.071, +0.097] | +2.05 [+1.80, +2.31] |
| Qwen Instruct + diffusion | no_stem | 1005 | +0.194 [+0.180, +0.209] | +0.109 [+0.099, +0.119] | +0.080 [+0.070, +0.091] | +2.00 [+1.79, +2.21] |
| Qwen Instruct + diffusion | not_alike | 766 | +0.093 [+0.083, +0.102] | +0.048 [+0.040, +0.056] | +0.068 [+0.057, +0.079] | +1.94 [+1.70, +2.18] |
| Aero + diffusion | seen | 47 | +0.151 [+0.082, +0.233] | +0.078 [+0.030, +0.132] | +0.104 [+0.055, +0.156] | +3.09 [+2.02, +4.30] |
| Aero + diffusion | unseen | 883 | +0.203 [+0.187, +0.219] | +0.108 [+0.097, +0.120] | +0.090 [+0.079, +0.101] | +2.17 [+1.95, +2.40] |
| Aero + diffusion | mostly_right | 658 | +0.216 [+0.197, +0.234] | +0.114 [+0.101, +0.127] | +0.097 [+0.084, +0.111] | +2.60 [+2.35, +2.86] |
| Aero + diffusion | mostly_wrong | 272 | +0.164 [+0.134, +0.194] | +0.088 [+0.068, +0.111] | +0.074 [+0.057, +0.092] | +1.25 [+0.89, +1.65] |
| Aero + diffusion | content | 649 | +0.203 [+0.186, +0.222] | +0.110 [+0.098, +0.123] | +0.086 [+0.073, +0.099] | +2.21 [+1.96, +2.46] |
| Aero + diffusion | no_stem | 905 | +0.192 [+0.177, +0.208] | +0.104 [+0.093, +0.115] | +0.084 [+0.074, +0.095] | +2.19 [+1.98, +2.42] |
| Aero + diffusion | not_alike | 689 | +0.090 [+0.080, +0.101] | +0.043 [+0.033, +0.052] | +0.077 [+0.065, +0.089] | +2.16 [+1.92, +2.41] |
| Arm A: phoneme CTC encoder + Aero | seen | 68 | +0.245 [+0.196, +0.293] | +0.139 [+0.104, +0.170] | +0.064 [+0.034, +0.097] | +2.66 [+1.73, +3.88] |
| Arm A: phoneme CTC encoder + Aero | unseen | 595 | +0.299 [+0.275, +0.322] | +0.159 [+0.142, +0.176] | +0.091 [+0.076, +0.106] | +2.14 [+1.88, +2.41] |
| Arm A: phoneme CTC encoder + Aero | mostly_right | 456 | +0.301 [+0.278, +0.326] | +0.161 [+0.143, +0.179] | +0.104 [+0.087, +0.120] | +2.65 [+2.32, +2.98] |
| Arm A: phoneme CTC encoder + Aero | mostly_wrong | 207 | +0.275 [+0.229, +0.320] | +0.147 [+0.119, +0.178] | +0.055 [+0.031, +0.079] | +1.19 [+0.76, +1.61] |
| Arm A: phoneme CTC encoder + Aero | content | 488 | +0.288 [+0.266, +0.310] | +0.149 [+0.133, +0.164] | +0.090 [+0.072, +0.107] | +2.26 [+1.94, +2.60] |
| Arm A: phoneme CTC encoder + Aero | no_stem | 631 | +0.280 [+0.258, +0.302] | +0.154 [+0.138, +0.170] | +0.072 [+0.060, +0.085] | +2.06 [+1.79, +2.33] |
| Arm A: phoneme CTC encoder + Aero | not_alike | 385 | +0.118 [+0.104, +0.132] | +0.054 [+0.042, +0.066] | +0.064 [+0.048, +0.080] | +2.31 [+1.98, +2.64] |
| Arm A + diffusion | seen | 68 | +0.294 [+0.220, +0.364] | +0.166 [+0.113, +0.221] | +0.069 [+0.037, +0.100] | +2.63 [+1.66, +3.74] |
| Arm A + diffusion | unseen | 588 | +0.308 [+0.285, +0.331] | +0.171 [+0.155, +0.188] | +0.083 [+0.069, +0.098] | +1.72 [+1.46, +2.00] |
| Arm A + diffusion | mostly_right | 517 | +0.308 [+0.285, +0.332] | +0.173 [+0.156, +0.191] | +0.089 [+0.075, +0.104] | +2.14 [+1.85, +2.45] |
| Arm A + diffusion | mostly_wrong | 139 | +0.299 [+0.242, +0.356] | +0.162 [+0.125, +0.199] | +0.055 [+0.023, +0.087] | +0.59 [+0.11, +1.07] |
| Arm A + diffusion | content | 482 | +0.305 [+0.282, +0.327] | +0.167 [+0.151, +0.183] | +0.082 [+0.066, +0.099] | +1.75 [+1.43, +2.08] |
| Arm A + diffusion | no_stem | 628 | +0.296 [+0.274, +0.318] | +0.168 [+0.152, +0.184] | +0.068 [+0.056, +0.081] | +1.74 [+1.48, +2.01] |
| Arm A + diffusion | not_alike | 373 | +0.126 [+0.112, +0.140] | +0.065 [+0.052, +0.078] | +0.064 [+0.048, +0.081] | +1.82 [+1.50, +2.16] |
| Arm C: contrastive (Aero L14) + Aero | seen | 49 | +0.088 [+0.037, +0.146] | +0.024 [-0.013, +0.070] | +0.071 [+0.024, +0.128] | +2.23 [+0.82, +3.88] |
| Arm C: contrastive (Aero L14) + Aero | unseen | 1174 | +0.137 [+0.123, +0.151] | +0.080 [+0.070, +0.090] | +0.064 [+0.055, +0.074] | +2.00 [+1.79, +2.23] |
| Arm C: contrastive (Aero L14) + Aero | mostly_right | 525 | +0.186 [+0.167, +0.206] | +0.106 [+0.091, +0.120] | +0.094 [+0.079, +0.109] | +3.27 [+2.99, +3.58] |
| Arm C: contrastive (Aero L14) + Aero | mostly_wrong | 698 | +0.097 [+0.079, +0.115] | +0.057 [+0.044, +0.069] | +0.042 [+0.032, +0.053] | +1.07 [+0.82, +1.33] |
| Arm C: contrastive (Aero L14) + Aero | content | 764 | +0.146 [+0.130, +0.162] | +0.082 [+0.071, +0.093] | +0.068 [+0.057, +0.080] | +2.30 [+2.04, +2.57] |
| Arm C: contrastive (Aero L14) + Aero | no_stem | 1205 | +0.130 [+0.116, +0.144] | +0.075 [+0.065, +0.085] | +0.062 [+0.053, +0.071] | +2.00 [+1.79, +2.23] |
| Arm C: contrastive (Aero L14) + Aero | not_alike | 1023 | +0.061 [+0.052, +0.071] | +0.032 [+0.025, +0.040] | +0.051 [+0.041, +0.060] | +1.94 [+1.71, +2.18] |
| Arm C + diffusion | seen | 44 | +0.213 [+0.143, +0.288] | +0.125 [+0.081, +0.179] | +0.129 [+0.068, +0.212] | +2.88 [+1.97, +4.14] |
| Arm C + diffusion | unseen | 925 | +0.213 [+0.197, +0.229] | +0.115 [+0.104, +0.126] | +0.088 [+0.078, +0.100] | +1.96 [+1.74, +2.19] |
| Arm C + diffusion | mostly_right | 639 | +0.238 [+0.220, +0.257] | +0.132 [+0.119, +0.145] | +0.101 [+0.088, +0.115] | +2.36 [+2.09, +2.63] |
| Arm C + diffusion | mostly_wrong | 330 | +0.164 [+0.141, +0.187] | +0.084 [+0.068, +0.100] | +0.069 [+0.050, +0.089] | +1.31 [+0.95, +1.68] |
| Arm C + diffusion | content | 678 | +0.210 [+0.193, +0.227] | +0.118 [+0.106, +0.130] | +0.077 [+0.064, +0.091] | +1.93 [+1.68, +2.19] |
| Arm C + diffusion | no_stem | 941 | +0.204 [+0.189, +0.220] | +0.114 [+0.103, +0.125] | +0.083 [+0.073, +0.094] | +1.98 [+1.76, +2.20] |
| Arm C + diffusion | not_alike | 702 | +0.100 [+0.089, +0.111] | +0.050 [+0.041, +0.059] | +0.071 [+0.060, +0.082] | +1.90 [+1.65, +2.16] |

## Slice differences within a run (a − b, 95% CI)

| Run | metric | a − b | diff |
|---|---|---|---|
| Fast-dLLM v2 (diffusion) | phon | seen − unseen | -0.030 [-0.078, +0.025] |
| Fast-dLLM v2 (diffusion) | phon | mostly_right − mostly_wrong | +0.042 [+0.013, +0.071] |
| Fast-dLLM v2 (diffusion) | sem | seen − unseen | +0.022 [-0.014, +0.065] |
| Fast-dLLM v2 (diffusion) | sem | mostly_right − mostly_wrong | +0.031 [+0.013, +0.050] |
| Fast-dLLM v2 (diffusion) | ctx | seen − unseen | +0.076 [-0.801, +1.118] |
| Fast-dLLM v2 (diffusion) | ctx | mostly_right − mostly_wrong | +1.161 [+0.775, +1.544] |
| Aero | phon | seen − unseen | -0.039 [-0.081, +0.009] |
| Aero | phon | mostly_right − mostly_wrong | +0.091 [+0.066, +0.116] |
| Aero | sem | seen − unseen | -0.024 [-0.054, +0.011] |
| Aero | sem | mostly_right − mostly_wrong | +0.060 [+0.040, +0.080] |
| Aero | ctx | seen − unseen | -0.642 [-1.846, +0.866] |
| Aero | ctx | mostly_right − mostly_wrong | +2.136 [+1.737, +2.545] |
| Qwen base, no prompt | phon | seen − unseen | -0.044 [-0.076, -0.011] |
| Qwen base, no prompt | phon | mostly_right − mostly_wrong | +0.047 [+0.025, +0.069] |
| Qwen base, no prompt | sem | seen − unseen | -0.001 [-0.031, +0.038] |
| Qwen base, no prompt | sem | mostly_right − mostly_wrong | +0.039 [+0.023, +0.056] |
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
| Aero, random-init encoder (4a) | sem | seen − unseen | +0.016 [-0.035, +0.103] |
| Aero, random-init encoder (4a) | sem | mostly_right − mostly_wrong | +0.065 [+0.046, +0.084] |
| Aero, random-init encoder (4a) | ctx | seen − unseen | -0.309 [-1.241, +0.566] |
| Aero, random-init encoder (4a) | ctx | mostly_right − mostly_wrong | +2.182 [+1.792, +2.598] |
| Qwen Instruct + diffusion | phon | seen − unseen | -0.045 [-0.098, +0.022] |
| Qwen Instruct + diffusion | phon | mostly_right − mostly_wrong | +0.050 [+0.023, +0.077] |
| Qwen Instruct + diffusion | sem | seen − unseen | +0.008 [-0.035, +0.059] |
| Qwen Instruct + diffusion | sem | mostly_right − mostly_wrong | +0.036 [+0.015, +0.057] |
| Qwen Instruct + diffusion | ctx | seen − unseen | +0.894 [-0.166, +2.063] |
| Qwen Instruct + diffusion | ctx | mostly_right − mostly_wrong | +1.430 [+0.984, +1.872] |
| Aero + diffusion | phon | seen − unseen | -0.052 [-0.123, +0.030] |
| Aero + diffusion | phon | mostly_right − mostly_wrong | +0.052 [+0.016, +0.087] |
| Aero + diffusion | sem | seen − unseen | +0.014 [-0.036, +0.067] |
| Aero + diffusion | sem | mostly_right − mostly_wrong | +0.024 [+0.001, +0.046] |
| Aero + diffusion | ctx | seen − unseen | +0.923 [-0.155, +2.155] |
| Aero + diffusion | ctx | mostly_right − mostly_wrong | +1.347 [+0.877, +1.799] |
| Arm A: phoneme CTC encoder + Aero | phon | seen − unseen | -0.053 [-0.107, -0.001] |
| Arm A: phoneme CTC encoder + Aero | phon | mostly_right − mostly_wrong | +0.027 [-0.024, +0.079] |
| Arm A: phoneme CTC encoder + Aero | sem | seen − unseen | -0.027 [-0.061, +0.010] |
| Arm A: phoneme CTC encoder + Aero | sem | mostly_right − mostly_wrong | +0.049 [+0.019, +0.078] |
| Arm A: phoneme CTC encoder + Aero | ctx | seen − unseen | +0.517 [-0.447, +1.774] |
| Arm A: phoneme CTC encoder + Aero | ctx | mostly_right − mostly_wrong | +1.461 [+0.924, +2.005] |
| Arm A + diffusion | phon | seen − unseen | -0.013 [-0.090, +0.060] |
| Arm A + diffusion | phon | mostly_right − mostly_wrong | +0.009 [-0.052, +0.071] |
| Arm A + diffusion | sem | seen − unseen | -0.015 [-0.049, +0.019] |
| Arm A + diffusion | sem | mostly_right − mostly_wrong | +0.034 [-0.002, +0.069] |
| Arm A + diffusion | ctx | seen − unseen | +0.901 [-0.120, +2.026] |
| Arm A + diffusion | ctx | mostly_right − mostly_wrong | +1.546 [+1.012, +2.123] |
| Arm C: contrastive (Aero L14) + Aero | phon | seen − unseen | -0.050 [-0.102, +0.010] |
| Arm C: contrastive (Aero L14) + Aero | phon | mostly_right − mostly_wrong | +0.089 [+0.063, +0.116] |
| Arm C: contrastive (Aero L14) + Aero | sem | seen − unseen | +0.007 [-0.040, +0.064] |
| Arm C: contrastive (Aero L14) + Aero | sem | mostly_right − mostly_wrong | +0.052 [+0.034, +0.070] |
| Arm C: contrastive (Aero L14) + Aero | ctx | seen − unseen | +0.221 [-1.196, +1.885] |
| Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right − mostly_wrong | +2.205 [+1.819, +2.591] |
| Arm C + diffusion | phon | seen − unseen | -0.000 [-0.072, +0.076] |
| Arm C + diffusion | phon | mostly_right − mostly_wrong | +0.074 [+0.045, +0.103] |
| Arm C + diffusion | sem | seen − unseen | +0.041 [-0.021, +0.124] |
| Arm C + diffusion | sem | mostly_right − mostly_wrong | +0.032 [+0.009, +0.056] |
| Arm C + diffusion | ctx | seen − unseen | +0.917 [-0.031, +2.184] |
| Arm C + diffusion | ctx | mostly_right − mostly_wrong | +1.048 [+0.587, +1.503] |

## Run contrasts (difference in mean excess, a − b; paired trial bootstrap)

| a | b | metric | slice | diff |
|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | Aero | phon | all | +0.074 [+0.058, +0.092] |
| Fast-dLLM v2 (diffusion) | Aero | phon | mostly_right | +0.031 [+0.006, +0.056] |
| Fast-dLLM v2 (diffusion) | Aero | phon|orth | all | +0.044 [+0.032, +0.057] |
| Fast-dLLM v2 (diffusion) | Aero | phon|orth | mostly_right | +0.020 [+0.002, +0.039] |
| Fast-dLLM v2 (diffusion) | Aero | sem | all | +0.022 [+0.010, +0.034] |
| Fast-dLLM v2 (diffusion) | Aero | sem | mostly_right | -0.005 [-0.024, +0.013] |
| Fast-dLLM v2 (diffusion) | Aero | ctx | all | +0.084 [-0.170, +0.343] |
| Fast-dLLM v2 (diffusion) | Aero | ctx | mostly_right | -0.883 [-1.257, -0.525] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon | all | +0.118 [+0.102, +0.134] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon | mostly_right | +0.098 [+0.074, +0.123] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon|orth | all | +0.064 [+0.053, +0.076] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | phon|orth | mostly_right | +0.054 [+0.037, +0.071] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | sem | all | +0.032 [+0.021, +0.043] |
| Fast-dLLM v2 (diffusion) | Qwen base, no prompt | sem | mostly_right | +0.014 [-0.003, +0.031] |
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
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | phon | all | -0.011 [-0.028, +0.006] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | phon | mostly_right | -0.014 [-0.037, +0.009] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | phon|orth | all | -0.005 [-0.017, +0.007] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | phon|orth | mostly_right | -0.008 [-0.025, +0.008] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | sem | all | -0.002 [-0.014, +0.010] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | sem | mostly_right | -0.004 [-0.019, +0.010] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | ctx | all | -0.120 [-0.347, +0.102] |
| Fast-dLLM v2 (diffusion) | Qwen Instruct + diffusion | ctx | mostly_right | -0.223 [-0.495, +0.039] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | phon | all | -0.008 [-0.026, +0.009] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | phon | mostly_right | -0.010 [-0.032, +0.012] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | phon|orth | all | +0.000 [-0.012, +0.013] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | phon|orth | mostly_right | +0.000 [-0.015, +0.016] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | sem | all | -0.007 [-0.019, +0.006] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | sem | mostly_right | -0.004 [-0.019, +0.011] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | ctx | all | -0.311 [-0.551, -0.070] |
| Fast-dLLM v2 (diffusion) | Aero + diffusion | ctx | mostly_right | -0.323 [-0.622, -0.024] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | phon | all | -0.101 [-0.125, -0.077] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.096 [-0.125, -0.068] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.050 [-0.067, -0.032] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.046 [-0.067, -0.025] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | sem | all | -0.005 [-0.021, +0.011] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | -0.010 [-0.029, +0.009] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | ctx | all | -0.293 [-0.593, +0.006] |
| Fast-dLLM v2 (diffusion) | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | -0.372 [-0.754, -0.009] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | phon | all | -0.114 [-0.137, -0.091] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | phon | mostly_right | -0.102 [-0.128, -0.076] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | phon|orth | all | -0.064 [-0.080, -0.047] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | phon|orth | mostly_right | -0.059 [-0.078, -0.039] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | sem | all | +0.002 [-0.014, +0.017] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | sem | mostly_right | +0.005 [-0.013, +0.022] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | ctx | all | +0.087 [-0.211, +0.385] |
| Fast-dLLM v2 (diffusion) | Arm A + diffusion | ctx | mostly_right | +0.137 [-0.236, +0.502] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | phon | all | +0.057 [+0.039, +0.074] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | +0.019 [-0.005, +0.043] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | +0.030 [+0.017, +0.042] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | +0.009 [-0.009, +0.027] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | sem | all | +0.019 [+0.007, +0.031] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | -0.000 [-0.017, +0.017] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | ctx | all | -0.110 [-0.364, +0.141] |
| Fast-dLLM v2 (diffusion) | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.996 [-1.330, -0.670] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | phon | all | -0.021 [-0.038, -0.004] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | phon | mostly_right | -0.033 [-0.055, -0.010] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | phon|orth | all | -0.009 [-0.022, +0.005] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | phon|orth | mostly_right | -0.018 [-0.035, -0.001] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | sem | all | -0.007 [-0.019, +0.005] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | sem | mostly_right | -0.008 [-0.023, +0.007] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | ctx | all | -0.100 [-0.353, +0.156] |
| Fast-dLLM v2 (diffusion) | Arm C + diffusion | ctx | mostly_right | -0.082 [-0.392, +0.243] |
| Aero | Qwen base, no prompt | phon | all | +0.043 [+0.029, +0.057] |
| Aero | Qwen base, no prompt | phon | mostly_right | +0.067 [+0.043, +0.093] |
| Aero | Qwen base, no prompt | phon|orth | all | +0.020 [+0.010, +0.031] |
| Aero | Qwen base, no prompt | phon|orth | mostly_right | +0.034 [+0.017, +0.052] |
| Aero | Qwen base, no prompt | sem | all | +0.010 [-0.001, +0.020] |
| Aero | Qwen base, no prompt | sem | mostly_right | +0.020 [-0.001, +0.040] |
| Aero | Qwen base, no prompt | ctx | all | +0.247 [+0.011, +0.493] |
| Aero | Qwen base, no prompt | ctx | mostly_right | +0.345 [-0.056, +0.751] |
| Aero | Qwen Instruct | phon | all | +0.070 [+0.056, +0.084] |
| Aero | Qwen Instruct | phon | mostly_right | +0.105 [+0.078, +0.133] |
| Aero | Qwen Instruct | phon|orth | all | +0.037 [+0.026, +0.047] |
| Aero | Qwen Instruct | phon|orth | mostly_right | +0.058 [+0.038, +0.078] |
| Aero | Qwen Instruct | sem | all | +0.024 [+0.014, +0.035] |
| Aero | Qwen Instruct | sem | mostly_right | +0.012 [-0.010, +0.035] |
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
| Aero | Qwen Instruct + diffusion | phon | all | -0.086 [-0.103, -0.069] |
| Aero | Qwen Instruct + diffusion | phon | mostly_right | -0.045 [-0.071, -0.019] |
| Aero | Qwen Instruct + diffusion | phon|orth | all | -0.049 [-0.061, -0.038] |
| Aero | Qwen Instruct + diffusion | phon|orth | mostly_right | -0.029 [-0.046, -0.011] |
| Aero | Qwen Instruct + diffusion | sem | all | -0.024 [-0.037, -0.012] |
| Aero | Qwen Instruct + diffusion | sem | mostly_right | +0.001 [-0.018, +0.021] |
| Aero | Qwen Instruct + diffusion | ctx | all | -0.204 [-0.474, +0.063] |
| Aero | Qwen Instruct + diffusion | ctx | mostly_right | +0.660 [+0.316, +1.013] |
| Aero | Aero + diffusion | phon | all | -0.083 [-0.101, -0.065] |
| Aero | Aero + diffusion | phon | mostly_right | -0.041 [-0.066, -0.016] |
| Aero | Aero + diffusion | phon|orth | all | -0.044 [-0.056, -0.032] |
| Aero | Aero + diffusion | phon|orth | mostly_right | -0.020 [-0.039, -0.001] |
| Aero | Aero + diffusion | sem | all | -0.029 [-0.042, -0.016] |
| Aero | Aero + diffusion | sem | mostly_right | +0.002 [-0.017, +0.020] |
| Aero | Aero + diffusion | ctx | all | -0.395 [-0.670, -0.122] |
| Aero | Aero + diffusion | ctx | mostly_right | +0.561 [+0.209, +0.919] |
| Aero | Arm A: phoneme CTC encoder + Aero | phon | all | -0.175 [-0.200, -0.151] |
| Aero | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.127 [-0.157, -0.097] |
| Aero | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.094 [-0.111, -0.076] |
| Aero | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.067 [-0.089, -0.044] |
| Aero | Arm A: phoneme CTC encoder + Aero | sem | all | -0.027 [-0.044, -0.011] |
| Aero | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | -0.005 [-0.028, +0.019] |
| Aero | Arm A: phoneme CTC encoder + Aero | ctx | all | -0.377 [-0.685, -0.076] |
| Aero | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | +0.512 [+0.093, +0.930] |
| Aero | Arm A + diffusion | phon | all | -0.188 [-0.213, -0.164] |
| Aero | Arm A + diffusion | phon | mostly_right | -0.133 [-0.162, -0.104] |
| Aero | Arm A + diffusion | phon|orth | all | -0.108 [-0.125, -0.090] |
| Aero | Arm A + diffusion | phon|orth | mostly_right | -0.079 [-0.101, -0.058] |
| Aero | Arm A + diffusion | sem | all | -0.020 [-0.036, -0.005] |
| Aero | Arm A + diffusion | sem | mostly_right | +0.010 [-0.010, +0.031] |
| Aero | Arm A + diffusion | ctx | all | +0.003 [-0.311, +0.324] |
| Aero | Arm A + diffusion | ctx | mostly_right | +1.021 [+0.609, +1.443] |
| Aero | Arm C: contrastive (Aero L14) + Aero | phon | all | -0.018 [-0.033, -0.001] |
| Aero | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | -0.012 [-0.036, +0.013] |
| Aero | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | -0.014 [-0.026, -0.003] |
| Aero | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | -0.011 [-0.028, +0.007] |
| Aero | Arm C: contrastive (Aero L14) + Aero | sem | all | -0.003 [-0.015, +0.009] |
| Aero | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | +0.005 [-0.014, +0.025] |
| Aero | Arm C: contrastive (Aero L14) + Aero | ctx | all | -0.194 [-0.452, +0.067] |
| Aero | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.112 [-0.484, +0.275] |
| Aero | Arm C + diffusion | phon | all | -0.095 [-0.113, -0.078] |
| Aero | Arm C + diffusion | phon | mostly_right | -0.064 [-0.089, -0.038] |
| Aero | Arm C + diffusion | phon|orth | all | -0.053 [-0.065, -0.040] |
| Aero | Arm C + diffusion | phon|orth | mostly_right | -0.038 [-0.056, -0.020] |
| Aero | Arm C + diffusion | sem | all | -0.029 [-0.042, -0.015] |
| Aero | Arm C + diffusion | sem | mostly_right | -0.002 [-0.022, +0.018] |
| Aero | Arm C + diffusion | ctx | all | -0.184 [-0.460, +0.098] |
| Aero | Arm C + diffusion | ctx | mostly_right | +0.802 [+0.426, +1.190] |
| Qwen base, no prompt | Qwen Instruct | phon | all | +0.027 [+0.015, +0.038] |
| Qwen base, no prompt | Qwen Instruct | phon | mostly_right | +0.038 [+0.012, +0.062] |
| Qwen base, no prompt | Qwen Instruct | phon|orth | all | +0.017 [+0.008, +0.025] |
| Qwen base, no prompt | Qwen Instruct | phon|orth | mostly_right | +0.024 [+0.005, +0.042] |
| Qwen base, no prompt | Qwen Instruct | sem | all | +0.015 [+0.006, +0.024] |
| Qwen base, no prompt | Qwen Instruct | sem | mostly_right | -0.007 [-0.029, +0.014] |
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
| Qwen base, no prompt | Qwen Instruct + diffusion | phon | all | -0.129 [-0.145, -0.113] |
| Qwen base, no prompt | Qwen Instruct + diffusion | phon | mostly_right | -0.112 [-0.137, -0.088] |
| Qwen base, no prompt | Qwen Instruct + diffusion | phon|orth | all | -0.069 [-0.080, -0.058] |
| Qwen base, no prompt | Qwen Instruct + diffusion | phon|orth | mostly_right | -0.063 [-0.079, -0.046] |
| Qwen base, no prompt | Qwen Instruct + diffusion | sem | all | -0.034 [-0.046, -0.022] |
| Qwen base, no prompt | Qwen Instruct + diffusion | sem | mostly_right | -0.019 [-0.037, -0.000] |
| Qwen base, no prompt | Qwen Instruct + diffusion | ctx | all | -0.451 [-0.722, -0.186] |
| Qwen base, no prompt | Qwen Instruct + diffusion | ctx | mostly_right | +0.315 [-0.057, +0.689] |
| Qwen base, no prompt | Aero + diffusion | phon | all | -0.126 [-0.143, -0.109] |
| Qwen base, no prompt | Aero + diffusion | phon | mostly_right | -0.108 [-0.132, -0.085] |
| Qwen base, no prompt | Aero + diffusion | phon|orth | all | -0.064 [-0.076, -0.052] |
| Qwen base, no prompt | Aero + diffusion | phon|orth | mostly_right | -0.054 [-0.071, -0.037] |
| Qwen base, no prompt | Aero + diffusion | sem | all | -0.039 [-0.051, -0.027] |
| Qwen base, no prompt | Aero + diffusion | sem | mostly_right | -0.018 [-0.036, -0.000] |
| Qwen base, no prompt | Aero + diffusion | ctx | all | -0.642 [-0.897, -0.393] |
| Qwen base, no prompt | Aero + diffusion | ctx | mostly_right | +0.215 [-0.154, +0.594] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | phon | all | -0.219 [-0.242, -0.196] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.194 [-0.224, -0.165] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.114 [-0.131, -0.097] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.101 [-0.122, -0.079] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | sem | all | -0.037 [-0.053, -0.021] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | -0.025 [-0.046, -0.004] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | ctx | all | -0.624 [-0.925, -0.317] |
| Qwen base, no prompt | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | +0.166 [-0.239, +0.589] |
| Qwen base, no prompt | Arm A + diffusion | phon | all | -0.232 [-0.255, -0.209] |
| Qwen base, no prompt | Arm A + diffusion | phon | mostly_right | -0.201 [-0.229, -0.172] |
| Qwen base, no prompt | Arm A + diffusion | phon|orth | all | -0.128 [-0.145, -0.111] |
| Qwen base, no prompt | Arm A + diffusion | phon|orth | mostly_right | -0.113 [-0.134, -0.092] |
| Qwen base, no prompt | Arm A + diffusion | sem | all | -0.030 [-0.045, -0.015] |
| Qwen base, no prompt | Arm A + diffusion | sem | mostly_right | -0.010 [-0.029, +0.010] |
| Qwen base, no prompt | Arm A + diffusion | ctx | all | -0.243 [-0.551, +0.063] |
| Qwen base, no prompt | Arm A + diffusion | ctx | mostly_right | +0.676 [+0.261, +1.085] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | phon | all | -0.061 [-0.076, -0.046] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | -0.079 [-0.103, -0.056] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | -0.035 [-0.046, -0.023] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | -0.045 [-0.062, -0.028] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | sem | all | -0.013 [-0.024, -0.001] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | -0.014 [-0.034, +0.005] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | ctx | all | -0.440 [-0.684, -0.201] |
| Qwen base, no prompt | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.458 [-0.838, -0.078] |
| Qwen base, no prompt | Arm C + diffusion | phon | all | -0.138 [-0.155, -0.122] |
| Qwen base, no prompt | Arm C + diffusion | phon | mostly_right | -0.131 [-0.156, -0.106] |
| Qwen base, no prompt | Arm C + diffusion | phon|orth | all | -0.073 [-0.085, -0.061] |
| Qwen base, no prompt | Arm C + diffusion | phon|orth | mostly_right | -0.072 [-0.090, -0.054] |
| Qwen base, no prompt | Arm C + diffusion | sem | all | -0.038 [-0.051, -0.026] |
| Qwen base, no prompt | Arm C + diffusion | sem | mostly_right | -0.022 [-0.041, -0.003] |
| Qwen base, no prompt | Arm C + diffusion | ctx | all | -0.431 [-0.684, -0.173] |
| Qwen base, no prompt | Arm C + diffusion | ctx | mostly_right | +0.456 [+0.073, +0.851] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon | all | -0.076 [-0.091, -0.062] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon | mostly_right | -0.105 [-0.133, -0.077] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon|orth | all | -0.042 [-0.052, -0.032] |
| Qwen Instruct | Aero, random-init encoder (4a) | phon|orth | mostly_right | -0.058 [-0.080, -0.036] |
| Qwen Instruct | Aero, random-init encoder (4a) | sem | all | -0.025 [-0.035, -0.014] |
| Qwen Instruct | Aero, random-init encoder (4a) | sem | mostly_right | -0.019 [-0.041, +0.003] |
| Qwen Instruct | Aero, random-init encoder (4a) | ctx | all | -0.632 [-0.856, -0.397] |
| Qwen Instruct | Aero, random-init encoder (4a) | ctx | mostly_right | -0.093 [-0.577, +0.387] |
| Qwen Instruct | Qwen Instruct + diffusion | phon | all | -0.155 [-0.172, -0.140] |
| Qwen Instruct | Qwen Instruct + diffusion | phon | mostly_right | -0.150 [-0.177, -0.124] |
| Qwen Instruct | Qwen Instruct + diffusion | phon|orth | all | -0.086 [-0.097, -0.074] |
| Qwen Instruct | Qwen Instruct + diffusion | phon|orth | mostly_right | -0.086 [-0.105, -0.067] |
| Qwen Instruct | Qwen Instruct + diffusion | sem | all | -0.049 [-0.060, -0.037] |
| Qwen Instruct | Qwen Instruct + diffusion | sem | mostly_right | -0.011 [-0.033, +0.010] |
| Qwen Instruct | Qwen Instruct + diffusion | ctx | all | -0.829 [-1.078, -0.590] |
| Qwen Instruct | Qwen Instruct + diffusion | ctx | mostly_right | +0.700 [+0.238, +1.162] |
| Qwen Instruct | Aero + diffusion | phon | all | -0.153 [-0.170, -0.135] |
| Qwen Instruct | Aero + diffusion | phon | mostly_right | -0.146 [-0.172, -0.120] |
| Qwen Instruct | Aero + diffusion | phon|orth | all | -0.080 [-0.092, -0.068] |
| Qwen Instruct | Aero + diffusion | phon|orth | mostly_right | -0.078 [-0.096, -0.059] |
| Qwen Instruct | Aero + diffusion | sem | all | -0.053 [-0.065, -0.041] |
| Qwen Instruct | Aero + diffusion | sem | mostly_right | -0.011 [-0.032, +0.011] |
| Qwen Instruct | Aero + diffusion | ctx | all | -1.021 [-1.275, -0.775] |
| Qwen Instruct | Aero + diffusion | ctx | mostly_right | +0.601 [+0.141, +1.058] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | phon | all | -0.245 [-0.268, -0.223] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.232 [-0.261, -0.203] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.130 [-0.146, -0.114] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.124 [-0.146, -0.103] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | sem | all | -0.051 [-0.066, -0.037] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | -0.017 [-0.041, +0.006] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | ctx | all | -1.002 [-1.287, -0.711] |
| Qwen Instruct | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | +0.552 [+0.059, +1.054] |
| Qwen Instruct | Arm A + diffusion | phon | all | -0.258 [-0.281, -0.236] |
| Qwen Instruct | Arm A + diffusion | phon | mostly_right | -0.239 [-0.268, -0.209] |
| Qwen Instruct | Arm A + diffusion | phon|orth | all | -0.144 [-0.161, -0.128] |
| Qwen Instruct | Arm A + diffusion | phon|orth | mostly_right | -0.137 [-0.158, -0.114] |
| Qwen Instruct | Arm A + diffusion | sem | all | -0.044 [-0.059, -0.030] |
| Qwen Instruct | Arm A + diffusion | sem | mostly_right | -0.002 [-0.024, +0.019] |
| Qwen Instruct | Arm A + diffusion | ctx | all | -0.622 [-0.914, -0.324] |
| Qwen Instruct | Arm A + diffusion | ctx | mostly_right | +1.061 [+0.575, +1.548] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | phon | all | -0.088 [-0.102, -0.072] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | -0.117 [-0.142, -0.092] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | -0.051 [-0.062, -0.040] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | -0.069 [-0.088, -0.050] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | sem | all | -0.027 [-0.038, -0.017] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | -0.007 [-0.029, +0.015] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | ctx | all | -0.819 [-1.051, -0.589] |
| Qwen Instruct | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.072 [-0.536, +0.393] |
| Qwen Instruct | Arm C + diffusion | phon | all | -0.165 [-0.182, -0.149] |
| Qwen Instruct | Arm C + diffusion | phon | mostly_right | -0.169 [-0.195, -0.142] |
| Qwen Instruct | Arm C + diffusion | phon|orth | all | -0.089 [-0.101, -0.077] |
| Qwen Instruct | Arm C + diffusion | phon|orth | mostly_right | -0.096 [-0.114, -0.076] |
| Qwen Instruct | Arm C + diffusion | sem | all | -0.053 [-0.065, -0.041] |
| Qwen Instruct | Arm C + diffusion | sem | mostly_right | -0.015 [-0.036, +0.007] |
| Qwen Instruct | Arm C + diffusion | ctx | all | -0.809 [-1.064, -0.561] |
| Qwen Instruct | Arm C + diffusion | ctx | mostly_right | +0.842 [+0.364, +1.328] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | phon | all | -0.080 [-0.097, -0.062] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | phon | mostly_right | -0.045 [-0.072, -0.018] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | phon|orth | all | -0.044 [-0.056, -0.032] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | phon|orth | mostly_right | -0.028 [-0.048, -0.008] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | sem | all | -0.024 [-0.038, -0.010] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | sem | mostly_right | +0.008 [-0.013, +0.029] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | ctx | all | -0.198 [-0.463, +0.064] |
| Aero, random-init encoder (4a) | Qwen Instruct + diffusion | ctx | mostly_right | +0.793 [+0.429, +1.171] |
| Aero, random-init encoder (4a) | Aero + diffusion | phon | all | -0.077 [-0.095, -0.058] |
| Aero, random-init encoder (4a) | Aero + diffusion | phon | mostly_right | -0.041 [-0.068, -0.013] |
| Aero, random-init encoder (4a) | Aero + diffusion | phon|orth | all | -0.038 [-0.051, -0.026] |
| Aero, random-init encoder (4a) | Aero + diffusion | phon|orth | mostly_right | -0.019 [-0.040, +0.002] |
| Aero, random-init encoder (4a) | Aero + diffusion | sem | all | -0.028 [-0.041, -0.015] |
| Aero, random-init encoder (4a) | Aero + diffusion | sem | mostly_right | +0.008 [-0.012, +0.028] |
| Aero, random-init encoder (4a) | Aero + diffusion | ctx | all | -0.389 [-0.645, -0.140] |
| Aero, random-init encoder (4a) | Aero + diffusion | ctx | mostly_right | +0.693 [+0.307, +1.080] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | phon | all | -0.169 [-0.193, -0.146] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.127 [-0.158, -0.095] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.088 [-0.106, -0.071] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.066 [-0.089, -0.043] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | sem | all | -0.026 [-0.042, -0.011] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | +0.002 [-0.020, +0.024] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | ctx | all | -0.370 [-0.678, -0.059] |
| Aero, random-init encoder (4a) | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | +0.644 [+0.198, +1.089] |
| Aero, random-init encoder (4a) | Arm A + diffusion | phon | all | -0.182 [-0.206, -0.159] |
| Aero, random-init encoder (4a) | Arm A + diffusion | phon | mostly_right | -0.133 [-0.164, -0.103] |
| Aero, random-init encoder (4a) | Arm A + diffusion | phon|orth | all | -0.103 [-0.119, -0.085] |
| Aero, random-init encoder (4a) | Arm A + diffusion | phon|orth | mostly_right | -0.078 [-0.101, -0.056] |
| Aero, random-init encoder (4a) | Arm A + diffusion | sem | all | -0.020 [-0.035, -0.004] |
| Aero, random-init encoder (4a) | Arm A + diffusion | sem | mostly_right | +0.017 [-0.005, +0.038] |
| Aero, random-init encoder (4a) | Arm A + diffusion | ctx | all | +0.010 [-0.302, +0.325] |
| Aero, random-init encoder (4a) | Arm A + diffusion | ctx | mostly_right | +1.154 [+0.702, +1.600] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | phon | all | -0.012 [-0.028, +0.005] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | -0.012 [-0.038, +0.014] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | -0.009 [-0.021, +0.002] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | -0.011 [-0.030, +0.009] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | sem | all | -0.002 [-0.014, +0.009] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | +0.012 [-0.009, +0.033] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | ctx | all | -0.187 [-0.434, +0.051] |
| Aero, random-init encoder (4a) | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | +0.021 [-0.373, +0.426] |
| Aero, random-init encoder (4a) | Arm C + diffusion | phon | all | -0.089 [-0.107, -0.071] |
| Aero, random-init encoder (4a) | Arm C + diffusion | phon | mostly_right | -0.063 [-0.090, -0.037] |
| Aero, random-init encoder (4a) | Arm C + diffusion | phon|orth | all | -0.047 [-0.060, -0.035] |
| Aero, random-init encoder (4a) | Arm C + diffusion | phon|orth | mostly_right | -0.037 [-0.057, -0.017] |
| Aero, random-init encoder (4a) | Arm C + diffusion | sem | all | -0.028 [-0.042, -0.014] |
| Aero, random-init encoder (4a) | Arm C + diffusion | sem | mostly_right | +0.004 [-0.016, +0.024] |
| Aero, random-init encoder (4a) | Arm C + diffusion | ctx | all | -0.178 [-0.448, +0.090] |
| Aero, random-init encoder (4a) | Arm C + diffusion | ctx | mostly_right | +0.934 [+0.519, +1.357] |
| Qwen Instruct + diffusion | Aero + diffusion | phon | all | +0.003 [-0.013, +0.019] |
| Qwen Instruct + diffusion | Aero + diffusion | phon | mostly_right | +0.004 [-0.018, +0.027] |
| Qwen Instruct + diffusion | Aero + diffusion | phon|orth | all | +0.006 [-0.006, +0.018] |
| Qwen Instruct + diffusion | Aero + diffusion | phon|orth | mostly_right | +0.009 [-0.007, +0.024] |
| Qwen Instruct + diffusion | Aero + diffusion | sem | all | -0.004 [-0.018, +0.009] |
| Qwen Instruct + diffusion | Aero + diffusion | sem | mostly_right | +0.001 [-0.016, +0.017] |
| Qwen Instruct + diffusion | Aero + diffusion | ctx | all | -0.191 [-0.428, +0.045] |
| Qwen Instruct + diffusion | Aero + diffusion | ctx | mostly_right | -0.099 [-0.394, +0.215] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | phon | all | -0.090 [-0.114, -0.065] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.082 [-0.112, -0.052] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.044 [-0.062, -0.026] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.038 [-0.060, -0.016] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | sem | all | -0.003 [-0.019, +0.014] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | -0.006 [-0.027, +0.015] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | ctx | all | -0.173 [-0.460, +0.115] |
| Qwen Instruct + diffusion | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | -0.148 [-0.522, +0.211] |
| Qwen Instruct + diffusion | Arm A + diffusion | phon | all | -0.103 [-0.127, -0.079] |
| Qwen Instruct + diffusion | Arm A + diffusion | phon | mostly_right | -0.088 [-0.116, -0.060] |
| Qwen Instruct + diffusion | Arm A + diffusion | phon|orth | all | -0.059 [-0.075, -0.041] |
| Qwen Instruct + diffusion | Arm A + diffusion | phon|orth | mostly_right | -0.050 [-0.070, -0.029] |
| Qwen Instruct + diffusion | Arm A + diffusion | sem | all | +0.004 [-0.013, +0.021] |
| Qwen Instruct + diffusion | Arm A + diffusion | sem | mostly_right | +0.009 [-0.011, +0.028] |
| Qwen Instruct + diffusion | Arm A + diffusion | ctx | all | +0.207 [-0.092, +0.495] |
| Qwen Instruct + diffusion | Arm A + diffusion | ctx | mostly_right | +0.361 [+0.007, +0.714] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | phon | all | +0.068 [+0.050, +0.087] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | +0.033 [+0.008, +0.059] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | +0.035 [+0.022, +0.048] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | +0.017 [-0.001, +0.036] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | sem | all | +0.022 [+0.009, +0.035] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | +0.004 [-0.015, +0.023] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | ctx | all | +0.011 [-0.243, +0.264] |
| Qwen Instruct + diffusion | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.772 [-1.119, -0.444] |
| Qwen Instruct + diffusion | Arm C + diffusion | phon | all | -0.010 [-0.028, +0.008] |
| Qwen Instruct + diffusion | Arm C + diffusion | phon | mostly_right | -0.018 [-0.042, +0.005] |
| Qwen Instruct + diffusion | Arm C + diffusion | phon|orth | all | -0.003 [-0.016, +0.009] |
| Qwen Instruct + diffusion | Arm C + diffusion | phon|orth | mostly_right | -0.009 [-0.026, +0.008] |
| Qwen Instruct + diffusion | Arm C + diffusion | sem | all | -0.004 [-0.018, +0.009] |
| Qwen Instruct + diffusion | Arm C + diffusion | sem | mostly_right | -0.003 [-0.020, +0.013] |
| Qwen Instruct + diffusion | Arm C + diffusion | ctx | all | +0.020 [-0.226, +0.273] |
| Qwen Instruct + diffusion | Arm C + diffusion | ctx | mostly_right | +0.142 [-0.168, +0.460] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | phon | all | -0.093 [-0.117, -0.069] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | phon | mostly_right | -0.086 [-0.114, -0.058] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | phon|orth | all | -0.050 [-0.067, -0.033] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | phon|orth | mostly_right | -0.046 [-0.067, -0.026] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | sem | all | +0.002 [-0.015, +0.019] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | sem | mostly_right | -0.007 [-0.027, +0.013] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | ctx | all | +0.018 [-0.285, +0.324] |
| Aero + diffusion | Arm A: phoneme CTC encoder + Aero | ctx | mostly_right | -0.049 [-0.425, +0.309] |
| Aero + diffusion | Arm A + diffusion | phon | all | -0.106 [-0.128, -0.083] |
| Aero + diffusion | Arm A + diffusion | phon | mostly_right | -0.092 [-0.118, -0.067] |
| Aero + diffusion | Arm A + diffusion | phon|orth | all | -0.064 [-0.081, -0.047] |
| Aero + diffusion | Arm A + diffusion | phon|orth | mostly_right | -0.059 [-0.079, -0.040] |
| Aero + diffusion | Arm A + diffusion | sem | all | +0.009 [-0.007, +0.024] |
| Aero + diffusion | Arm A + diffusion | sem | mostly_right | +0.008 [-0.010, +0.027] |
| Aero + diffusion | Arm A + diffusion | ctx | all | +0.399 [+0.118, +0.689] |
| Aero + diffusion | Arm A + diffusion | ctx | mostly_right | +0.460 [+0.107, +0.809] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | phon | all | +0.065 [+0.046, +0.084] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | +0.029 [+0.005, +0.054] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | +0.029 [+0.016, +0.043] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | +0.009 [-0.010, +0.027] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | sem | all | +0.026 [+0.013, +0.038] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | +0.004 [-0.014, +0.021] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | ctx | all | +0.202 [-0.049, +0.467] |
| Aero + diffusion | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.673 [-1.015, -0.334] |
| Aero + diffusion | Arm C + diffusion | phon | all | -0.012 [-0.030, +0.006] |
| Aero + diffusion | Arm C + diffusion | phon | mostly_right | -0.023 [-0.044, -0.001] |
| Aero + diffusion | Arm C + diffusion | phon|orth | all | -0.009 [-0.022, +0.004] |
| Aero + diffusion | Arm C + diffusion | phon|orth | mostly_right | -0.018 [-0.034, -0.002] |
| Aero + diffusion | Arm C + diffusion | sem | all | +0.000 [-0.013, +0.013] |
| Aero + diffusion | Arm C + diffusion | sem | mostly_right | -0.004 [-0.020, +0.012] |
| Aero + diffusion | Arm C + diffusion | ctx | all | +0.211 [-0.037, +0.450] |
| Aero + diffusion | Arm C + diffusion | ctx | mostly_right | +0.241 [-0.060, +0.543] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | phon | all | -0.013 [-0.035, +0.009] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | phon | mostly_right | -0.007 [-0.032, +0.020] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | phon|orth | all | -0.014 [-0.031, +0.003] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | phon|orth | mostly_right | -0.012 [-0.032, +0.007] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | sem | all | +0.007 [-0.010, +0.023] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | sem | mostly_right | +0.015 [-0.004, +0.034] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | ctx | all | +0.380 [+0.093, +0.662] |
| Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | ctx | mostly_right | +0.509 [+0.160, +0.854] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | phon | all | +0.158 [+0.135, +0.181] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | +0.115 [+0.088, +0.143] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | +0.079 [+0.062, +0.096] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | +0.055 [+0.034, +0.076] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | sem | all | +0.024 [+0.009, +0.040] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | +0.010 [-0.012, +0.031] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | ctx | all | +0.183 [-0.114, +0.481] |
| Arm A: phoneme CTC encoder + Aero | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -0.624 [-1.009, -0.229] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | phon | all | +0.080 [+0.056, +0.104] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | phon | mostly_right | +0.063 [+0.036, +0.091] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | phon|orth | all | +0.041 [+0.024, +0.057] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | phon|orth | mostly_right | +0.029 [+0.009, +0.049] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | sem | all | -0.002 [-0.019, +0.015] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | sem | mostly_right | +0.003 [-0.018, +0.023] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | ctx | all | +0.193 [-0.110, +0.503] |
| Arm A: phoneme CTC encoder + Aero | Arm C + diffusion | ctx | mostly_right | +0.290 [-0.085, +0.666] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | phon | all | +0.171 [+0.148, +0.194] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | phon | mostly_right | +0.122 [+0.093, +0.150] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | phon|orth | all | +0.093 [+0.076, +0.110] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | phon|orth | mostly_right | +0.068 [+0.046, +0.089] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | sem | all | +0.017 [+0.002, +0.033] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | sem | mostly_right | -0.005 [-0.025, +0.014] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | ctx | all | -0.197 [-0.515, +0.122] |
| Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | ctx | mostly_right | -1.133 [-1.526, -0.755] |
| Arm A + diffusion | Arm C + diffusion | phon | all | +0.093 [+0.070, +0.116] |
| Arm A + diffusion | Arm C + diffusion | phon | mostly_right | +0.070 [+0.044, +0.096] |
| Arm A + diffusion | Arm C + diffusion | phon|orth | all | +0.055 [+0.039, +0.072] |
| Arm A + diffusion | Arm C + diffusion | phon|orth | mostly_right | +0.041 [+0.022, +0.060] |
| Arm A + diffusion | Arm C + diffusion | sem | all | -0.008 [-0.025, +0.008] |
| Arm A + diffusion | Arm C + diffusion | sem | mostly_right | -0.012 [-0.031, +0.007] |
| Arm A + diffusion | Arm C + diffusion | ctx | all | -0.187 [-0.484, +0.110] |
| Arm A + diffusion | Arm C + diffusion | ctx | mostly_right | -0.219 [-0.580, +0.149] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | phon | all | -0.077 [-0.096, -0.059] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | phon | mostly_right | -0.052 [-0.076, -0.028] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | phon|orth | all | -0.038 [-0.051, -0.025] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | phon|orth | mostly_right | -0.027 [-0.045, -0.008] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | sem | all | -0.026 [-0.039, -0.013] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | sem | mostly_right | -0.007 [-0.024, +0.009] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | ctx | all | +0.009 [-0.272, +0.291] |
| Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion | ctx | mostly_right | +0.914 [+0.565, +1.264] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | phon | all | +0.192 [+0.177, +0.207] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | phon|orth | all | +0.103 [+0.093, +0.114] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | sem | all | +0.074 [+0.064, +0.084] |
| Fast-dLLM v2 (diffusion) | Fast-dLLM v2 (diffusion) · null_shuffled | ctx | all | +1.716 [+1.504, +1.931] |
| Aero | Aero · null_shuffled | phon | all | +0.117 [+0.104, +0.131] |
| Aero | Aero · null_shuffled | phon|orth | all | +0.060 [+0.050, +0.070] |
| Aero | Aero · null_shuffled | sem | all | +0.055 [+0.044, +0.065] |
| Aero | Aero · null_shuffled | ctx | all | +1.650 [+1.409, +1.903] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | phon | all | +0.077 [+0.066, +0.088] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | phon|orth | all | +0.042 [+0.034, +0.050] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | sem | all | +0.046 [+0.037, +0.055] |
| Qwen base, no prompt | Qwen base, no prompt · null_shuffled | ctx | all | +1.439 [+1.236, +1.642] |
| Qwen Instruct | Qwen Instruct · null_shuffled | phon | all | +0.044 [+0.036, +0.053] |
| Qwen Instruct | Qwen Instruct · null_shuffled | phon|orth | all | +0.022 [+0.015, +0.029] |
| Qwen Instruct | Qwen Instruct · null_shuffled | sem | all | +0.030 [+0.023, +0.037] |
| Qwen Instruct | Qwen Instruct · null_shuffled | ctx | all | +1.020 [+0.859, +1.188] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | phon | all | +0.123 [+0.110, +0.137] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | phon|orth | all | +0.065 [+0.055, +0.075] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | sem | all | +0.054 [+0.044, +0.064] |
| Aero, random-init encoder (4a) | Aero, random-init encoder (4a) · null_shuffled | ctx | all | +1.626 [+1.408, +1.845] |
| Qwen Instruct + diffusion | Qwen Instruct + diffusion · null_shuffled | phon | all | +0.204 [+0.189, +0.220] |
| Qwen Instruct + diffusion | Qwen Instruct + diffusion · null_shuffled | phon|orth | all | +0.110 [+0.099, +0.121] |
| Qwen Instruct + diffusion | Qwen Instruct + diffusion · null_shuffled | sem | all | +0.077 [+0.066, +0.088] |
| Qwen Instruct + diffusion | Qwen Instruct + diffusion · null_shuffled | ctx | all | +1.828 [+1.600, +2.059] |
| Aero + diffusion | Aero + diffusion · null_shuffled | phon | all | +0.201 [+0.185, +0.217] |
| Aero + diffusion | Aero + diffusion · null_shuffled | phon|orth | all | +0.104 [+0.092, +0.116] |
| Aero + diffusion | Aero + diffusion · null_shuffled | sem | all | +0.082 [+0.071, +0.094] |
| Aero + diffusion | Aero + diffusion · null_shuffled | ctx | all | +2.013 [+1.785, +2.253] |
| Arm A: phoneme CTC encoder + Aero | Arm A: phoneme CTC encoder + Aero · null_shuffled | phon | all | +0.294 [+0.272, +0.317] |
| Arm A: phoneme CTC encoder + Aero | Arm A: phoneme CTC encoder + Aero · null_shuffled | phon|orth | all | +0.155 [+0.139, +0.171] |
| Arm A: phoneme CTC encoder + Aero | Arm A: phoneme CTC encoder + Aero · null_shuffled | sem | all | +0.081 [+0.067, +0.095] |
| Arm A: phoneme CTC encoder + Aero | Arm A: phoneme CTC encoder + Aero · null_shuffled | ctx | all | +2.055 [+1.780, +2.341] |
| Arm A + diffusion | Arm A + diffusion · null_shuffled | phon | all | +0.305 [+0.283, +0.328] |
| Arm A + diffusion | Arm A + diffusion · null_shuffled | phon|orth | all | +0.167 [+0.151, +0.184] |
| Arm A + diffusion | Arm A + diffusion · null_shuffled | sem | all | +0.074 [+0.061, +0.088] |
| Arm A + diffusion | Arm A + diffusion · null_shuffled | ctx | all | +1.675 [+1.394, +1.960] |
| Arm C: contrastive (Aero L14) + Aero | Arm C: contrastive (Aero L14) + Aero · null_shuffled | phon | all | +0.137 [+0.122, +0.151] |
| Arm C: contrastive (Aero L14) + Aero | Arm C: contrastive (Aero L14) + Aero · null_shuffled | phon|orth | all | +0.075 [+0.065, +0.086] |
| Arm C: contrastive (Aero L14) + Aero | Arm C: contrastive (Aero L14) + Aero · null_shuffled | sem | all | +0.057 [+0.047, +0.067] |
| Arm C: contrastive (Aero L14) + Aero | Arm C: contrastive (Aero L14) + Aero · null_shuffled | ctx | all | +1.854 [+1.610, +2.100] |
| Arm C + diffusion | Arm C + diffusion · null_shuffled | phon | all | +0.213 [+0.197, +0.229] |
| Arm C + diffusion | Arm C + diffusion · null_shuffled | phon|orth | all | +0.113 [+0.101, +0.124] |
| Arm C + diffusion | Arm C + diffusion · null_shuffled | sem | all | +0.082 [+0.071, +0.095] |
| Arm C + diffusion | Arm C + diffusion · null_shuffled | ctx | all | +1.806 [+1.570, +2.047] |

## 20 most common substitutions (reference → decoded, count)

| # | Fast-dLLM v2 (diffusion) | Aero | Qwen base, no prompt | Qwen Instruct | Aero, random-init encoder (4a) | Qwen Instruct + diffusion | Aero + diffusion | Arm A: phoneme CTC encoder + Aero | Arm A + diffusion | Arm C: contrastive (Aero L14) + Aero | Arm C + diffusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | has → as (4) | the → a (3) | in → the (2) | of → in (3) | in → and (3) | the → they (4) | its → it's (3) | in → it (7) | in → it (4) | got → not (3) | dallas → d (3) |
| 2 | the → they (3) | to → the (3) | the → it's (2) | as → a (3) | for → your (3) | its → it's (3) | he's → it's (3) | has → as (4) | has → as (3) | they → the (3) | its → it's (3) |
| 3 | their → there (3) | he's → it's (3) | these → those (2) | in → and (3) | in → it (3) | in → it (3) | have → i (3) | his → is (4) | his → is (3) | these → this (3) | has → as (3) |
| 4 | such → just (3) | out → up (3) | on → and (2) | of → the (3) | to → do (3) | anyway → anymore (2) | in → it (3) | understand → appreciate (3) | its → it's (3) | and → in (3) | such → just (3) |
| 5 | clothes → close (3) | the → that (2) | the → a (2) | a → and (3) | well → will (2) | oh → who (2) | how → i (3) | been → bin (2) | failed → field (2) | next → this (2) | had → and (3) |
| 6 | in → and (3) | it → in (2) | a → to (2) | in → it (3) | has → as (2) | a → he (2) | than → that (2) | that → not (2) | as → has (2) | slow → still (2) | more → for (2) |
| 7 | and → a (3) | the → do (2) | not → but (2) | in → of (3) | a → he (2) | as → is (2) | got → not (2) | service → services (2) | york → your (2) | the → a (2) | got → not (2) |
| 8 | in → it (3) | of → is (2) | to → the (2) | it → it's (3) | an → a (2) | in → and (2) | these → those (2) | been → bit (2) | the → they (2) | look → like (2) | the → they (2) |
| 9 | anyway → anymore (2) | up → of (2) | to → a (2) | of → over (3) | in → is (2) | on → in (2) | served → sold (2) | did → didn't (2) | that → not (2) | as → have (2) | understand → outside (2) |
| 10 | is → in (2) | such → just (2) | that → there (2) | in → the (2) | as → you (2) | such → just (2) | at → it (2) | lot → light (2) | how → i (2) | light → like (2) | during → going (2) |
| 11 | it → he (2) | look → like (2) | the → go (2) | on → of (2) | how → i (2) | how → i (2) | such → just (2) | two → to (2) | right → what (2) | and → an (2) | and → a (2) |
| 12 | oh → who (2) | him → of (2) | of → do (2) | in → with (2) | are → a (2) | they → the (2) | and → in (2) | in → and (2) | fan → fine (2) | what → one (2) | what's → what (2) |
| 13 | got → not (2) | year → here (2) | as → have (2) | in → a (2) | as → for (2) | it → in (2) | which → was (2) | fill → feel (2) | in → and (2) | choice → easy (2) | over → all (2) |
| 14 | its → it's (2) | a → he (2) | i → you (2) | the → little (2) | year → here (2) | dallas → d (2) | had → and (2) | sew → so (2) | sew → so (2) | avoid → avoiding (2) | in → it (2) |
| 15 | has → is (2) | and → a (2) | for → of (2) | work → week (2) | the → they (2) | had → and (2) | they're → they (2) | had → and (2) | either → any (2) | card → course (2) | fan → fine (2) |
| 16 | bay → pay (2) | of → the (2) | more → for (2) | in → to (2) | education → experience (2) | they're → there (2) | one → what (2) | clothes → close (2) | turn → turned (2) | new → two (2) | high → i (2) |
| 17 | once → what's (2) | had → and (2) | money → morning (2) | for → of (2) | building → buying (2) | those → this (2) | they're → there (2) | those → this (2) | had → and (2) | get → gotten (2) | they're → they (2) |
| 18 | look → like (2) | new → you (2) | his → is (2) | up → it (2) | its → it's (2) | has → as (2) | those → this (2) | in → on (2) | eight → at (2) | of → in (2) | on → at (2) |
| 19 | half → have (2) | they're → there (2) | run → one (2) | her → the (2) | high → i (2) | or → are (2) | have → how (2) | delivered → devoted (1) | clothes → close (2) | done → that (1) | at → and (2) |
| 20 | what's → what (2) | as → has (2) | have → i (2) | of → her (2) | and → at (2) | come → go (1) | these → this (2) | foxes → favorites (1) | delivered → delighted (1) | anyway → america (1) | do → to (2) |
