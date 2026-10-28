# Novel test words: recall of words absent from the training sentences

`scripts/novel_word_analysis.py` (CPU; saved BIT-test greedy decodes, real input). Paired cluster bootstrap over the 713 test trials, 10000 resamples, seed 0; 95% percentile CIs. Recall = reference words aligned as correct (fewest-edit word alignment, most hits among ties). One seed per run.

## Training vocabulary

- Training sentences: 8072 train trials from 45 sessions, 7307 distinct; 3744 word types. Every run's `seen_in_train` flags match these sentences: yes.
- Test references: 4770 tokens, 1297 types; **novel: 284 tokens (6.0%), 274 types (21.1%)**, in 195 of 713 trials (632 trials are unseen sentences; seen sentences have no novel word by construction).
- Novel vs in-vocab (unseen trials) tokens: median letters 7 vs 4; median English Zipf 4.13 vs 6.25; 264 of the novel types occur once in the test set.
- Most frequent novel words: delivered (2), garbage (2), failed (2), bay (2), courier (2), corn (2), ran (2), sew (2), understanding (2), drastic (2), foxes (1), raccoons (1), raiding (1), michigan (1), dwelling (1).

## Per run

| Run | WER | In-vocab recall (all) | In-vocab recall (unseen trials) | **Novel recall** [CI] | Novel sub | Novel del | Wrong novel -> known word | Wrong in-vocab -> known word | Novel outputs (correct) |
|---|---|---|---|---|---|---|---|---|---|
| B AR (Aero) | 34.2 | 73.0 | 71.9 | **6.0 [3.4, 8.9]** | 87.3 | 6.7 | 82.3 (204/248) | 91.9 | 172 (17) |
| B diffusion (Aero) | 21.8 | 83.6 | 82.7 | **5.3 [2.8, 8.1]** | 90.5 | 4.2 | 81.3 (209/257) | 90.7 | 130 (15) |
| A AR (Aero) | 15.5 | 88.7 | 89.0 | **26.8 [21.1, 32.4]** | 70.4 | 2.8 | 62.5 (125/200) | 84.3 | 231 (76) |
| A diffusion (Aero) | 15.6 | 89.0 | 89.4 | **19.4 [14.2, 24.7]** | 77.8 | 2.8 | 65.2 (144/221) | 81.4 | 222 (55) |
| C AR (Aero) | 29.2 | 76.8 | 75.1 | **8.1 [5.1, 11.4]** | 87.3 | 4.6 | 83.1 (206/248) | 92.0 | 151 (23) |
| C diffusion (Aero) | 22.7 | 82.5 | 81.3 | **6.3 [3.6, 9.3]** | 90.1 | 3.5 | 79.7 (204/256) | 91.1 | 135 (18) |
| B AR (Instruct) | 55.7 | 52.3 | 49.1 | **1.8 [0.4, 3.4]** | 82.4 | 15.8 | 89.7 (210/234) | 93.6 | 152 (5) |
| B diffusion (Instruct) | 23.6 | 81.6 | 81.1 | **3.2 [1.3, 5.4]** | 92.3 | 4.6 | 81.3 (213/262) | 90.9 | 133 (9) |
| B Fast-dLLM v2 | 25.2 | 79.9 | 79.1 | **3.2 [1.3, 5.4]** | 92.6 | 4.2 | 82.1 (216/263) | 91.4 | 131 (9) |

Percentages. Novel sub / del: share of novel reference tokens substituted / deleted (recall + sub + del = 100). Wrong -> known word: share of substituted reference words whose decoded word is in the training vocabulary. Novel outputs: decoded tokens not in the training vocabulary (how many are aligned as correct).

## By training frequency (in-vocab words, unseen-sentence trials)

| Run | Novel (0) | 1-2 | 3-10 | >10 |
|---|---|---|---|---|
| B AR (Aero) | 6.0 (n=284) | 16.2 (n=235) | 31.9 (n=323) | 79.6 (n=3389) |
| B diffusion (Aero) | 5.3 (n=284) | 28.1 (n=235) | 48.3 (n=323) | 89.7 (n=3389) |
| A AR (Aero) | 26.8 (n=284) | 53.2 (n=235) | 66.9 (n=323) | 93.6 (n=3389) |
| A diffusion (Aero) | 19.4 (n=284) | 56.6 (n=235) | 68.1 (n=323) | 93.7 (n=3389) |
| C AR (Aero) | 8.1 (n=284) | 22.6 (n=235) | 33.1 (n=323) | 82.7 (n=3389) |
| C diffusion (Aero) | 6.3 (n=284) | 21.7 (n=235) | 43.7 (n=323) | 89.1 (n=3389) |
| B AR (Instruct) | 1.8 (n=284) | 4.7 (n=235) | 10.5 (n=323) | 55.9 (n=3389) |
| B diffusion (Instruct) | 3.2 (n=284) | 18.3 (n=235) | 39.6 (n=323) | 89.4 (n=3389) |
| B Fast-dLLM v2 | 3.2 (n=284) | 12.3 (n=235) | 40.9 (n=323) | 87.4 (n=3389) |

## Novel words: inflections of training words vs new stems

Inflection (heuristic: -s, -es, -ed, -d, -ing, -'s either way, with e-drop, doubling, y/i): 65 of 284 novel tokens, e.g. adding, agreement, animal, bed, besides, caused, cities, comment. New stems, e.g. academy, accelerated, accidents, actual, addiction, affectionate, algorithm, ally.

| Run | Inflection recall [CI] | New-stem recall [CI] | Novel words found anywhere in the decode |
|---|---|---|---|
| B AR (Aero) | 15.4 [7.1, 24.7] (n=65) | 3.2 [1.0, 5.7] (n=219) | 17 (aligned correct: 17) |
| B diffusion (Aero) | 7.7 [1.7, 15.0] (n=65) | 4.6 [2.0, 7.6] (n=219) | 15 (aligned correct: 15) |
| A AR (Aero) | 41.5 [29.2, 53.8] (n=65) | 22.4 [16.5, 28.4] (n=219) | 76 (aligned correct: 76) |
| A diffusion (Aero) | 23.1 [13.1, 33.9] (n=65) | 18.3 [12.6, 24.2] (n=219) | 55 (aligned correct: 55) |
| C AR (Aero) | 16.9 [8.5, 27.0] (n=65) | 5.5 [2.7, 8.6] (n=219) | 23 (aligned correct: 23) |
| C diffusion (Aero) | 7.7 [1.6, 14.8] (n=65) | 5.9 [3.0, 9.3] (n=219) | 18 (aligned correct: 18) |
| B AR (Instruct) | 7.7 [1.7, 14.9] (n=65) | 0.0 [0.0, 0.0] (n=219) | 5 (aligned correct: 5) |
| B diffusion (Instruct) | 10.8 [3.9, 18.8] (n=65) | 0.9 [0.0, 2.4] (n=219) | 9 (aligned correct: 9) |
| B Fast-dLLM v2 | 9.2 [3.0, 16.9] (n=65) | 1.4 [0.0, 3.1] (n=219) | 9 (aligned correct: 9) |

## Matched in-vocab words (unseen-sentence trials)

Novel recall minus the recall of in-vocab words standardized to the novel words' cells. len: letters; len x Zipf: letters and English word frequency; rare len: in-vocab words seen 1-2 times in training, by letters. Coverage = share of novel tokens in cells that have in-vocab tokens.

| Run | Novel | len: matched / gap | len x Zipf: matched / gap | rare len: matched / gap |
|---|---|---|---|---|
| B AR (Aero) | 6.0 | 50.8 / -44.8 [-49.3, -40.2] | 22.4 / -16.2 [-23.0, -10.3] | 15.4 / -9.4 [-14.9, -3.8] |
| B diffusion (Aero) | 5.3 | 65.8 / -60.5 [-64.8, -56.0] | 45.0 / -39.5 [-48.8, -30.0] | 27.0 / -21.7 [-28.9, -15.0] |
| A AR (Aero) | 26.8 | 79.8 / -53.1 [-59.2, -46.9] | 55.1 / -27.6 [-38.6, -17.8] | 53.4 / -26.7 [-36.0, -17.8] |
| A diffusion (Aero) | 19.4 | 81.0 / -61.7 [-67.3, -55.9] | 63.1 / -43.0 [-52.8, -32.9] | 56.7 / -37.4 [-46.2, -28.3] |
| C AR (Aero) | 8.1 | 53.7 / -45.6 [-50.5, -40.7] | 32.8 / -24.4 [-33.2, -15.3] | 21.3 / -13.2 [-19.6, -7.3] |
| C diffusion (Aero) | 6.3 | 63.4 / -57.0 [-61.4, -52.6] | 37.7 / -31.1 [-40.9, -23.0] | 21.2 / -14.9 [-21.2, -8.8] |
| B AR (Instruct) | 1.8 | 26.6 / -24.9 [-28.3, -21.5] | 8.5 / -6.6 [-10.4, -3.6] | 4.1 / -2.4 [-5.6, +0.5] |
| B diffusion (Instruct) | 3.2 | 61.8 / -58.7 [-62.6, -54.6] | 32.5 / -29.2 [-38.0, -21.9] | 19.0 / -15.8 [-21.6, -10.1] |
| B Fast-dLLM v2 | 3.2 | 58.5 / -55.3 [-59.3, -51.2] | 26.1 / -22.8 [-29.5, -17.1] | 11.6 / -8.4 [-13.4, -3.8] |

Coverage: gap_len 100.0% (n novel 284, pool 3947), gap_len_zipf 96.1% (n novel 284, pool 3947), gap_rare_len 100.0% (n novel 284, pool 235).

## Paired contrasts (pp; a - b)

| Contrast | Novel recall | In-vocab recall (unseen) | Gap vs len x Zipf-matched |
|---|---|---|---|
| B diff - B AR | -0.7 [-4.0, +2.6] (p=0.769) | +10.8 [+9.1, +12.6] | -23.3 [-32.4, -13.5] (p<0.001) |
| A diff - A AR | -7.4 [-13.1, -1.5] (p=0.020) | +0.4 [-0.6, +1.3] | -15.4 [-26.5, -5.1] (p=0.002) |
| C diff - C AR | -1.8 [-5.4, +1.8] (p=0.386) | +6.2 [+4.6, +7.9] | -6.8 [-16.3, +2.9] (p=0.162) |
| C AR - B AR | +2.1 [-1.2, +5.5] (p=0.254) | +3.2 [+1.5, +4.9] | -8.2 [-16.3, +0.8] (p=0.079) |
| C diff - B diff | +1.1 [-2.1, +4.2] (p=0.593) | -1.3 [-2.6, -0.1] | +8.4 [+0.1, +17.9] (p=0.048) |
| A AR - B AR | +20.8 [+15.2, +26.3] (p<0.001) | +17.1 [+15.2, +19.1] | -11.4 [-22.4, -1.4] (p=0.027) |
| A diff - B diff | +14.1 [+8.9, +19.5] (p<0.001) | +6.7 [+5.5, +7.9] | -3.5 [-12.4, +4.6] (p=0.389) |
| Instruct diff - Instruct AR | +1.4 [-0.7, +3.7] (p=0.265) | +32.0 [+29.9, +34.1] | -22.6 [-31.5, -15.3] (p<0.001) |
| (A diff - A AR) - (B diff - B AR) | -6.7 [-13.4, +0.0] (p=0.054) | -10.4 [-12.3, -8.6] | +7.9 [-3.3, +19.7] (p=0.182) |
| (C diff - C AR) - (B diff - B AR) | -1.1 [-5.8, +3.8] (p=0.690) | -4.6 [-6.6, -2.5] | +16.5 [+2.5, +30.8] (p=0.017) |

## Sound of the wrong word (substitutions)

Phoneme-similarity excess over the run's own length-matched substitutes and the sound-alike share (phon >= 0.5; null in brackets), from `scripts/phonetic_error_analysis.py` (read-only reuse). Only pairs with both words in the CMU dictionary.

| Run | Novel: n (in dict) | Novel: excess | Novel: alike (null) | In-vocab: excess | In-vocab: alike (null) | Novel - in-vocab excess |
|---|---|---|---|---|---|---|
| B AR (Aero) | 248 (245) | +0.107 [+0.081, +0.134] | 11.0 (1.1) | +0.120 | 17.3 (3.0) | -0.014 [-0.042, +0.016] |
| B diffusion (Aero) | 257 (245) | +0.197 [+0.170, +0.225] | 22.4 (1.7) | +0.202 | 27.1 (3.0) | -0.005 [-0.037, +0.028] |
| A AR (Aero) | 200 (188) | +0.290 [+0.255, +0.327] | 39.4 (2.2) | +0.295 | 43.1 (3.7) | -0.005 [-0.048, +0.038] |
| A diffusion (Aero) | 221 (199) | +0.308 [+0.276, +0.342] | 40.7 (2.5) | +0.305 | 44.2 (3.4) | +0.003 [-0.037, +0.044] |
| C AR (Aero) | 248 (244) | +0.140 [+0.113, +0.168] | 14.8 (1.6) | +0.137 | 17.0 (2.6) | +0.003 [-0.026, +0.033] |
| C diffusion (Aero) | 256 (245) | +0.213 [+0.187, +0.241] | 24.5 (1.8) | +0.214 | 28.5 (3.0) | -0.001 [-0.032, +0.032] |
| B AR (Instruct) | 234 (232) | +0.048 [+0.031, +0.065] | 1.7 (0.8) | +0.048 | 7.8 (2.7) | -0.000 [-0.019, +0.018] |
| B diffusion (Instruct) | 262 (248) | +0.186 [+0.158, +0.215] | 19.8 (1.7) | +0.209 | 27.4 (3.0) | -0.022 [-0.056, +0.012] |
| B Fast-dLLM v2 | 263 (252) | +0.176 [+0.150, +0.203] | 19.0 (1.6) | +0.200 | 25.5 (3.0) | -0.024 [-0.055, +0.008] |

Novel-word excess, paired: B diff - B AR +0.091 [+0.059, +0.121]; A diff - A AR +0.018 [-0.020, +0.055]; C diff - C AR +0.073 [+0.043, +0.104]; C AR - B AR +0.033 [+0.002, +0.064]; C diff - B diff +0.016 [-0.015, +0.047]; A AR - B AR +0.184 [+0.141, +0.226]; A diff - B diff +0.111 [+0.077, +0.145]; Instruct diff - Instruct AR +0.138 [+0.106, +0.171].

## Meaning of the wrong word (substitutions)

When a decoder replaces a word, is the replacement related in meaning? Excess = observed - null, the null being the run's own substitutes of the same phoneme length (phonetic_error_analysis.py's, read-only reuse). sem: cosine of decontextualized distilbert-base-uncased word vectors (centred); related: sem >= 0.4 (share, null in brackets); ctx: masked-LM log-probability (nats) of the decoded word in the reference word's slot (one-piece decoded words only; coverage in brackets); contextual: cosine of roberta-large layer-17 vectors of the reference word in the reference sentence and of the decoded word put in its slot (centred; 3 null draws per pair, 38490 vectors). Pairs with both words in the CMU dictionary.

| Run | Novel subs (in dict) | sem excess | related (null) | ctx excess (coverage) | contextual excess |
|---|---|---|---|---|---|
| B AR (Aero) | 248 (245) | +0.051 [+0.031, +0.073] | 2.4 (0.8) | +2.10 [+1.71, +2.50] (92%) | +0.035 [+0.014, +0.058] |
| B diffusion (Aero) | 257 (245) | +0.073 [+0.054, +0.093] | 4.1 (0.9) | +2.05 [+1.67, +2.42] (91%) | +0.038 [+0.020, +0.056] |
| A AR (Aero) | 200 (188) | +0.080 [+0.057, +0.104] | 5.3 (1.5) | +1.83 [+1.39, +2.31] (88%) | +0.064 [+0.041, +0.090] |
| A diffusion (Aero) | 221 (199) | +0.069 [+0.047, +0.092] | 5.0 (2.2) | +1.54 [+1.07, +2.02] (86%) | +0.054 [+0.032, +0.077] |
| C AR (Aero) | 248 (244) | +0.037 [+0.020, +0.055] | 2.0 (1.1) | +2.22 [+1.81, +2.65] (96%) | +0.041 [+0.024, +0.058] |
| C diffusion (Aero) | 256 (245) | +0.075 [+0.054, +0.098] | 5.7 (1.4) | +1.74 [+1.37, +2.12] (91%) | +0.062 [+0.046, +0.080] |
| B AR (Instruct) | 234 (232) | +0.025 [+0.009, +0.043] | 1.7 (0.7) | +1.24 [+0.93, +1.57] (95%) | +0.006 [-0.013, +0.025] |
| B diffusion (Instruct) | 262 (248) | +0.069 [+0.046, +0.092] | 6.0 (1.5) | +2.00 [+1.62, +2.40] (91%) | +0.046 [+0.025, +0.066] |
| B Fast-dLLM v2 | 263 (252) | +0.067 [+0.049, +0.085] | 3.2 (0.9) | +1.82 [+1.44, +2.22] (92%) | +0.049 [+0.031, +0.068] |

In-vocab substitutions, and novel - in-vocab:

| Run | In-vocab subs (in dict) | sem excess | related (null) | ctx excess | contextual excess | Novel - in-vocab: sem | related | contextual |
|---|---|---|---|---|---|---|---|---|
| B AR (Aero) | 1104 (1104) | +0.063 [+0.053, +0.074] | 10.1 (4.4) | +1.80 [+1.56, +2.06] | +0.041 [+0.031, +0.053] | -0.012 [-0.035, +0.013] | -4.1 [-6.8, -1.1] | -0.006 [-0.030, +0.018] |
| B diffusion (Aero) | 687 (665) | +0.098 [+0.085, +0.112] | 11.6 (2.8) | +2.30 [+2.04, +2.57] | +0.063 [+0.049, +0.078] | -0.024 [-0.048, -0.001] | -5.6 [-9.0, -2.1] | -0.026 [-0.049, -0.002] |
| A AR (Aero) | 483 (471) | +0.093 [+0.076, +0.110] | 12.7 (4.1) | +2.35 [+2.03, +2.68] | +0.058 [+0.039, +0.078] | -0.013 [-0.042, +0.016] | -4.8 [-8.5, -0.8] | +0.006 [-0.023, +0.036] |
| A diffusion (Aero) | 479 (457) | +0.088 [+0.071, +0.104] | 12.0 (2.9) | +1.94 [+1.63, +2.26] | +0.056 [+0.039, +0.073] | -0.018 [-0.046, +0.008] | -6.3 [-10.0, -2.5] | -0.002 [-0.030, +0.027] |
| C AR (Aero) | 963 (960) | +0.073 [+0.062, +0.084] | 11.4 (3.7) | +1.99 [+1.74, +2.24] | +0.044 [+0.032, +0.057] | -0.036 [-0.057, -0.015] | -6.6 [-9.1, -4.0] | -0.004 [-0.025, +0.018] |
| C diffusion (Aero) | 727 (710) | +0.097 [+0.084, +0.110] | 12.8 (3.0) | +2.13 [+1.88, +2.40] | +0.066 [+0.051, +0.081] | -0.021 [-0.047, +0.005] | -5.5 [-8.9, -1.7] | -0.003 [-0.025, +0.019] |
| B AR (Instruct) | 1783 (1780) | +0.043 [+0.036, +0.050] | 9.2 (5.3) | +1.24 [+1.07, +1.41] | +0.022 [+0.014, +0.031] | -0.018 [-0.036, +0.001] | -2.9 [-4.9, -0.8] | -0.016 [-0.036, +0.004] |
| B diffusion (Instruct) | 793 (774) | +0.091 [+0.079, +0.103] | 9.9 (2.8) | +2.01 [+1.76, +2.27] | +0.059 [+0.046, +0.073] | -0.022 [-0.047, +0.003] | -2.6 [-6.0, +1.1] | -0.014 [-0.038, +0.010] |
| B Fast-dLLM v2 | 850 (833) | +0.090 [+0.079, +0.101] | 11.8 (3.7) | +1.93 [+1.71, +2.17] | +0.054 [+0.041, +0.067] | -0.023 [-0.043, -0.002] | -5.8 [-8.6, -2.8] | -0.005 [-0.027, +0.018] |

Paired contrasts of the novel-word excess (a - b; related in pp):

| Contrast | sem | related | ctx | contextual |
|---|---|---|---|---|
| B diff - B AR | +0.022 [-0.006, +0.049] | +1.6 [-1.3, +4.6] | -0.05 [-0.52, +0.42] | +0.003 [-0.023, +0.028] |
| A diff - A AR | -0.010 [-0.037, +0.017] | -1.0 [-4.6, +2.5] | -0.29 [-0.80, +0.20] | -0.010 [-0.039, +0.018] |
| C diff - C AR | +0.038 [+0.014, +0.063] | +3.4 [+0.4, +6.5] | -0.48 [-0.96, -0.01] | +0.022 [+0.001, +0.042] |
| C AR - B AR | -0.014 [-0.038, +0.008] | -0.6 [-3.0, +1.7] | +0.12 [-0.37, +0.62] | +0.006 [-0.017, +0.028] |
| C diff - B diff | +0.002 [-0.023, +0.028] | +1.2 [-2.0, +4.6] | -0.31 [-0.71, +0.10] | +0.025 [+0.003, +0.046] |
| A AR - B AR | +0.029 [-0.004, +0.060] | +2.2 [-1.6, +6.2] | -0.27 [-0.78, +0.25] | +0.029 [-0.003, +0.062] |
| A diff - B diff | -0.004 [-0.031, +0.023] | -0.3 [-3.8, +3.3] | -0.51 [-1.03, +0.03] | +0.016 [-0.009, +0.042] |
| Instruct diff - Instruct AR | +0.043 [+0.018, +0.068] | +3.6 [+0.8, +6.5] | +0.76 [+0.30, +1.22] | +0.040 [+0.013, +0.065] |

## Sentence meaning when a novel word is missed

BERTScore F1 (roberta-large L17, rescaled; results/per_trial/) of unseen-sentence trials where the run missed >= 1 novel word, against unseen trials without a novel word, standardized to the same word-error count (second: word errors x reference length). Negative gap = missing a novel word costs more sentence meaning than other errors of the same count.

| Run | Trials missing a novel word | Mean word errors | F1 there | Matched F1 (errors) | Gap [CI] | Gap, errors x length [CI] | F1, unseen trials without novel words |
|---|---|---|---|---|---|---|---|
| B AR (Aero) | 181 | 3.96 | 0.395 | 0.400 | -0.005 [-0.042, +0.030] | -0.017 [-0.049, +0.014] (98%) | 0.687 |
| B diffusion (Aero) | 184 | 2.67 | 0.438 | 0.455 | -0.015 [-0.048, +0.018] | -0.029 [-0.058, +0.002] (96%) | 0.750 |
| A AR (Aero) | 146 | 2.51 | 0.491 | 0.513 | -0.003 [-0.041, +0.034] | -0.010 [-0.043, +0.024] (90%) | 0.847 |
| A diffusion (Aero) | 169 | 2.28 | 0.498 | 0.496 | +0.003 [-0.039, +0.039] | -0.002 [-0.035, +0.029] (91%) | 0.842 |
| C AR (Aero) | 181 | 3.45 | 0.420 | 0.433 | -0.009 [-0.041, +0.024] | -0.023 [-0.053, +0.008] (98%) | 0.718 |
| C diffusion (Aero) | 181 | 2.81 | 0.411 | 0.440 | -0.028 [-0.061, +0.005] | -0.033 [-0.061, -0.005] (99%) | 0.735 |
| B AR (Instruct) | 193 | 5.15 | 0.323 | 0.346 | -0.023 [-0.055, +0.008] | -0.020 [-0.048, +0.010] (98%) | 0.499 |
| B diffusion (Instruct) | 188 | 2.77 | 0.409 | 0.443 | -0.032 [-0.064, -0.000] | -0.048 [-0.078, -0.016] (96%) | 0.726 |
| B Fast-dLLM v2 | 188 | 2.84 | 0.390 | 0.428 | -0.036 [-0.069, -0.004] | -0.049 [-0.076, -0.020] (97%) | 0.694 |

Paired contrasts of the gap (errors-matched; errors x length in brackets): B diff - B AR -0.010 [-0.046, +0.027] (-0.012 [-0.048, +0.026]); A diff - A AR +0.006 [-0.038, +0.045] (+0.009 [-0.030, +0.043]); C diff - C AR -0.019 [-0.052, +0.014] (-0.009 [-0.046, +0.026]); C AR - B AR -0.004 [-0.035, +0.029] (-0.006 [-0.040, +0.028]); C diff - B diff -0.013 [-0.045, +0.019] (-0.003 [-0.038, +0.029]); A AR - B AR +0.002 [-0.039, +0.044] (+0.007 [-0.033, +0.048]); A diff - B diff +0.018 [-0.026, +0.056] (+0.028 [-0.013, +0.063]); Instruct diff - Instruct AR -0.009 [-0.041, +0.024] (-0.028 [-0.062, +0.006]).

## Examples: novel reference words and each decoder's output there

20 novel tokens drawn at random (seed 0) among those some Aero run gets wrong; the same trials for every run. Novel word in CAPS; * = related in meaning (sem >= 0.4).

| Reference | B AR (Aero) | B diffusion (Aero) | A AR (Aero) | A diffusion (Aero) | C AR (Aero) | C diffusion (Aero) |
|---|---|---|---|---|---|---|
| you actually look forward to foxes or raccoons raiding your GARBAGE | paycheck | cover | story | car | magazine | garden |
| it's a WEAKNESS and all | a-waist | qu | winnie | and | rectangle | w |
| stones with runes on them served as CHECKPOINTS | champions | = (correct) | topics | toys | companies | sports |
| some INTERVIEWS will be a lot more casual | actors | decisions | interchanges | interests | engineers | interactions |
| nobody REMEMBERS him as a loser | (deleted) | programs | = (correct) | = (correct) | goodbye | bedrooms * |
| LAWMAKERS passed a measure last year | lumping | neglect | s | loben | condenses | (deleted) |
| either way they always SPELL it wrong | bring | bring | pull | bear | back | pick |
| one of the biggest creatures in the animal KINGDOM | state | game | conference | camp | county | game |
| i SEW clothes for them every once in a while | (deleted) | still | so | so | saw | saw |
| he SCORED better than we did on the final exam | (deleted) | ignored | ruined | tried | (deleted) | worked |
| because that's PERFECTLY clean | probably | definitely | preferably | basically | probably | probably |
| BLIZZARDS now have a new visual effect | places | lolosures | blissed | pleys | places | liver |
| the beauty of the view STUNNED the young boy | at | starred | stood | stood | town | scored |
| employee CONTROLLER | material | contractor | policy | etiquette | participate | control * |
| eat slowly CHEW well and enjoy the meal | cool | simple | grew | cho | grew | chill |
| the hogs were FED chopped corn and garbage | built | (deleted) | = (correct) | = (correct) | built | found |
| the hogs were fed chopped CORN and garbage | of | shocked | cord | = (correct) | on | buried |
| that would be DEVASTATING i'm sure | always | fascinating | = (correct) | = (correct) | fantastic | frustrating * |
| he's one of the big PROPONENTS of that | brokers | participants | = (correct) | punishment | pitchers | principals * |
| supported according key arrive CRUSH | cups | trash | church | cruise | this | trash |

## Caveats

- Small sample: 284 novel tokens (274 types) in 195 trials, most types once, so novel recall has CIs of several pp and run contrasts on novel words alone are weakly powered; one seed per run, so seed variance is not in the CIs.
- Novel means absent from the T15 training transcripts only. The LLMs know every word from pretraining, and SSL pretraining saw unlabeled neural data; novel recall mixes sub-word signal with the LLM's guess from context. It is a stand-in for held-out phonemes, not a test of them (every phoneme occurs in training).
- Novel words are longer and rarer in English than in-vocab words; the matched gaps adjust for letters and English Zipf frequency (direct standardization), not for position in the sentence, part of speech or proper names, and drop novel tokens in cells without in-vocab words.
- In-vocab words in seen sentences are partly memorized; the frequency bins, matched pools and the in-vocab contrast column use unseen-sentence trials only.
- Recall depends on the alignment (among fewest-edit alignments, the one with the most hits; WER unchanged). For novel words it does not matter here: in every run the aligned-correct count equals the alignment-free count (the novel word appears anywhere in the decode).
- The inflection / new-stem split is a suffix heuristic (-s, -es, -ed, -d, -ing, -'s) with a few false hits (a name like tim / timed); its 65-token side is small.
- Meaning of a word pair: sem is distilbert's decontextualized word vector (centred on every word here, so values differ a little from phonetic_errors.md), related = sem >= 0.4 (calibrated by phonetic_error_analysis.py on common words); a rare novel word is several word pieces averaged, so its vector is noisier. The contextual measure (roberta-large layer 17, BERTScore's) puts the decoded word in the reference slot: it scores fit to the true sentence as much as the word's own meaning, and its null is a few draws per pair (Monte Carlo noise is small next to the bootstrap CI). Only pairs with both words in the CMU dictionary (the null is phoneme-length matched); ctx needs a one-piece decoded word (coverage in the table).
- Sentence meaning: BERTScore F1 (rescaled) from results/per_trial/; trials that miss a novel word are compared with unseen trials without one at the same word-error count, not the same error types or positions; which trials miss a novel word differs by run, so run contrasts of the gap compare different trial sets.
