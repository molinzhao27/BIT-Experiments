# Public human intracortical pretraining data

Inventory date: **2026-10-18**. Scope: metadata and local archive inspection only. No training, GPU use, remote compute, large downloads, or extraction of large data files. Public requests retrieved API JSON, landing-page HTML, source code and small documentation PDFs; one existing 12.7 MB HDF5 member was inspected in memory. Only this document and `configs/pretrain_datasets.yaml` are deliverables.

Sizes are **decimal MB/GB** (10^6 / 10^9 bytes), as in Dryad. Tables give compressed/download sizes unless stated otherwise; the YAML preserves exact bytes and full source SHA-256 values. API checksums do not imply local checksum verification unless stated explicitly. **Unverified** means missing/conflicting evidence, never zero. Channels mean electrodes per feature; TX+SBP doubles the feature dimension. Hours mean recording time, not electrode-hours.

BIT v5 §3.1 reports approximately **98 human hours** and specifies **threshold-crossing-only SSL pretraining**; SBP is added for decoding (§A.3). Using TX+SBP for SSL would therefore be an extension. Its §A.4 permits unlabeled benchmark holdout data in pretraining; preserve split metadata so later work can distinguish that transductive protocol from strict held-out evaluation. [BIT]

The largest uncertainty is **unique usable hours**. No complete duration ledger is exposed by the catalog APIs. We report source-backed figures and unresolved totals separately, rather than treating overlapping exports, padded tensors, or clinical device-use hours as new neural recordings.

## 1. Willett et al. 2021 — handwriting

DOI **10.5061/dryad.wh70rxwmv**. Latest API version **5**, version id **144934**, published **2021-10-15**. License **CC0-1.0**. Version/license: [willett2021_API]; every file byte count and checksum below: [willett2021_FILES]. Content descriptions: [willett2021_README].

| File | Size MB | SHA-256 prefix | SSL use |
|---|---:|---|---|
| `handwritingBCIData.tar.gz` | 1,408.201020 | `f41ff3577f8e` | Use raw neural content |
| `readme.pdf` | 0.288173 | `f5755861a6f3` | Documentation |

**Usable neural content and fields.** In `handwritingBCIData.tar.gz`, use `Datasets/<session>/sentences.mat`, `singleLetters.mat`, and the straight-line experiment's `straightLines.mat`. `neuralActivityTimeSeries` holds time × 192 threshold counts. `neuralActivityCube` and `neuralActivityCube_{character}` are convenience trial views of the same recordings. Use `blockNumsTimeSeries` to respect recording gaps. `numTimeBinsPerSentence` must crop sentence cubes: their trailing bins can contain subsequent sentences, not just padding. `RNNTrainingSteps` contains processed/time-warped duplicates, labels and decoders; `BigramLM` and mouse trajectory templates are not extra neural data. ([W21_README](https://zenodo.org/records/4695519/files/readme.pdf), pp.1–7)

**Acquisition.** Participant **T5**, hand-knob motor cortex/precentral gyrus; **2 × 96 electrodes = 192 channels**, TX only, **−3.5 × RMS**, **10 ms** bins. Sum adjacent pairs within blocks for 20 ms. SBP is not supplied. ([W21_README](https://zenodo.org/records/4695519/files/readme.pdf), pp.2–3; [BIT](https://arxiv.org/html/2511.21740v5) §A.1)

**Extent and splits.** Dryad reports **1,000 sentences, 43,501 characters, 10.7 h**. The README introduction says ten sessions, but its session table lists **11 dates**, comprising **10 sentence/character days plus the 2019-06-03 straight-line day** (overall 2019-05-08 to 2020-01-15). Do not silently discard that eleventh date. There are **33 excluded sentences**; the exclusion flag describes analysis quality, not missing neural recordings. Character/straight-line trial totals and complete continuous-neural hours are **unverified**. Raw `sentenceCondition` identifies open/closed-loop tasks; derived training code supplies train/test partitions, but a single official train/val/test count is **unverified**. ([willett2021_README](https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv), abstract; [W21_README](https://zenodo.org/records/4695519/files/readme.pdf), pp.1,6; [W21_CODE](https://github.com/fwillett/handwritingBCI/blob/main/dataPreprocessing.py))

The small README PDF was obtained from its author-uploaded Zenodo mirror; its full SHA-256 **matches the Dryad `readme.pdf` manifest**, establishing identical documentation. Archive member sizes beyond the outer manifest are unverified because the neural archive was not downloaded.

## 2. Willett et al. 2023 — B2T ’24, T12

DOI **10.5061/dryad.x69p8czpq**. Latest API version **9**, version id **253293**, published **2023-09-01**. License **CC0-1.0**. Version/license: [willett2023_API]; every file byte count and checksum below: [willett2023_FILES]. Content descriptions: [willett2023_README].

| File | Size MB | SHA-256 prefix | SSL use |
|---|---:|---|---|
| `competitionData_readme.txt` | 0.006247 | `63d9da90bddd` | Documentation |
| `competitionData.tar.gz` | 3,669.638707 | `5b5c9784d80d` | Use raw neural content |
| `derived_readme.txt` | 0.004957 | `aa90051845d3` | Documentation |
| `derived.tar.gz` | 8,062.039321 | `44d31076a689` | Skip duplicate representation |
| `diagnosticBlocks_readme.txt` | 0.006126 | `d0821b9950c1` | Documentation |
| `diagnosticBlocks.tar.gz` | 612.391492 | `dc204dda6a67` | Use; extra task |
| `languageModel_5gram.tar.gz` | 38,245.502441 | `bb2b15cc13d4` | Skip language model |
| `languageModel_readme.txt` | 0.001816 | `d4d29bbcd83b` | Documentation |
| `languageModel.tar.gz` | 14,106.240558 | `7b64d36c6476` | Skip language model |
| `README.md` | 0.004413 | `678841c9f31a` | Documentation |
| `sentences_readme.txt` | 0.008416 | `89a78456dbac` | Documentation |
| `sentences.tar.gz` | 14,803.440418 | `d354c97d3fa5` | Optional context; overlaps |
| `tuningTasks_readme.txt` | 0.006514 | `22ab4d03ae92` | Documentation |
| `tuningTasks.tar.gz` | 941.344404 | `8e245a00391f` | Use; extra task |

**Local availability:** `competitionData.tar.gz` is downloaded, SHA-256-verified against the API, and extracted to `data/raw/willett_2023_text/{train,test,competitionHoldOut}/` on the Mac and the GPU host (the `data/competitionData*.tar*.gz` files are redundant browser-download copies).

**Acquisition and fields.** T12 has **4 × 64 = 256 recorded electrodes**: 128 in ventral premotor area 6v and 128 in inferior frontal area 44. Both TX and SBP are supplied at **20 ms**; no rebinning is needed. In `competitionData`, `tx1[0,i]` and `spikePow[0,i]` are sentence-wise time × channel arrays; `sentenceText` and `blockIdx` carry labels and blocks. The standard preprocessing selects columns `0:128` of each, producing **256 features from 128 electrodes**, not 256 electrodes. Exact RMS mappings for `tx1`/`tx2` remain **unverified** here. ([W23_CODE](https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/makeTFRecordsFromSession.py); [W23_TUNING](https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/tuningHeatmaps.ipynb); [willett2023_README](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq))

**Extent and split.** The API/README describes **12,100 sentence trials** across the study; exact trial counts in this corrected competition archive and total hours are **unverified**. Archive listing verifies `train`: **24 files**; labeled development `test`: **24 files**; unlabeled `competitionHoldOut`: **15 files**. BIT §3.1 says **25 sessions**; the complete local archive contains **24 dates**, so the manifest uses 24. The scientific README calls the corrected release “version 4,” whereas the current API version number is **9**; these version labels are not interchangeable. ([willett2023_README](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq); local TAR listing; [BIT](https://arxiv.org/html/2511.21740v5))

**What the other archives add:**

| Archive | Neural content / variables | Incremental SSL value |
|---|---|---|
| `competitionData.tar.gz` | Trialized MAT; `tx1`, `spikePow`; above splits | Core representation, already downloaded |
| `sentences.tar.gz` | Annotated sentence experiments including neural activity and BCI output; exact continuous variable keys **unverified** | Overlaps competition trials; potentially useful delay/context activity. Unique extra hours **unverified**, not a second independent sentence corpus |
| `tuningTasks.tar.gz` | MAT orofacial movement, word and phoneme tuning. `tx2` directly confirmed by official code | Additional tasks suitable for SSL. Examples reference 2022-04-21, 2022-04-26 and 2022-05-03; full session/trial/hour counts **unverified** |
| `diagnosticBlocks.tar.gz` | MAT repeated seven-word plus do-nothing task; `tx2`, `blockNum`, `goTrialEpochs`, `trialCues` confirmed in code | Additional repeated-word recordings. Official analysis selects **12 vocal sessions**; describes **8 repetitions × 8 conditions = 64 trials/block**. These are not audited full-archive totals |
| `derived.tar.gz` | Normalized neural TFRecords (`inputFeatures`, `nTimeSteps`) plus RNN products | Contains usable neural representations, but official generator derives them from `competitionData`; no demonstrated new recordings |
| `languageModel*.tar.gz` | Language model artifacts | No neural SSL value |

Archive purposes: [willett2023_README]. Field names and task examples: [W23_CODE], [W23_TUNING], [W23_DIAGNOSTIC], [W23_DERIVED]. SBP/other threshold fields in the uninspected tuning/diagnostic MATs are not assumed to have a verified key layout. The individual Dryad text READMEs were catalogued, but their payloads were not fetched through blocked download routes.

The corrected README removes a misaligned 2022-06-28 block, fixes two train/test partitions, and corrects holdout overlap. Do not substitute an older archive. `sentences` and competition data must be deduplicated by session/block/trial before combining. [willett2023_README]

**Size flags:** full release **80.441 GB**, well over 20 GB. The **38.246 GB** 5-gram model alone exceeds 20 GB and is unnecessary for SSL. Competition+tuning+diagnostic is only **5.223 GB**; adding sentences brings those four archives to **20.027 GB compressed**, already over the planning threshold before extraction.

## 3. Fan et al. 2023 — handwriting self-recalibration / CORP

DOI **10.5061/dryad.hqbzkh1p6**. Latest API version **4**, version id **263395**, published **2023-11-06**. License **CC0-1.0**. Version/license: [fan2023_API]; every file byte count and checksum below: [fan2023_FILES]. Content descriptions: [fan2023_README].

| File | Size MB | SHA-256 prefix | SSL use |
|---|---:|---|---|
| `CORP_data_release.zip` | 3,567.072761 | `f768e7817bef` | Use raw neural content |
| `README.md` | 0.004402 | `a5f4ae4bd527` | Documentation |

**Usable neural content.** `seed_model_training_data/mat` and `online_evaluation_data/{no_recalibration,recalibration}` hold session MAT files with `tx_feats`, a list/cell of **time × 192** trial arrays, plus `sentences` and `blocks`; online data additionally includes decoder strings. The `tfrecord` representation duplicates the MATLAB data. The ZIP also bundles a seed decoder and language model, which do not add recordings. [fan2023_README]

**Acquisition.** **T5**, left precentral hand knob; **2 × 96 = 192 electrodes**, **20 ms TX**, **−4.5 × RMS**, no supplied SBP. No rebinning required. The supplement's online z-scoring method should not be taken as proof that every released `tx_feats` array is already z-scored; inspect payload values later. ([fan2023_README](https://datadryad.org/dataset/doi:10.5061/dryad.hqbzkh1p6); [FAN_SUPP](https://proceedings.neurips.cc/paper_files/paper/2023/file/83a14a36de4502bac5b580db36e81858-Supplemental-Conference.pdf) §2.4; [H2_API](https://api.dandiarchive.org/api/dandisets/000950/versions/0.241029.1403/))

**Extent and split.** **21 public sessions**: **11 seed** plus **10 online evaluation**. The study collected **26**, including five later evaluation sessions not in this Dryad release. Supplement Table 2 covers 2022-05-18 to 2023-10-09; released sessions end 2023-02-28. Summing the table gives **770** collected seed sentences and **667** sentences in the first ten assessment days, **1,437 including warmups**. That is a study-collection tally, **not a verified released MAT trial count**: the README describes two evaluation block categories and does not establish warmup inclusion. Total usable hours and released train/val/test trial counts are **unverified**. ([fan2023_README](https://datadryad.org/dataset/doi:10.5061/dryad.hqbzkh1p6); [FAN_SUPP](https://proceedings.neurips.cc/paper_files/paper/2023/file/83a14a36de4502bac5b580db36e81858-Supplemental-Conference.pdf) §2.2 and Table 2)

**Overlap.** FALCON H2 repackages a subset of these same T5 sessions. Its five later calibration days add only tiny excerpts after this release. Willett 2021 also concerns T5, but different recording dates; share participant identity while retaining session identity. Do not count MAT/TFRecord copies or H2 held-in calibration as independent hours. ([H2_API](https://api.dandiarchive.org/api/dandisets/000950/versions/0.241029.1403/); [H2_CODE](https://github.com/snel-repo/falcon-challenge/blob/main/preproc/h2_preproc.py))

## 4. Karpowicz et al. 2024 — FALCON H2

**DANDI 000950**, latest published version **0.241029.1403**, DOI **10.48324/dandi.000950/0.241029.1403**. Draft has the same **47 assets / 1,224,956,736 bytes**, but pin the published version. License **CC BY 4.0**. Version/license/content: [H2_API]; exact file bytes and SHA-256 below: [H2_FILES].

The path prefixes below encode **splits**, not separate people: all data come from **T5**, despite DANDI's `numberOfSubjects: 3`. All files are `.nwb` (HDF5), containing **192 TX channels**, **2 × 96 arrays**, left precentral hand knob, **20 ms**. Neural values are `acquisition/binned_spikes/data`; `acquisition/eval_mask/data` and `intervals/trials/{cue,block_num,start_time,stop_time}` carry masks and trial metadata. No SBP is documented. ([H2_API](https://api.dandiarchive.org/api/dandisets/000950/versions/0.241029.1403/); [H2_CODE](https://github.com/snel-repo/falcon-challenge/blob/main/preproc/h2_preproc.py))

**Splits and time:** **21 held-in-calib + 21 held-in-minival + 5 held-out-calib files**, representing **26 unique dates**. Minival is a two-trial subset of calibration; omit it from SSL accumulation. The paper reserves 40% of trials for evaluation, releases the remaining held-in calibration portion, and releases only **3 trials per held-out day** (**15 held-out calibration trials**). Full evaluation/oracle files are absent from this published manifest. Full public trial count and hours are **unverified**. Paper §A.3.6 states 7–45 calibration minutes per held-in session and 1–2 per held-out session: this yields only a broad **2.53–15.92 h bound**, not an estimate of unique additional data. ([H2_PAPER](https://papers.nips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf) §A.3.6; [H2_CODE](https://github.com/snel-repo/falcon-challenge/blob/main/preproc/h2_preproc.py))

The converter creates **synthetic 10-second gaps between trials**. Later duration accounting must use neural row counts × 0.020 s rather than timestamp span. After downloading Fan, prioritize only the **five held-out-calib files, 18.813 MB**; held-in calibration is an alternative representation of the overlapping data. Exact correspondence should still be checked when loading. [H2_CODE]

| File | Size MB | SHA-256 prefix | SSL use |
|---|---:|---|---|
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220518.nwb` | 49.922021 | `c9346c5e31de` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220523.nwb` | 84.807875 | `de5846396e51` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220525.nwb` | 91.034941 | `4129284c4e68` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220601.nwb` | 85.840150 | `9ba0539fdb7f` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220603.nwb` | 58.897380 | `e4ea3aad4d59` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220606.nwb` | 105.989592 | `591676e3a0ee` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220608.nwb` | 66.344109 | `b8cf69260fac` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220613.nwb` | 84.965145 | `be2931d49d1a` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220615.nwb` | 65.648065 | `5a88c6b12ea5` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220622.nwb` | 73.296905 | `1ac337f1faeb` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220901.nwb` | 43.713344 | `cf6e150a3e64` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20220929.nwb` | 39.452476 | `5ecc24bbccfb` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221006.nwb` | 38.858231 | `fdf164808426` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221018.nwb` | 32.711681 | `d22def1f0b30` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221025.nwb` | 29.314201 | `945943015c1e` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221027.nwb` | 29.486116 | `2c9bc0f8cc29` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221101.nwb` | 31.772036 | `365154fe9e00` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221103.nwb` | 35.419931 | `84e3befcae29` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221208.nwb` | 36.141346 | `fa1dfc8dd869` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20221215.nwb` | 17.448705 | `24d0507c1fcd` | Alternative to Fan; overlaps |
| `sub-T5-held-in-calib/sub-T5-held-in-calib_ses-20230228.nwb` | 34.683601 | `9e66432c8e9b` | Alternative to Fan; overlaps |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220518.nwb` | 4.296815 | `d0f7b60e60c7` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220523.nwb` | 1.226680 | `ee4c05afdf27` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220525.nwb` | 3.552635 | `a2eab830cf01` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220601.nwb` | 2.580020 | `8710e117b559` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220603.nwb` | 2.997640 | `58b53bb6975d` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220606.nwb` | 4.295245 | `49e5ea86833c` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220608.nwb` | 6.060710 | `917feaf8a163` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220613.nwb` | 4.168860 | `aae60018d0b6` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220615.nwb` | 4.131180 | `978baab8d489` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220622.nwb` | 2.959960 | `fca7b02cdd84` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220901.nwb` | 3.014910 | `48ec9c3de77f` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20220929.nwb` | 3.982815 | `830d807afdd2` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221006.nwb` | 2.600430 | `9c795104f1b0` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221018.nwb` | 2.762925 | `3814d938a464` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221025.nwb` | 2.751150 | `a3f9f7af2706` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221027.nwb` | 2.842995 | `6d96da646a6a` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221101.nwb` | 2.848490 | `9542354408d7` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221103.nwb` | 2.317045 | `b58506c4ca37` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221208.nwb` | 3.760660 | `40eefe1dbf43` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20221215.nwb` | 2.993715 | `f7401df801c8` | Skip duplicate subset |
| `sub-T5-held-in-minival/sub-T5-held-in-minival_ses-20230228.nwb` | 4.251285 | `30f195dd7fba` | Skip duplicate subset |
| `sub-T5-held-out-calib/sub-T5-held-out-calib_ses-20230417.nwb` | 3.312425 | `0de25c2b32e2` | Use after Fan |
| `sub-T5-held-out-calib/sub-T5-held-out-calib_ses-20230531.nwb` | 3.499255 | `bc604bbbad9e` | Use after Fan |
| `sub-T5-held-out-calib/sub-T5-held-out-calib_ses-20230628.nwb` | 3.684515 | `5bce73e437eb` | Use after Fan |
| `sub-T5-held-out-calib/sub-T5-held-out-calib_ses-20230816.nwb` | 3.525945 | `b5518f4f90e1` | Use after Fan |
| `sub-T5-held-out-calib/sub-T5-held-out-calib_ses-20231009.nwb` | 4.790580 | `3d0c469b2d71` | Use after Fan |

Summed groups: held-in-calib **1,135.748 MB**; redundant minival **70.396 MB**; held-out-calib **18.813 MB**. All non-minival files total **1.155 GB** if using H2 as a standalone alternative to Fan. Asset-level SHA-256 comes directly from `metadata.digest["dandi:sha2-256"]`; it is not the DANDI multipart ETag. [H2_FILES]

## 5. Card et al. 2024 — B2T ’25, T15

DOI **10.5061/dryad.dncjsxm85**. Latest API version **5**, version id **374060**, published **2025-07-03**. License **CC0-1.0**. Version/license: [card2024_API]; every file byte count and checksum below: [card2024_FILES]. Content descriptions: [card2024_README].

| File | Size MB | SHA-256 prefix | SSL use |
|---|---:|---|---|
| `README.md` | 0.003183 | `3516e6c4c20c` | Documentation |
| `t15_copyTask_neuralData.zip` | 11,051.034483 | `fb828fa7ef93` | Use; already local |
| `t15_copyTask.pkl` | 57.786348 | `7313b9a11027` | Not neural / not selected |
| `t15_personalUse.pkl` | 1.132664 | `1d18d59e6ee4` | Not neural / not selected |
| `t15_pretrained_rnn_baseline.zip` | 484.944889 | `04ab2275eeb4` | Skip weights |

**Already local:** `data/t15_copyTask_neuralData.zip`, **11.051 GB**, with **45 session directories** and **127 HDF5 files**. The ZIP member lengths sum to **12,916,863,318 bytes**. The archive-plus-extracted footprint is therefore about **23.968 GB**, over 20 GB even though the download alone is not. A second `data/t15_copyTask_neuralData2.zip` has the same size/member listing; full byte identity was not recomputed, so it is not counted again. No files were removed.

**Layout and variables.** `hdf5_data_final/<session>/data_{train,val,test}.hdf5` contains `trial_XXXX/input_features`, **time × 512 float32**: **256 TX + 256 SBP**. Trial attributes include `n_time_steps`, `session`, `block_num`, `trial_num`, and labels where available. This was checked in the smallest local member, `t15.2023.10.20/data_test.hdf5` (**12,704,320 bytes, 9 trial groups**, first feature shape **1109 × 512**), read in memory only. The feature ordering is confirmed by the BIT loader. ([T15_LOADER](https://github.com/yzhang511/bit/blob/main/data/brandman_2024_text/prepare_data.py); local ZIP/HDF5 metadata)

**Acquisition.** **T15**, **4 × 64 = 256 electrodes**, left precentral gyrus (inferior 6v, M1/area 4, area 55b, superior 6v in local notes), **20 ms**, TX threshold **−4.5 × RMS**. Features are **already z-scored using the preceding 20 trials**, so neither rebinning nor a second presumed raw-data normalization is needed. ([card2024_README](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85); [KUNZ_NOTES](../../kunz/docs/DATA_NOTES.md))

**Extent and splits.** **45 sessions spanning 20 months**, 2023-08-11 to 2025-04-13 by ZIP directory. There are separate train/val/test files, but not all sessions contain all three. Test trials omit labels. The official description CSV sums to **8,234 Train + 2,928 Val/Test = 11,162 sentences**; BIT §3.1 reports **10,948**, and Dryad says **11,000+**. These are unresolved source differences: **actual total HDF5 trial groups and recording hours are unverified**, not silently equated to either count. ([T15_CSV](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/data/t15_copyTaskData_description.csv); [card2024_README](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85); [BIT](https://arxiv.org/html/2511.21740v5))

**Excluded files.** `t15_copyTask.pkl` documents **1,718 evaluation trials** with logits/decoded outputs, labels and durations; `t15_personalUse.pkl` documents **22,126 conversation summaries**, not neural feature matrices. Neither is selected for neural SSL. The baseline ZIP is model weights. Clinical conversation-use hours are not copy-task neural hours and are not added to this inventory. [card2024_README]

Local split file counts: **train: 45**, **val: 41**, **test: 41**. Source: local ZIP central directory.

## 6. Kunz et al. 2025 — inner speech and other tasks

DOI **10.5061/dryad.gf1vhhn1j**. Latest API version **2**, version id **379376**, published **2025-07-30**. License **CC0-1.0**. Version/license: [kunz2025_API]; every file byte count and checksum below: [kunz2025_FILES]. Content descriptions: [kunz2025_README].

| File | Size MB | SHA-256 prefix | SSL use |
|---|---:|---|---|
| `chittychittybangbang.zip` | 1,106.682880 | `67d62e3c2b7c` | Use; already local |
| `conjunctiveCounting.zip` | 322.390416 | `be5fa91b8a8f` | Use; already local |
| `interleavedVerbalBehaviors.zip` | 777.419058 | `19f90f09f2ea` | Use; already local |
| `isolatedVerbalBehaviors.zip` | 3,525.037526 | `40bf63f0b824` | Use; already local |
| `README.md` | 0.007035 | `cece9d44f903` | Documentation |
| `sentenceData.zip` | 3,932.159336 | `a92c1e47f878` | Use; already local |
| `seqRecallSpeech.zip` | 79.443728 | `077f44c2ffcd` | Use; already local |
| `seqRecallUninstructed3ElementArrow.zip` | 364.978730 | `4be73e065ad8` | Use; already local |
| `seqRecallUninstructed3ElementLines.zip` | 371.089478 | `524e144b3e90` | Use; already local |
| `seqRecallUninstructedSingleArrow.zip` | 38.768898 | `27f56c7a6edc` | Use; already local |
| `seqRecallVerbalMemory.zip` | 175.492315 | `f5aca99cb74a` | Use; already local |
| `seqRecallVisualMemory.zip` | 105.176285 | `1d5a9fddd830` | Use; already local |

**All eleven data ZIPs are already present in `data/raw/`.** Local sizes match the current API; prior SHA verification is recorded in [KUNZ_NOTES], but ZIP payloads were not all rehashed in this inventory. The manifest includes **69 MAT files** across **21 unique participant-date pairs** (not the sum of per-archive session counts). All eleven archives document neural features usable for SSL, including the cognitive and motor-control tasks. [local ZIP listings; archive READMEs]

**Feature layout.** `binnedTX` and `spikePow` are time × electrode matrices. SBP is mean squared voltage after >250 Hz high-pass filtering and linear-regression referencing. `binSize`, `chanSets`, `chanSetNames`, `blockNum`, `goTrialEpochs`, `delayTrialEpochs` describe sampling, electrodes, blocks and trials. Sentence/keyword files additionally supply `sentenceFeatures`, a go-period-only concatenation `[TX, SBP]` reconstructible from the continuous variables; it is not extra data. `sentenceDurations` is in bins. Sequence-recall READMEs state that **both TX and SBP have already had block-wise means subtracted**. [archive READMEs]

| Participant | Released electrodes / physical arrays in inspected isolated+interleaved data | Areas | Isolated / interleaved bins |
|---|---|---|---|
| T12 | 128 / 2 × 64 | i6v, s6v | 20 / 20 ms |
| T15 | 256 / 4 × 64 | i6v, M1, 55b, s6v | 10 / 10 ms |
| T16 | 256 / 4 × 64 | PEF, ventral6v/i6v, 6d (two arrays combined) | 10 / 20 ms |
| T17 | 256 / 4 × 64 | s6v-v, s6v-d, 55b (two arrays combined) | 10 / 10 ms |

Source: [KUNZ_NOTES] and [KUNZ_COUNTS]. Feature widths double these electrode counts when combining TX+SBP. These are **saved-channel maps**, not implanted totals: T12's area-44 arrays and T17's 6d arrays are absent from these inspected exports. Channel counts/area maps of each unopened MAT payload remain **unverified**. Read `chanSets` rather than assuming all sessions share a channel order.

For 10 ms raw recordings, aggregate adjacent TX counts by sum and SBP by mean to 20 ms **within blocks**. Do not bridge recording gaps. Every sequence-recall filename says `20ms`; this is filename evidence, while the generic README says 10/20 ms, so its actual `binSize` scalar remains **unverified** until loaded. Sentence/keyword/counting files require their own `binSize` check; do not extrapolate from participant alone.

**Counts and hours already established locally:**

| Subset | Participant sessions / unique participant-days | Trials, including do-nothing | Recorded hours |
|---|---:|---:|---:|
| Isolated (`allBehaviors` only, counted once) | 4 / 4 | 4,664 | 5.850683 |
| Interleaved | 4 / 4 | 2,069 | 2.933936 |
| Union of those subsets | 8 task-session records / 7 days | 6,733 | **8.784619** |

Computed from existing [KUNZ_COUNTS]: `sum(n_bins * bin_ms) / 3_600_000`. This includes recorded within-block context, not only go epochs. T17's isolated and interleaved tasks share 2024-12-09 but use distinct blocks. The four `allBehaviors` files duplicate the 28 per-behavior files; use one representation. Full-release trials/hours, including the other archives below, are **unverified**. `*_eval.mat` marks real-time evaluation in sentence/keyword tasks; no release-wide official train/val/test partition is documented. ([KUNZ_NOTES](../../kunz/docs/DATA_NOTES.md); archive READMEs)

**Other archives: actual neural members and tasks.** Listed sizes below are uncompressed member sizes from ZIP central directories, with no large payload extraction. Each archive's internal README confirms `binnedTX` and `spikePow`. Counts are unique participant-date pairs within the archive; shared dates across archives must not be added as new days.

### `chittychittybangbang.zip`

T12 inner-speech keyword-gated sentence BCI, with/without keyword and real-time evaluation. **4 MAT files, 3 participant-days**, participants T12. Trials and hours: **unverified**. README: `chittychittybangbang/chittychittybangbang_readme.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t12.2024.12.19_chittychittybangbang_50words.mat` | 271.211857 |
| `t12.2024.12.06_chittychittybangbang_50words.mat` | 399.993862 |
| `t12.2024.12.19_chittychittybangbang_50words_eval.mat` | 162.880626 |
| `t12.2024.12.17_chittychittybangbang_50words.mat` | 278.801277 |

### `conjunctiveCounting.zip`

T15/T16 silently count shape-color conjunctions then vocalize the count. **2 MAT files, 2 participant-days**, participants T15, T16. Trials and hours: **unverified**. README: `conjunctiveCounting/conjunctiveCounting_readme.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t16.2024.12.18_conjunctive_counting.mat` | 46.693047 |
| `t15.2024.12.15_conjunctive_counting.mat` | 276.976722 |

### `sentenceData.zip`

Attempted/imagined sentence production; 50-word and large-vocabulary conditions; open-loop and *_eval real-time evaluation files. **15 MAT files, 6 participant-days**, participants T12, T15, T16. Trials and hours: **unverified**. README: `sentenceData/sentences_readme.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t16.2024.12.18_imagined_125kwords_eval.mat` | 79.796567 |
| `t15.2025.01.05_imagined_125kwords.mat` | 538.430004 |
| `t15.2024.03.03_imagined_50words_eval.mat` | 165.114179 |
| `t15.2025.01.05_imagined_125kwords_eval.mat` | 392.014346 |
| `t12.2023.11.28_imagined_50words_eval.mat` | 53.789261 |
| `t16.2024.04.29_attempted_50words.mat` | 105.770556 |
| `t15.2024.12.15_imagined_125kwords_eval.mat` | 90.311172 |
| `t15.2024.12.15_imagined_125kwords.mat` | 818.097296 |
| `t12.2023.11.28_imagined_50words.mat` | 69.016628 |
| `t16.2024.04.29_imagined_50words_eval.mat` | 35.387995 |
| `t15.2024.03.03_imagined_50words.mat` | 718.057376 |
| `t16.2024.04.29_imagined_50words.mat` | 89.287188 |
| `t16.2024.12.18_imagined_125kwords.mat` | 204.976159 |
| `t12.2023.11.28_attempted_50words.mat` | 47.377469 |
| `t15.2024.03.03_attempted_50words.mat` | 542.337626 |

### `seqRecallSpeech.zip`

T12/T16 attempted direction-word speech; T12 sequences, T16 single words. **2 MAT files, 2 participant-days**, participants T12, T16. Trials and hours: **unverified**. README: `seqRecallSpeech/README.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t12.2023.09.28_binnedTX_spikePow_20ms_speak.mat` | 87.207800 |
| `t16.2024.07.17_binnedTX_spikePow_20ms.mat` | 42.923144 |

### `seqRecallUninstructed3ElementArrow.zip`

T12/T16 arrow-cued movement sequences that may elicit spontaneous inner speech. **2 MAT files, 2 participant-days**, participants T12, T16. Trials and hours: **unverified**. README: `seqRecallUninstructed3ElementArrow/README.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t12.2023.05.16_binnedTX_spikePow_20ms_joystickSeq.mat` | 434.128800 |
| `t16.2024.04.08_binnedTX_spikePow_20ms.mat` | 114.320480 |

### `seqRecallUninstructed3ElementLines.zip`

T12/T16 visually cued line-sequence control. **2 MAT files, 2 participant-days**, participants T12, T16. Trials and hours: **unverified**. README: `seqRecallUninstructed3ElementLines/README.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t12.2023.09.19_binnedTX_spikePow_20ms_lineSequenes.mat` | 442.242696 |
| `t16.2024.04.08_binnedTX_spikePow_20ms_lines.mat` | 114.316640 |

### `seqRecallUninstructedSingleArrow.zip`

T12 single-arrow joystick control. **1 MAT files, 1 participant-days**, participants T12. Trials and hours: **unverified**. README: `seqRecallUninstructedSingleArrow/README.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t12.2023.05.16_binnedTX_spikePow_20ms_joystickSingle.mat` | 50.948696 |

### `seqRecallVerbalMemory.zip`

T12/T16 movement-sequence recall with instructed verbal/inner-speech strategy. **3 MAT files, 3 participant-days**, participants T12, T16. Trials and hours: **unverified**. README: `seqRecallVerbalMemory/README.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t12.2023.09.28_binnedTX_spikePow_20ms_arrow.mat` | 92.433000 |
| `t16.2024.07.17_binnedTX_spikePow_20ms_verbal.mat` | 114.320480 |
| `t12.2023.10.12_binnedTX_spikePow_20ms_verbalMemory.mat` | 92.481640 |

### `seqRecallVisualMemory.zip`

T12/T16 movement-sequence recall with instructed visual strategy. **2 MAT files, 2 participant-days**, participants T12, T16. Trials and hours: **unverified**. README: `seqRecallVisualMemory/README.txt` within the local archive.

| MAT member (relative to archive task folder) | Size MB |
|---|---:|
| `t16.2024.07.17_binnedTX_spikePow_20ms_visual.mat` | 114.320480 |
| `t12.2023.10.12_binnedTX_spikePow_20ms_visualMemory.mat` | 92.555880 |

The sentence archive contains **T12, T15 and T16**, not just the T12/T15 subset used for BIT imagined-speech decoding. Keyword trials contribute three T12 dates. Counting contributes T15/T16 recordings. Sequence-recall data includes non-speech controls as well as instructed and uninstructed inner speech; these remain useful candidate motor-cortex SSL data. Session IDs and all neural member byte counts are preserved in the YAML. The original readmes sometimes use inconsistent figure numbers/task wording, so task descriptions above follow the actual experimental instructions and filenames.

## Summary and recommended acquisition order

| Dataset | All catalogued files GB | Selected SSL download GB | Sessions/days | Hours supported here |
|---|---:|---:|---|---|
| Willett et al. 2021 — handwriting | 1.408489 | 1.408201 | 11 dates / 10 sentence days | 10.7 reported sentence corpus; full raw total unverified |
| Willett et al. 2023 — B2T 2024, T12 | 80.440636 | 5.223375 | 24 | Unverified |
| Fan et al. 2023 — handwriting self-recalibration (CORP) | 3.567077 | 3.567073 | 21 | Unverified |
| Karpowicz et al. 2024 — FALCON H2 | 1.224957 | 0.018813 | 26 dates / 47 assets | Unverified; broad public-calibration bound 2.53–15.92, overlaps Fan |
| Card et al. 2024 / B2T 2025 — T15 copy task | 11.594902 | 11.051034 | 45 | Unverified |
| Kunz et al. 2025 — inner speech and cognitive control tasks | 10.798646 | 10.798639 | 21 | 8.784619 verified subset; full release unverified |
| **Total** | **109.034706** | **32.067134** | Do not sum overlapping releases | **19.484619 h source-covered subtotal + unverified remainder** |

“Selected” means neural archive choices in `needed_for_ssl: true`: T15 neural ZIP, all Kunz ZIPs, T12 competition+tuning+diagnostic, Willett handwriting, Fan CORP, and only H2's five later calibration files after Fan. It excludes documentation bytes, redundant mini-validation, overlapping H2 held-in data, optional `sentences`, duplicate `derived`, language models and pretrained weights. It is still subject to block/trial deduplication, especially across studies sharing participants.

**Hours are not a verified 98-hour sum.** The source-covered subtotal is 10.7 reported handwriting hours + 8.784619 locally inventoried Kunz hours. It is not a full-corpus estimate, and it mixes a repository-reported sentence duration with continuous within-block duration. BIT's **~98 h** is the best cited overall reference for its human pretraining corpus, but reproducing that total from these releases is **unverified**, particularly because Fan/H2 and multiple export formats overlap. Do not add the broad H2 duration bound to Fan. ([BIT](https://arxiv.org/html/2511.21740v5) §3.1; [willett2021_README](https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv); [KUNZ_COUNTS](../../kunz/results/phase0/inventory.json))

**Recommended order:**

1. **Use the already downloaded B2T ’25 and Kunz archives first**: **21.850 GB compressed**. Start with T15 and deduplicated Kunz isolated/interleaved; sentence, keyword and cognitive archives can follow without network transfer. Preserve evaluation and block identities.
2. **B2T ’24 `competitionData.tar.gz` (3.670 GB)** is next scientifically, but is **already downloaded and verified** (`data/raw/willett_2023_text/competitionData.tar.gz`, SHA-256 matches; extracted on the Mac and the GPU host). Do not download again. Current selected local coverage totals **25.519 GB**.
3. **Willett 2021** (`handwritingBCIData.tar.gz`, **1.408 GB**) adds T5 handwriting, characters and straight-line tasks; rebin 10→20 ms and use unwarped series.
4. **Fan CORP** (**3.567 GB**) adds newer T5 sessions. Then obtain only **H2's five held-out calibration NWBs (18.813 MB)** for the later dates. If using H2 instead of CORP, download non-minival H2 (**1.155 GB**) but recognize that it contains less data than full CORP and does not replace the full 21-day release.
5. **T12 `diagnosticBlocks` (0.612 GB) and `tuningTasks` (0.941 GB)** add distinct tasks for **1.554 GB** combined. Steps 3–5 require **6.548 GB additional transfer** given the current local files.
6. **Optional last:** `sentences.tar.gz` (**14.803 GB**) only if continuous context is needed and overlap can be resolved. Skip `derived` and both language models for this encoder inventory.

**20 GB flags:** the complete manifest is **109.035 GB**, mostly inflated by T12 language-model/derived archives. Core selected archives before extra T12 tasks total **30.513 GB**; selected including those tasks totals **32.067 GB**, and adding optional sentences raises it to **46.871 GB**. These totals exceed 20 GB cumulatively even though the recommended individual downloads do not. T15 archive+extraction alone is **23.968 GB**; all eleven Kunz ZIPs plus all listed non-resource-fork contents are also over **20 GB** before deduplicating exports. T12 competition+sentences+tuning+diagnostic is **20.027 GB compressed**. Plan storage from member sizes, not only ZIP size. Unknown expanded sizes are **unverified**, not assumed to equal compressed size.

**Remaining verification gaps before implementation:** raw trials and summed bin durations for uninspected archives; precise T12 threshold mappings; T15 actual HDF5 trial count versus CSV/BIT; channel/bin scalars in the additional Kunz MAT files; cross-release block/trial duplicates. Later duration accounting should use actual valid neural rows, clip trial views correctly, and avoid both synthetic timestamp gaps and repeated feature representations. This inventory does not load/train an encoder or make those later processing choices.

## Sources and reproducibility

Every remote source was accessed on 2026-10-18. Dryad `stash:version` was followed from each dataset API, then `/files?per_page=100` was read; all responses fit one page (2, 14, 2, 5 and 12 files respectively). DANDI returned all 47 assets in one page with metadata and SHA-256. The API version is authoritative when a README's release label differs.

Local evidence: ZIP central-directory filenames and `file_size`; small internal README text; one small HDF5 member; complete streaming TAR listing and SHA-256 of the finished T12 archive; existing `kunz/docs/DATA_NOTES.md` and `kunz/results/phase0/inventory.json`. No local dataset files were rewritten. Full Kunz SHA values are source manifest values; local prior verification is attributed explicitly rather than rerun.

The paired [YAML manifest](../../configs/pretrain_datasets.yaml) contains the full file list, full checksums, local availability, field names, split details, provenance IDs and unknown-value markers. Selected totals are computed by summing exact `size_bytes` where `needed_for_ssl` is true. The Markdown intentionally uses rounded GB in narrative; file tables retain bytes to six decimal MB digits.

[BIT]: https://arxiv.org/html/2511.21740v5

[W21_README]: https://zenodo.org/records/4695519/files/readme.pdf

[W21_CODE]: https://github.com/fwillett/handwritingBCI/blob/main/dataPreprocessing.py

[W23_CODE]: https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/makeTFRecordsFromSession.py

[W23_TUNING]: https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/tuningHeatmaps.ipynb

[W23_DIAGNOSTIC]: https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/neuralCorrelationAcrossDays.ipynb

[W23_DERIVED]: https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/rnn_step1_makeTFRecords.ipynb

[FAN_SUPP]: https://proceedings.neurips.cc/paper_files/paper/2023/file/83a14a36de4502bac5b580db36e81858-Supplemental-Conference.pdf

[H2_API]: https://api.dandiarchive.org/api/dandisets/000950/versions/0.241029.1403/

[H2_FILES]: https://api.dandiarchive.org/api/dandisets/000950/versions/0.241029.1403/assets/?page_size=100&metadata=true

[H2_CODE]: https://github.com/snel-repo/falcon-challenge/blob/main/preproc/h2_preproc.py

[H2_PAPER]: https://papers.nips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf

[T15_CSV]: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/data/t15_copyTaskData_description.csv

[T15_LOADER]: https://github.com/yzhang511/bit/blob/main/data/brandman_2024_text/prepare_data.py

[KUNZ_NOTES]: ../../kunz/docs/DATA_NOTES.md

[KUNZ_COUNTS]: ../../kunz/results/phase0/inventory.json

[willett2021_API]: https://datadryad.org/api/v2/datasets/10.5061%2Fdryad.wh70rxwmv

[willett2021_FILES]: https://datadryad.org/api/v2/versions/144934/files?per_page=100

[willett2021_README]: https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv

[willett2023_API]: https://datadryad.org/api/v2/datasets/10.5061%2Fdryad.x69p8czpq

[willett2023_FILES]: https://datadryad.org/api/v2/versions/253293/files?per_page=100

[willett2023_README]: https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq

[fan2023_API]: https://datadryad.org/api/v2/datasets/10.5061%2Fdryad.hqbzkh1p6

[fan2023_FILES]: https://datadryad.org/api/v2/versions/263395/files?per_page=100

[fan2023_README]: https://datadryad.org/dataset/doi:10.5061/dryad.hqbzkh1p6

[card2024_API]: https://datadryad.org/api/v2/datasets/10.5061%2Fdryad.dncjsxm85

[card2024_FILES]: https://datadryad.org/api/v2/versions/374060/files?per_page=100

[card2024_README]: https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85

[kunz2025_API]: https://datadryad.org/api/v2/datasets/10.5061%2Fdryad.gf1vhhn1j

[kunz2025_FILES]: https://datadryad.org/api/v2/versions/379376/files?per_page=100

[kunz2025_README]: https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j
