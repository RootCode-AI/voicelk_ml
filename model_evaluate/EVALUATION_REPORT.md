# VoiceLK Text-to-Speech — Model Evaluation Report

**Project:** VoiceLK — Sinhala/English code-switched TTS for G.C.E. O/L ICT content  
**Report generated:** 19 September 2026, 04:16  
**Evaluation run:** 2026-09-18 10:51:51  
**Systems evaluated:** 3  
**Source notebook:** `model_evaluate/model_evaluation_full_suite.ipynb`

---

## Executive summary

- **All systems synthesise faster than real time on CPU.** End-to-end real-time factor ranges 0.36–0.51; System A is fastest at 0.36 (i.e. one second of speech costs 0.36 s of compute).
- **Spectral accuracy on unseen sentences is 11.5 dB MCD at best** (System B), measured against the run's own held-out recordings. Lower is better; see §2.2 for the exact definition and the calibration anchors.
- **The checkpoint selected by validation loss is not the best-sounding one.** System A: `best_model.pth` (10.37 dB) vs `checkpoint_70000.pth` (9.45 dB). Coqui's `best_model.pth` is chosen on loss, which does not track spectral distortion — §4 gives the full sweep.
- **The duration predictor is close to collapsed in both custom runs.** 83% of input tokens receive the minimum one-frame duration (16 ms), against 36% for the best system. This is the most likely cause of rushed, flat-sounding output and is the single highest-value fix (§3.7).
- **System C silently discards phonemes it was never trained on** — 4.8% of the IPA characters produced by the front-end are dropped by the tokenizer without any error (`ɪ`, `ə`, `ˈ`, `w`, `z`, `ʊ`, `ɯ`, `ɛ`). These are almost entirely English phonemes, so code-switched input degrades invisibly (§5.2).
- **System C carries a historical data-pipeline defect.** Its config declares 234 speakers for a corpus with one, and the same parsing bug truncated about 31% of its training transcripts. The checkpoint's numbers are therefore a measurement of that specific artefact, not of the corpus or the architecture — §1.5 derives this from the metadata.
- **The pronunciation lexicon is applied reliably but covers little of the corpus.** 98.9% of English and 94.8% of Sinhala lexicon entries are reproduced exactly end-to-end, yet the lexicon only covers a single-digit percentage of dataset tokens (§5.3–§5.4).
- **No human listening results yet.** MOS, CMOS, AB/ABX, MUSHRA, the transcription test, the Turing-style test and the ICT comprehension quiz are fully prepared (blinded stimuli, hidden keys, browser test page, scoring scripts) but not yet administered — §7. Until they are, every quality claim in this report rests on objective proxies.

### Evaluation environment

All figures in this report were produced on one machine, so latency and real-time-factor
numbers are comparable between systems even though the systems were *trained* in different
environments (§1.4).

| property | value |
|---|---|
| Platform | Windows-11-10.0.26200-SP0 |
| CPU | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Logical cores | 8 |
| Inference device | cpu |
| GPU | none (CPU inference) |
| torch threads | 4 |
| Python | 3.14.7 |
| PyTorch | 2.14.0+cpu |
| librosa / numpy / pandas | 1.0.0 / 2.5.2 / 3.0.5 |
| Held-out reference pairs | 2 |
| Training-split reference pairs | 3 |
| RTF repeats per sentence | 1 |


---

## 1. Systems under evaluation

### 1.1 Identification

Systems are referred to by the labels below throughout the report.

| label | run folder | config run_name | checkpoint evaluated |
|---|---|---|---|
| System A | custom | voicelk_vits_custom | best_model.pth |
| System B | voicelk_vits_custom-September-08-2026_05+59AM-fd04a0a | voicelk_vits_custom | best_model_63079.pth |
| System C | voicelk_vits_pathnirwana-September-04-2026_07+22PM-722d5af | voicelk_vits_pathnirwana | best_model.pth |

### 1.2 Architecture and audio configuration

All three are VITS (end-to-end, adversarially trained, with a stochastic duration
predictor). They differ in the dataset, the sample rate, the phoneme vocabulary and
whether a speaker-embedding table exists at all — which bounds what can be compared
between them.

| system | sample_rate | mel_fmax | vocab slots | num_speakers | speaker embedding | batch_size | eval_split_size | formatter | n_checkpoints |
|---|---|---|---|---|---|---|---|---|---|
| System A | 16000 | 8,000 | 90 | 0 | False | 16 | 0.01 | custom_formatter | 5 |
| System B | 16000 | 8,000 | 90 | 0 | False | 16 | 0.01 | custom_formatter | 6 |
| System C | 22050 | – | 41 | 234 | True | 16 | 0.01 | pathnirwana_formatter | 13 |


### 1.3 Model size and footprint

| system | checkpoint | checkpoint (MB) | parameters | weights only, fp32 (MB) | step | epoch | train_loss | eval_loss |
|---|---|---|---|---|---|---|---|---|
| System A | best_model.pth | 951.6 | 83051308 | 316.8 | 63079 | 56 | 18.089 | 18.247 |
| System B | best_model_63079.pth | 951.6 | 83051308 | 316.8 | 63079 | 56 | 18.089 | 18.247 |
| System C | best_model.pth | 990.6 | 86453036 | 329.8 | 191661 | 251 | 17.207 | 15.119 |


The on-disk checkpoint is far larger than the deployable model because Coqui stores the
optimizer state alongside the weights. The *weights only* column is the figure relevant
to deployment; halving it again with fp16 export is straightforward.

**Byte-identical checkpoints detected** (same weights stored under different
names, verified by MD5): `System A/best_model.pth`, `System B/best_model_63079.pth`. These are not independent
models and are not treated as such anywhere in this report.


![Checkpoint sizes across all runs](outputs/02_checkpoint_footprint.png)

*Figure: Checkpoint sizes across all runs.*

### 1.4 Training environments

| property | ENV-A | ENV-B |
|---|---|---|
| environment | ENV-A — Kaggle Notebooks (2x T4, one GPU used) | ENV-B — Google Colab (L4, High-RAM) |
| systems trained here | System A, System B | System C |
| dataset | voicelk_custom (YouTube-scraped), 860 utterances | pathnirwana, 3300 utterances |
| speakers in corpus | 1 | 1 |
| accelerator | NVIDIA Tesla T4 | NVIDIA L4 |
| GPUs | 1 of 2 used | 1 |
| GPU memory (GB) | 15 | 22.5 |
| system RAM (GB) | 30 | 53 |
| OS | Linux 6.12.90+ x86_64, glibc 2.35 | Linux 6.6.122+ x86_64, glibc 2.39 |
| Python | 3.12.13 | 3.13.15 |
| PyTorch | 2.10.0+cu128 | 2.11.0+cu128 |
| CUDA | 12.8 (cu128 wheel) | 12.8 (cu128 wheel) |
| numpy / pandas | 2.0.2 / 2.3.3 | 2.1.3 / 2.2.3 |
| Coqui TTS | vendored in voicelk_ml/model_training/TTS | vendored in voicelk_ml/model_training/TTS |
| mixed precision | enabled (fp16) | enabled (fp16) |
| batch size | 16 train / 8 eval | 16 train / 8 eval |
| data loader workers | 2 | 2 |
| training seed | 54321 | 54321 |
| epochs configured | 1000 | 1000 |
| session limit (h) | 12 | – |
| GPU hours | – | – |
| cost model | free tier (~30 GPU-hours/week) | – |


**Configured schedule against what was actually reached.**

| system | environment | epochs configured | highest global step on disk | step of evaluated checkpoint |
|---|---|---|---|---|
| System A | ENV-A | 1000 | 63079 | 63079 |
| System B | ENV-A | 1000 | 63079 | 63079 |
| System C | ENV-B | 1000 | 191661 | 191661 |


Every run was configured for 1000 epochs. None of them ran that schedule to completion: training stopped or was interrupted at the step counts above. The per-checkpoint epoch field cannot be used as an epoch count here, because resuming from a checkpoint resets that counter while the global step continues — global step is the only monotonic progress measure available for these runs.

**ENV-A — Kaggle Notebooks (2x T4, one GPU used).** Repeated session restarts: 7 TensorBoard event files for one run, and the per-checkpoint epoch counter is non-monotonic (291, 185, 370, 185, 56), which is the signature of training being resumed from a checkpoint with the epoch counter reset while the global step continued. train_custom.py pins CUDA_VISIBLE_DEVICES=0, so only one of the two T4s was used - this is single-GPU training on a multi-GPU machine, not distributed. Corroborated by the config output_path /kaggle/working/voicelk_ml/model_training/runs/custom.

**ENV-B — Google Colab (L4, High-RAM).** 8 TensorBoard event files for this run, indicating several session restarts; checkpoints were written to Google Drive. Config output_path /content/drive/MyDrive/VoiceLK-V1.0/voicelk_ml/model_training/runs/pathnirwana - checkpoints persisted to mounted Drive, so Drive I/O is in the training loop. NOTE: although the corpus has a single speaker, this run's config declares num_speakers=234 with use_speaker_embedding=True; see the data-pipeline defect section of the report.

**Consequence for this report.** Because the systems were trained on different
hardware under different session limits, *training-side* numbers (wall-clock hours,
steps per second, GPU cost) describe the environment as much as the model and are
not used to rank systems. All *inference-side* numbers (§3, §6) were measured on the
single evaluation machine in the table above and are directly comparable.

### 1.5 Data-pipeline defect affecting the declared speaker count

**System C** declares `num_speakers = 234` with `use_speaker_embedding = True`, but `pathnirwana_metadata.txt` contains **1 speaker identity** across 3,300 utterances, which matches the 1 speaker recorded for its training environment. The declared figure is reproducible as a parsing artefact:

| quantity | value |
|---|---|
| utterances in metadata | 3,300 |
| distinct speaker ids, correct outside-in parse | 1 |
| distinct values in column 2 under a naive split('\|') | 234 |
| declared num_speakers in config.json | 234 |
| lines with an embedded '\|' in the IPA (text truncated by a naive split) | 1,021 (30.9%) |


The IPA sequences in this corpus legitimately contain `|` characters (left over from an older G2P phoneme separator). A plain `split('|')` therefore shifts the columns on those lines: column 1 keeps only the text up to the first `|`, and column 2 — read as the speaker id — picks up a fragment of phonemes instead. Doing exactly that to this file yields **234 distinct ids**, matching the configured `num_speakers` exactly, and the resulting ids are IPA fragments: `ə`, `a`, `ɯ`, `eː .`, `k`, `i`.

Two consequences for this checkpoint, both of which bear on how its results in §3 should be read:

1. **Around 31% of its training transcripts were truncated** at the first embedded `|`, so for roughly a third of the corpus the model was trained on audio paired with an incomplete phoneme sequence.
2. **The speaker-embedding table has 234 rows for one voice.** Utterances whose columns shifted were attributed to spurious identities, splitting one speaker across many embedding rows.

The formatters in `model_training/formatters.py` now split from the outside in (`partition`/`rpartition`) and are not subject to this, so the defect is historical — it is baked into this checkpoint, not into current code. Re-training this dataset with the fixed formatter and `num_speakers = 1` is the remedy.

**Which embedding rows actually trained.** Row norms were compared against the bulk distribution (robust z-score on median/MAD); a row that never received a gradient stays at its initialisation.

| system | declared speakers | dim | rows off init | speaker_id used | its robust z | is trained | unused rows | dead parameters |
|---|---|---|---|---|---|---|---|---|
| System C | 234 | 256 | 9 | 0 | -18.7 | True | 225 | 57600 |


The `speaker_id` used at synthesis time lands on a trained row in every case, so the audio evaluated in §3 is the voice the model actually learned — the surplus rows are inert rather than harmful. They remain a latent hazard: any caller passing a different speaker id would get an untrained embedding.


---

## 2. Methodology

### 2.1 Evaluation corpora

**Prompt set (17 sentences).** Hand-written to exercise the failure modes this system is expected to meet in ICT teaching material: plain Sinhala, ICT domain vocabulary, Sinhala–English code-switching, acronyms, numbers and dates, URLs, mathematical symbols and mixed script.

| category | prompts |
|---|---|
| general | 3 |
| code_switch | 3 |
| number | 3 |
| ict_domain | 2 |
| acronym | 2 |
| symbol | 1 |
| url | 1 |
| mixed_script | 1 |
| long | 1 |


**Reference set (recording + text pairs).** Coqui's `split_dataset()` shuffles with a fixed `np.random.seed(0)` before slicing off the evaluation portion, so the notebook reproduces each run's evaluation split *exactly* rather than approximating it. Every reference-based metric is therefore measured on audio the model never saw.

Because `eval_split_size = 0.01` leaves only 8 held-out utterances, a training-split sample is scored alongside as a separate, explicitly-labelled condition. The held-out figure is always the reported result; the training-split figure provides sample size and, by difference, a generalisation measure.

| system | split | utterances |
|---|---|---|
| System A | heldout | 2 |
| System A | train | 3 |
| System B | heldout | 2 |
| System B | train | 3 |


### 2.2 Metric definitions

**Real-time factor (RTF).** Wall-clock synthesis time divided by the duration of the audio produced. Reported both for the acoustic model alone and end-to-end including the text front-end. RTF < 1 means faster than real time. First call per system is a discarded warm-up.

**Mel-cepstral distortion (MCD).** 80-band mel spectrogram (`n_fft=1024`, `hop=256`), both signals RMS-normalised to −27 dBFS, `power_to_db(ref=max, top_db=40)` converted to natural log, DCT-II scaled by 2/N, coefficients c1–c13 (energy term c0 discarded), frames below −40 dB of the utterance peak excluded, DTW alignment on the cepstra, then `MCD = (10/ln10) · mean_t sqrt(2 · Σ_i (c_i^ref − c_i^syn)²)`. Calibration anchors measured on this corpus: identical signals 0.00 dB, a 30 dB-SNR noisy copy 0.46 dB, two *different* utterances by the same speaker ≈16 dB. This is a filterbank cepstrum rather than an SPTK `mcep` envelope, so values are internally comparable but should not be quoted directly against published MCD figures.

**F0 metrics.** `librosa.pyin` (65–400 Hz) on both signals, contours DTW-aligned on log-F0. RMSE is reported in Hz and in cents (cents being the perceptually meaningful scale), with Pearson correlation of the aligned contours and the voiced/unvoiced agreement rate.

**PESQ and STOI.** Intrusive metrics designed for a degraded signal that is sample-aligned with its reference. TTS re-generates its own timing, so the pair is not aligned; the notebook truncates to the shorter signal and the values are treated as a relative yardstick between these systems only, never as absolute quality scores.

**ASR-based WER/CER.** Synthesised audio is transcribed by a Sinhala ASR model and compared with the input text. Sinhala ASR is itself error-prone, so the same recogniser is also run over the *real recordings* of the same sentences to establish a topline; the reported quantity of interest is the gap between synthetic and real, and separately the gap between synthesis from the dataset's IPA label and synthesis from raw text (which isolates front-end damage).

**UTMOS.** A neural MOS predictor (UTMOS22-strong) trained on English and Japanese evaluation campaigns. Used as a ranking signal for Sinhala, not as a calibrated MOS, with the real recordings scored alongside as a ceiling.

**Speaker similarity and consistency.** Cosine distance between speaker embeddings: synthesis versus that speaker's real recording (similarity), and between different synthesised utterances of one system (consistency). ECAPA-TDNN x-vectors where `speechbrain` is installed; otherwise a labelled spectral proxy that is explicitly not a verification score. A recording-versus-recording cosine is always computed as calibration.

**Duration and prosody.** Per-token durations from the VITS duration predictor; utterance length ratio against the real recording; and timing drift, the RMS departure of the synthesis-to-recording DTW path from the diagonal, which catches systems that drift inside an utterance even when total lengths agree.

### 2.3 Statistical treatment

Objective metrics are reported as means with sample sizes, and standard deviations where the sample supports it. The subjective instruments in §7 carry the inferential machinery appropriate to each design: 95% confidence intervals for MOS and MUSHRA, a paired t-test for CMOS, and binomial tests against chance for AB preference, ABX discrimination and the Turing-style test. Single training runs with fixed seeds mean no variance estimate is available *across* runs; that limitation is restated in §9.


---

## 3. Objective results

### 3.0 Results at a glance

| system | RTF | MCD (dB) | F0 RMSE (cents) | pitch r | spk. similarity |
|---|---|---|---|---|---|
| System A | 0.365 | 11.91 | 299.4 | 0.688 | 0.105 |
| System B | 0.376 | 11.48 | 401.5 | 0.333 | -0.331 |
| System C | 0.513 | – | – | – | – |


Directions: RTF, MCD, F0 RMSE and WER lower is better; pitch correlation, UTMOS and speaker similarity higher is better.

### 3.1 Real-time factor and single-request latency

| system | RTF (model) | RTF (end-to-end) | p50 latency (ms) | p95 latency (ms) | front-end (ms) | n |
|---|---|---|---|---|---|---|
| System A | 0.37 | 0.376 | 1,103.623 | 2,088.931 | 19.774 | 17 |
| System B | 0.419 | 0.422 | 1,167.189 | 1,998.378 | 11.826 | 17 |
| System C | 0.491 | 0.494 | 2,126.637 | 4,045.027 | 13.358 | 17 |


The front-end costs on the order of 10–20 ms, negligible against the acoustic model, so RTF is essentially an acoustic-model property here.

![RTF distribution per system and latency against input length](outputs/10_rtf.png)

*Figure: RTF distribution per system and latency against input length.*

RTF by prompt category:

| category | System A | System B | System C |
|---|---|---|---|
| acronym | 0.361 | 0.399 | 0.486 |
| code_switch | 0.397 | 0.348 | 0.528 |
| general | 0.363 | 0.567 | 0.536 |
| ict_domain | 0.363 | 0.402 | 0.491 |
| long | 0.379 | 0.393 | 0.459 |
| mixed_script | 0.389 | 0.483 | 0.448 |
| number | 0.374 | 0.367 | 0.485 |
| symbol | 0.38 | 0.451 | 0.459 |
| url | 0.39 | 0.397 | 0.436 |


### 3.2 Mel-cepstral distortion

| system | split | MCD mean (dB) | std | min | max | n |
|---|---|---|---|---|---|---|
| System A | heldout | 10.872 | 0.054 | 10.834 | 10.911 | 2 |
| System A | train | 8.247 | 0.803 | 7.59 | 9.142 | 3 |
| System B | heldout | 11.283 | 1.492 | 10.228 | 12.339 | 2 |
| System B | train | 7.367 | 0.193 | 7.193 | 7.574 | 3 |


**Generalisation gap** (held-out minus training-split MCD): System A +2.62 dB; System B +3.92 dB. A large positive gap indicates the model reproduces sentences it trained on considerably better than new ones.

### 3.3 F0 accuracy and pitch correlation

| system | split | F0 RMSE (Hz) | F0 RMSE (cents) | pitch correlation r | voicing agreement | n |
|---|---|---|---|---|---|---|
| System A | heldout | 48.802 | 439.737 | 0.322 | 0.973 | 2 |
| System A | train | 29.91 | 252.223 | 0.551 | 0.995 | 3 |
| System B | heldout | 51.948 | 468.105 | 0.237 | 0.969 | 2 |
| System B | train | 25.897 | 219.123 | 0.678 | 0.994 | 3 |


Voicing agreement near 0.97 means the systems place voiced and unvoiced regions almost exactly where the speaker does; the pitch correlation is the weaker dimension, i.e. the intonation *contour* is the part that diverges.

### 3.4 PESQ and STOI

Not measured in this run — see §10 for the reason and what it would take.

### 3.5 ASR-based intelligibility (WER / CER)

Not measured in this run — see §10.

### 3.6 Predicted naturalness (UTMOS)

Not measured in this run — see §10.

### 3.7 Duration, prosody and timing

| system | mean frames/token | tokens at minimum duration (%) | length ratio vs recording | timing drift (frames) | speaking rate (chars/s) |
|---|---|---|---|---|---|
| System A | 1.288 | 82.773 | 0.838 | 11.621 | 16.549 |
| System B | 1.272 | 83.41 | 0.858 | 11.412 | 16.858 |
| System C | 3.044 | 35.476 | – | – | 10.496 |


**Finding.** System B assigns the minimum one-frame duration to 83% of input tokens. One frame is 16 ms at `hop_length=256`, so the majority of phonemes are given the shortest duration the model can express. The stochastic duration predictor has effectively stopped differentiating between phonemes, which flattens rhythm and is audible as rushed, monotone speech even when total utterance length happens to match the reference.


![Per-token predicted durations — custom](outputs/18_durations_custom.png)

*Figure: Per-token predicted durations — custom.*


![Per-token predicted durations — voicelk_vits_custom-Septembe](outputs/18_durations_voicelk_vits_custom-Septembe.png)

*Figure: Per-token predicted durations — voicelk_vits_custom-Septembe.*


![Per-token predicted durations — voicelk_vits_pathnirwana-Sep](outputs/18_durations_voicelk_vits_pathnirwana-Sep.png)

*Figure: Per-token predicted durations — voicelk_vits_pathnirwana-Sep.*


![Synthesised utterance length against recording length](outputs/18_duration_alignment.png)

*Figure: Synthesised utterance length against recording length.*

### 3.8 Speaker similarity and consistency

| system | cosine to own recording | std | n |
|---|---|---|---|
| System A | 0.105 | 0.547 | 2 |
| System B | -0.331 | 0.275 | 2 |


Self-consistency across utterances, with the real recordings as calibration:

| system | embedding | self_consistency | n |
|---|---|---|---|
| System A | proxy | -0.5 | 2 |
| System B | proxy | 0.441 | 2 |
| Ground truth (real recordings) | proxy | 0.337 | 4 |


> **Caveat.** `speechbrain` was not installed for this run, so these are spectral-proxy cosines (long-term average spectrum plus F0 statistics, mean-centred over the evaluation set), not speaker-verification embeddings. They react to content and recording channel as well as to voice identity and are only interpretable relative to the ground-truth row. Install `speechbrain` for ECAPA-TDNN x-vectors before quoting these numbers.


---

## 4. Checkpoint selection analysis

Coqui saves `best_model.pth` by lowest validation **loss**. Loss and perceptual quality are not the same objective, so every checkpoint in every run was loaded and scored on the same held-out sentences (byte-identical duplicates skipped).

| system | checkpoint | step | loss-selected | MCD (dB) | length ratio | tokens/s | n |
|---|---|---|---|---|---|---|---|
| System A | checkpoint_60000.pth | 60000 | False | 10.36 | 0.854 | 50.327 | 2 |
| System A | best_model.pth | 63079 | True | 10.374 | 0.849 | 50.684 | 2 |
| System A | best_model_63079.pth | 63079 | False | 11.101 | 0.87 | 49.422 | 2 |
| System A | checkpoint_70000.pth | 70000 | False | 9.451 | 0.862 | 49.894 | 2 |
| System A | checkpoint_80000.pth | 80000 | False | 11.41 | 0.842 | 51.069 | 2 |
| System B | checkpoint_20000.pth | 20000 | False | 10.796 | 0.873 | 49.254 | 2 |
| System B | checkpoint_30000.pth | 30000 | False | 11.3 | 0.952 | 45.178 | 2 |
| System B | checkpoint_40000.pth | 40000 | False | 9.757 | 0.847 | 50.769 | 2 |
| System B | checkpoint_50000.pth | 50000 | False | 9.998 | 0.891 | 48.277 | 2 |
| System B | best_model_51837.pth | 51837 | False | 11.465 | 0.857 | 50.146 | 2 |


![MCD across training checkpoints, loss-selected ones circled](outputs/11_checkpoint_sweep.png)

*Figure: MCD across training checkpoints, loss-selected ones circled.*


**Interpretation.**

- **System A**: the shipped checkpoint `best_model.pth` scores 10.37 dB, while `checkpoint_70000.pth` (step 70000) scores 9.45 dB — a 0.92 dB improvement available at no cost beyond changing which file is loaded.

This does not mean validation loss is useless — it means checkpoint selection should be confirmed against a perceptual or spectral metric, and ultimately against listeners, before a checkpoint is released. The sample here is small (see §9), so the recommendation is to re-score the leading candidates on a larger set rather than to switch on this evidence alone.


---

## 5. Component-level results

These sections evaluate the text front-end (`model_engine/`) rather than the acoustic model. Except for vocabulary coverage, they are model-independent.

### 5.1 G2P and lexicon fidelity

| lexicon | entries | exact match (%) | phoneme error rate |
|---|---|---|---|
| en_lexicon | 374 | 98.93 | 0.007 |
| si_lexicon | 115 | 94.783 | 0.016 |


Each lexicon entry is pushed through the full pipeline (normaliser → G2P) and compared with the IPA the lexicon specifies. A miss means the override never fired — tokenisation, normalisation or casing consumed the token — which surfaces as a silent mispronunciation of exactly the domain terms the lexicon exists to protect.

Largest mismatches:

| lexicon | word | lexicon IPA | pipeline IPA | PER |
|---|---|---|---|---|
| en_lexicon | rj45 | ˌɑː dʒeɪ ˈfɔː faɪv | rj | 1 |
| en_lexicon | fat32 | ˌef eɪ tiː ˈθɜːti tuː | fæt | 0.905 |
| en_lexicon | blu-ray | bluː reɪ | blu rɯɳa reɪ | 0.625 |
| si_lexicon | ඊ-කසළ | iː kasaɭa | iː rɯɳa kasaɭa | 0.556 |
| si_lexicon | විද්‍යුත් තැපෑල | ʋidjut̪ təpæl | ʋid̪jut̪ t̪æpæːla | 0.385 |
| si_lexicon | තාගස්_ණය | t̪aːkʃaɳaja | t̪aːgas ɳaja | 0.364 |
| si_lexicon | දත්ත සංචිත | dət̪t̪a saŋtʃita | d̪at̪t̪a satʃit̪a | 0.25 |
| si_lexicon | සත්‍ය වගුව | satjə ʋaɡuʋa | sat̪ja ʋaguʋa | 0.25 |
| en_lexicon | g2p_engine | dʒiː tuː piː ˈendʒɪn | dʒiː piː ˈendʒɪn | 0.2 |
| si_lexicon | ජීවන චක්‍රය | dʒiːʋana tʃakrajə | dʒiːʋana tʃakraja | 0.059 |


> **Scope limit.** This measures whether the lexicon is *applied*, not whether it is *correct* — the pipeline reads the same file used as reference. A genuine G2P accuracy figure requires a phonetician-verified gold set (§10).

### 5.2 IPA-to-vocabulary coverage

| system | vocab size | IPA chars seen | dropped | coverage (%) | distinct dropped |
|---|---|---|---|---|---|
| System A | 90 | 1274 | 0 | 100 | 0 |
| System B | 90 | 1274 | 0 | 100 | 0 |
| System C | 41 | 1274 | 61 | 95.212 | 15 |


Coqui's tokenizer discards characters outside the trained vocabulary **without raising an error**. Any front-end output the model never saw is therefore deleted silently, and the corresponding sound simply does not appear in the audio.

Most frequently discarded characters:

| system | char | unicode | name | count |
|---|---|---|---|---|
| System C | ɪ | U+026A | LATIN LETTER SMALL CAPITAL I | 18 |
| System C | ə | U+0259 | LATIN SMALL LETTER SCHWA | 16 |
| System C | ˈ | U+02C8 | MODIFIER LETTER VERTICAL LINE | 8 |
| System C | w | U+0077 | LATIN SMALL LETTER W | 3 |
| System C | z | U+007A | LATIN SMALL LETTER Z | 2 |
| System C | ʊ | U+028A | LATIN SMALL LETTER UPSILON | 2 |
| System C | ɯ | U+026F | LATIN SMALL LETTER TURNED M | 2 |
| System C | ɛ | U+025B | LATIN SMALL LETTER OPEN E | 2 |
| System C | ˌ | U+02CC | MODIFIER LETTER LOW VERTICAL LINE | 2 |
| System C | ð | U+00F0 | LATIN SMALL LETTER ETH | 1 |
| System C | ʧ | U+02A7 | LATIN SMALL LETTER TESH DIGRAPH | 1 |
| System C | θ | U+03B8 | GREEK SMALL LETTER THETA | 1 |
| System C | ʌ | U+028C | LATIN SMALL LETTER TURNED V | 1 |
| System C | ʰ | U+02B0 | MODIFIER LETTER SMALL H | 1 |
| System C | c | U+0063 | LATIN SMALL LETTER C | 1 |


### 5.3 Language routing (code-switch detection)

Token-level routing accuracy: **100.0%** over 22 labelled tokens spanning Sinhala words, English words, acronyms, digits and punctuation.

Confusion matrix (rows expected, columns routed):

| expected | english | other | sinhala |
|---|---|---|---|
| english | 8 | 0 | 0 |
| other | 0 | 8 | 0 |
| sinhala | 0 | 0 | 6 |


At sentence level, 0 digit token(s) survive normalisation across the prompt set. Digits that reach the router are classified as 'other' and passed through unspoken, so this count is the number of places a number would be silently omitted from the audio.

| id | category | tokens | sinhala | english | other | unspoken_digit_tokens |
|---|---|---|---|---|---|---|
| gen01 | general | 11 | 10 | 0 | 1 | 0 |
| gen02 | general | 5 | 4 | 0 | 1 | 0 |
| gen03 | general | 8 | 7 | 0 | 1 | 0 |
| ict01 | ict_domain | 13 | 11 | 0 | 2 | 0 |
| ict02 | ict_domain | 9 | 8 | 0 | 1 | 0 |
| cs01 | code_switch | 8 | 5 | 2 | 1 | 0 |
| cs02 | code_switch | 13 | 10 | 2 | 1 | 0 |
| cs03 | code_switch | 11 | 1 | 9 | 1 | 0 |
| acr01 | acronym | 10 | 6 | 3 | 1 | 0 |
| acr02 | acronym | 10 | 6 | 3 | 1 | 0 |
| num01 | number | 10 | 5 | 0 | 5 | 0 |
| num02 | number | 10 | 9 | 0 | 1 | 0 |
| num03 | number | 13 | 12 | 0 | 1 | 0 |
| sym01 | symbol | 11 | 5 | 5 | 1 | 0 |
| url01 | url | 15 | 13 | 1 | 1 | 0 |
| mix01 | mixed_script | 12 | 8 | 3 | 1 | 0 |
| long01 | long | 23 | 20 | 0 | 3 | 0 |


### 5.4 Lexicon coverage rate

| source | language | tokens | types | token coverage (%) | type coverage (%) |
|---|---|---|---|---|---|
| eval_prompts | english | 28 | 24 | 17.857 | 20.833 |
| eval_prompts | sinhala | 140 | 119 | 18.571 | 16.807 |
| dataset_transcripts | english | 6 | 6 | 0 | 0 |
| dataset_transcripts | sinhala | 27083 | 3309 | 4.161 | 0.604 |
| dataset_code_switch | english | 8838 | 2029 | 9.289 | 5.471 |
| dataset_code_switch | sinhala | 17633 | 1299 | 6.414 | 1.078 |


Low coverage is not automatically a defect — the rule-based fallback handles ordinary Sinhala vocabulary. It does mean that for every uncovered English loanword and acronym, pronunciation is left to `eng_to_ipa` and the letter-spelling heuristics rather than being specified.

Most frequent out-of-lexicon English tokens in the corpus:

| token | count | source |
|---|---|---|
| So | 277 | dataset_code_switch |
| use | 149 | dataset_code_switch |
| AI | 122 | dataset_code_switch |
| laptop | 121 | dataset_code_switch |
| And | 81 | dataset_code_switch |
| and | 81 | dataset_code_switch |
| phone | 62 | dataset_code_switch |
| app | 60 | dataset_code_switch |
| click | 49 | dataset_code_switch |
| iPhone | 43 | dataset_code_switch |
| Google | 40 | dataset_code_switch |
| Microsoft | 38 | dataset_code_switch |
| feature | 38 | dataset_code_switch |
| PC | 36 | dataset_code_switch |
| warranty | 36 | dataset_code_switch |

_(first 15 of 30 rows; full table in the CSV)_


### 5.5 Speaker consistency

Reported in §3.8 alongside speaker similarity, since both derive from the same embeddings.


---

## 6. System-level results

### 6.1 End-to-end latency under load

| system | concurrency | requests | throughput (req/s) | audio generated (× real time) | p50_ms | p95_ms | max_ms | failures |
|---|---|---|---|---|---|---|---|---|
| System A | 1 | 4 | 1.063 | 2.78 | 826.3 | 1,487.7 | 1,566.2 | 0 |
| System A | 2 | 4 | 1.379 | 3.53 | 1,221.7 | 1,467.8 | 1,477.3 | 0 |
| System B | 1 | 4 | 1.155 | 2.96 | 788.6 | 1,424.5 | 1,508.5 | 0 |
| System B | 2 | 4 | 1.289 | 3.41 | 1,137.2 | 1,682 | 1,738.6 | 0 |
| System C | 1 | 4 | 0.484 | 2.05 | 1,828.2 | 3,211.8 | 3,443.6 | 0 |
| System C | 2 | 4 | 0.528 | 2.4 | 2,785.7 | 4,340.3 | 4,576.6 | 0 |



![Throughput and tail latency against concurrency](outputs/30_load_latency.png)

*Figure: Throughput and tail latency against concurrency.*


The model runs in-process, so concurrency is bounded by the GIL and by torch's intra-op threads. The informative part is the shape: throughput saturates quickly while p95 latency keeps climbing, which is the signature of queueing rather than parallelism. A production deployment should therefore scale by processes or replicas, not by threads, and these figures are a per-process ceiling rather than a capacity estimate.

### 6.2 Robustness and edge cases

| system | clean (%) |
|---|---|
| System A | 90.909 |
| System B | 90.909 |
| System C | 45.455 |


Each of the 22 probes goes through the deployed path (normalise → G2P → tokenize → synthesise). A case is flagged only when the output is implausible rather than merely different: no audio for non-empty text, a speaking rate outside 3–40 characters/second, near-total silence, clipping, or discarded characters.

Flagged cases:

| system | case | input_chars | audio_s | chars_per_s | dropped_chars | flags |
|---|---|---|---|---|---|---|
| System A | whitespace | 3 | 0.112 | 26.786 | – | text_produced_no_phonemes |
| System A | emoji_only | 3 | 0.08 | 37.5 | – | text_produced_no_phonemes |
| System B | whitespace | 3 | 0.112 | 26.786 | – | text_produced_no_phonemes |
| System B | emoji_only | 3 | 0.224 | 13.393 | – | text_produced_no_phonemes |
| System C | whitespace | 3 | 0.163 | 18.457 | – | text_produced_no_phonemes |
| System C | emoji_only | 3 | 0.163 | 18.457 | – | text_produced_no_phonemes |
| System C | repeated_char | 19 | 8.94 | 2.125 | – | implausible_rate(2.1c/s) |
| System C | numbers_plain | 33 | 4.888 | 6.752 | ə | chars_dropped |
| System C | percentage | 26 | 2.972 | 8.748 | ə | chars_dropped |
| System C | acronym_caps | 32 | 10.101 | 3.168 | ɒɪʌˈˌ | chars_dropped |
| System C | acronym_word | 23 | 3.994 | 5.759 | ɪʊ | chars_dropped |
| System C | mixed_script | 46 | 2.577 | 17.847 | wɑəɪʊʤˈˌθ | chars_dropped |
| System C | english_only | 44 | 2.287 | 19.238 | wzðɑɔəɪʊʤˈ | chars_dropped |
| System C | url_email | 46 | 7.175 | 6.411 | czəɛɪˈ | chars_dropped |
| System C | symbols_math | 29 | 5.097 | 5.69 | wzəɪʌʰ | chars_dropped |
| System C | code_snippet | 28 | 10.124 | 2.766 | əɪʤ | implausible_rate(2.8c/s),chars_dropped |


### 6.3 Footprint

| system | checkpoint (MB) | parameters | weights fp32 (MB) |
|---|---|---|---|
| System A | 951.6 | 83051308 | 316.8 |
| System B | 951.6 | 83051308 | 316.8 |
| System C | 990.6 | 86453036 | 329.8 |



---

## 7. Subjective evaluation

Human listening is the primary evidence for TTS quality; every objective number above is a proxy for it. The material for seven listener-facing instruments has been generated, blinded and randomised, and the scoring code is written — what remains is administering them.

| instrument | trials prepared | sheet | what it measures |
|---|---|---|---|
| MOS | 57 | `40_listening_test/mos_rating_sheet.csv` | Absolute category rating, 1–5, audio only (no text shown), on naturalness, audio quality and listening effort. |
| CMOS | 51 | `40_listening_test/cmos_sheet.csv` | Same sentence from two systems back to back, rated −3…+3. Paired, so it resolves smaller differences than MOS. |
| AB preference | 51 | `40_listening_test/ab_preference_sheet.csv` | Which of two systems sounds better. |
| ABX | 51 | `40_listening_test/abx_sheet.csv` | Whether the two systems can be told apart at all; a preference result is only meaningful if ABX beats chance. |
| Transcription (intelligibility) | 57 | `40_listening_test/transcription_sheet.csv` | Listener types what they hear, with no text on screen; scored by WER/CER against the synthesised sentence, with real recordings mixed in as controls. |
| MUSHRA | 4 | `40_listening_test/mushra_sheet.csv` | 0–100 rating of every version of a sentence against a labelled reference, including a hidden reference and a 3.5 kHz low-pass anchor for listener post-screening. |
| Turing-style | 6 | `40_listening_test/turing_sheet.csv` | Real or synthetic, forced choice, with confidence. |
| ICT comprehension | 15 | `40_listening_test/ict_comprehension_sheet.csv` | Students answer questions about a passage they only heard — the downstream task this system exists to serve. |


**Blinding.** Rating sheets carry anonymous sample identifiers only; the mapping from identifier to system lives in a separate `*_key.csv` that listeners never receive. Presentation order in the paired tests is randomised per trial under a fixed seed, so the design is reproducible.

**Administration.** Open `outputs/40_listening_test/listening_test.html` from inside that folder — it plays every instrument in the browser and exports one CSV of responses. `ingest_html_responses()` in §19.9 of the notebook converts that export into the per-test sheets, and the `score_*()` helpers produce the statistics.

**Sample size.** At least 15 listeners × 10 sentences per system for MOS; CMOS needs fewer for equivalent power because it is paired; ABX needs at least 25 trials per listener for the binomial test to be informative; MUSHRA requires post-screening on the hidden reference (discard a listener who rates it below 90).

> **No listening results yet.** This section will populate once the sheets come back and the scoring helpers are run. Until then this report supports no claim about perceived quality, only about the measurable proxies in §3.


---

## 8. Error taxonomy

Counting failures by type is what turns a quality score into a work plan. The table below aggregates the automatically detectable failures; the categories that require a human listening pass are listed in `50_error_taxonomy.csv` awaiting that pass.

| category | System A | Text front-end | System B | System C |
|---|---|---|---|---|
| dropped_phoneme | 0 | 0 | 0 | 69 |
| mispronunciation_english | 0 | 4 | 0 | 0 |
| mispronunciation_sinhala | 0 | 6 | 0 | 0 |
| silence_or_no_output | 2 | 0 | 2 | 2 |
| speaking_rate | 0 | 0 | 0 | 2 |


41 issues were detected automatically; 13 categories remain for the manual pass.

Automatically detected issues:

| system | category | description | severity | count | evidence |
|---|---|---|---|---|---|
| System C | dropped_phoneme | tokenizer discards 'ɪ' (U+026A LATIN LETTER SMALL CAPITAL I) | 2 | 18 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ə' (U+0259 LATIN SMALL LETTER SCHWA) | 2 | 16 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ˈ' (U+02C8 MODIFIER LETTER VERTICAL LINE) | 2 | 8 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'w' (U+0077 LATIN SMALL LETTER W) | 2 | 3 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'z' (U+007A LATIN SMALL LETTER Z) | 2 | 2 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ʊ' (U+028A LATIN SMALL LETTER UPSILON) | 2 | 2 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ɯ' (U+026F LATIN SMALL LETTER TURNED M) | 2 | 2 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ɛ' (U+025B LATIN SMALL LETTER OPEN E) | 2 | 2 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ˌ' (U+02CC MODIFIER LETTER LOW VERTICAL LINE) | 2 | 2 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ð' (U+00F0 LATIN SMALL LETTER ETH) | 2 | 1 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ʧ' (U+02A7 LATIN SMALL LETTER TESH DIGRAPH) | 2 | 1 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'θ' (U+03B8 GREEK SMALL LETTER THETA) | 2 | 1 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ʌ' (U+028C LATIN SMALL LETTER TURNED V) | 2 | 1 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'ʰ' (U+02B0 MODIFIER LETTER SMALL H) | 2 | 1 | 20_ipa_dropped_chars.csv |
| System C | dropped_phoneme | tokenizer discards 'c' (U+0063 LATIN SMALL LETTER C) | 2 | 1 | 20_ipa_dropped_chars.csv |
| System A | silence_or_no_output | edge case 'whitespace' flagged: text_produced_no_phonemes | 2 | 1 | 31_robustness.csv |
| System A | silence_or_no_output | edge case 'emoji_only' flagged: text_produced_no_phonemes | 2 | 1 | 31_robustness.csv |
| System B | silence_or_no_output | edge case 'whitespace' flagged: text_produced_no_phonemes | 2 | 1 | 31_robustness.csv |
| System B | silence_or_no_output | edge case 'emoji_only' flagged: text_produced_no_phonemes | 2 | 1 | 31_robustness.csv |
| System C | silence_or_no_output | edge case 'whitespace' flagged: text_produced_no_phonemes | 2 | 1 | 31_robustness.csv |
| System C | silence_or_no_output | edge case 'emoji_only' flagged: text_produced_no_phonemes | 2 | 1 | 31_robustness.csv |
| System C | speaking_rate | edge case 'repeated_char' flagged: implausible_rate(2.1c/s) | 2 | 1 | 31_robustness.csv |
| System C | dropped_phoneme | edge case 'numbers_plain' flagged: chars_dropped | 2 | 1 | 31_robustness.csv |
| System C | dropped_phoneme | edge case 'percentage' flagged: chars_dropped | 2 | 1 | 31_robustness.csv |
| System C | dropped_phoneme | edge case 'acronym_caps' flagged: chars_dropped | 2 | 1 | 31_robustness.csv |

_(first 25 of 41 rows; full table in the CSV)_



---

## 9. Limitations and threats to validity

1. **Small held-out set.** `eval_split_size = 0.01` leaves only 2 held-out utterances per run, so every reference-based mean (MCD, F0, PESQ/STOI, ASR) rests on a handful of sentences. Training-split figures are reported alongside for sample size, but they are not a measure of performance on unseen input. Raising `eval_split_size` and retraining, or setting aside a purpose-built test set, would fix this properly.
2. **Single training run per configuration.** Each system is one run with one seed, so no variance estimate exists across runs and small differences between systems cannot be called significant. Repeated runs with different seeds would be needed for that claim.
3. **The systems are not a controlled comparison.** They differ simultaneously in dataset, sample rate, phoneme vocabulary and speaker-embedding configuration, so a difference in any metric cannot be attributed to a single cause. Cross-system numbers are context, not an ablation.
4. **Speaker metrics use a proxy embedding.** Without `speechbrain`, similarity and consistency come from a spectral descriptor that also responds to content and channel. Those two rows should be regenerated with ECAPA-TDNN x-vectors before being cited.
5. **System C was trained through a defective data pipeline** (§1.5): roughly a third of its transcripts were truncated and its speaker-embedding table holds 234 rows for a single voice. Its metrics describe that checkpoint faithfully, but they are not a fair measure of the architecture or of the pathnirwana corpus — a comparison against the custom systems is confounded by the data defect, not only by the dataset change.
6. **Objective metrics are proxies for listening.** MCD, F0 error and UTMOS correlate with perceived quality imperfectly, and UTMOS in particular was trained on English and Japanese systems. No conclusion about naturalness is final until §7 is completed.
7. **Training-side figures reflect their environments.** The systems were trained on different platforms with different session limits and interruption patterns (§1.4), so wall-clock hours and step throughput are properties of those environments, not model efficiency.


---

## 10. Conclusions and recommended next steps

1. **Re-score checkpoint selection for System A.** `checkpoint_70000.pth` scores 0.92 dB better than the loss-selected `best_model.pth`. Confirm on a larger reference set and in a CMOS test, then ship whichever wins.
2. **Investigate the duration predictor.** With most tokens at the minimum one-frame duration, rhythm carries almost no information. Check the duration-loss weight and `length_scale` at inference, and verify the IPA labels are not systematically shorter than the audio they are paired with. This is the highest-value acoustic fix available.
3. **Close the phoneme-vocabulary gap.** Characters the front-end emits but the model cannot represent are deleted silently. Either restrict the front-end to the trained vocabulary and map the remainder to the nearest in-vocabulary phoneme, or retrain with the full inventory. At minimum, make the tokenizer log discarded characters instead of swallowing them.
4. **Re-train System C with the fixed formatter and `num_speakers = 1`.** The current checkpoint was trained with ~31% of its transcripts truncated and a 234-row embedding table for one voice (§1.5). The formatter bug is already fixed in `model_training/formatters.py`, so this is a re-run rather than a code change, and it is the only way to find out what this corpus is actually worth.
5. **Extend the ICT lexicon by corpus frequency.** `22_lexicon_oov_top300.csv` is already ordered by frequency; adding the top entries covers the most speech per entry written and directly targets the code-switched vocabulary this system is for.
6. **Run the listening tests.** The blinded material, the browser test page and the scoring code are ready. MOS plus a CMOS between the two leading checkpoints, with ABX to confirm the systems are distinguishable at all, is the minimum needed to make a defensible quality claim.
7. **Complete the optional objective metrics.** Installing `pystoi`, `transformers` and `speechbrain` (and building `pesq`, or running that section on Python 3.11) fills in STOI, ASR WER/CER, UTMOS and ECAPA speaker similarity, which are already implemented and currently recorded as skips.
8. **Enlarge the test set.** A purpose-built held-out set — ideally recorded specifically for evaluation and covering the ICT syllabus, code-switching and numeric content in known proportions — would remove the single largest statistical weakness in this report.


---

## Appendix A — complete metric table

| group | metric | system | value | unit | n | note |
|---|---|---|---|---|---|---|
| system | model_size_on_disk | System A | 951.6 | MB | – | checkpoint=best_model.pth (optimizer state included) |
| system | parameters | System A | 83051308 | params | – | – |
| system | weights_only_fp32 | System A | 316.8 | MB | – | weights only — deployable footprint without optimizer state |
| system | model_size_on_disk | System B | 951.6 | MB | – | checkpoint=best_model_63079.pth (optimizer state included) |
| system | parameters | System B | 83051308 | params | – | – |
| system | weights_only_fp32 | System B | 316.8 | MB | – | weights only — deployable footprint without optimizer state |
| system | model_size_on_disk | System C | 990.6 | MB | – | checkpoint=best_model.pth (optimizer state included) |
| system | parameters | System C | 86453036 | params | – | – |
| system | weights_only_fp32 | System C | 329.8 | MB | – | weights only — deployable footprint without optimizer state |
| objective | rtf_model_only | System A | 0.36 | ratio | 17 | acoustic model only; <1 is faster than real time |
| objective | rtf_end_to_end | System A | 0.365 | ratio | 17 | front-end + model |
| system | latency_p95_single_request | System A | 1863.8 | ms | 17 | – |
| system | frontend_latency_mean | System A | 17.4 | ms | 17 | normaliser + G2P |
| objective | rtf_model_only | System B | 0.373 | ratio | 17 | acoustic model only; <1 is faster than real time |
| objective | rtf_end_to_end | System B | 0.376 | ratio | 17 | front-end + model |
| system | latency_p95_single_request | System B | 1908.8 | ms | 17 | – |
| system | frontend_latency_mean | System B | 14.0 | ms | 17 | normaliser + G2P |
| objective | rtf_model_only | System C | 0.511 | ratio | 17 | acoustic model only; <1 is faster than real time |
| objective | rtf_end_to_end | System C | 0.513 | ratio | 17 | front-end + model |
| system | latency_p95_single_request | System C | 4174.3 | ms | 17 | – |
| system | frontend_latency_mean | System C | 12.6 | ms | 17 | normaliser + G2P |
| objective | MCD | System A | 11.91 | dB | 2 | std 0.09; held-out refs; notebook-internal scale |
| objective | MCD | System B | 11.48 | dB | 2 | std 1.04; held-out refs; notebook-internal scale |
| objective | best_checkpoint_by_MCD | System A | checkpoint_70000.pth | – | 70,000 | MCD 9.45 dB on 2 held-out pairs |
| objective | best_checkpoint_by_MCD | System B | checkpoint_40000.pth | – | 40,000 | MCD 9.76 dB on 2 held-out pairs |
| objective | F0_RMSE | System A | 31.4 | Hz | 2 | – |
| objective | F0_RMSE_cents | System A | 299.4 | cents | 2 | – |
| objective | pitch_correlation | System A | 0.688 | r | 2 | – |
| objective | voicing_agreement | System A | 0.963 | fraction | 2 | – |
| objective | F0_RMSE | System B | 44.7 | Hz | 2 | – |
| objective | F0_RMSE_cents | System B | 401.5 | cents | 2 | – |
| objective | pitch_correlation | System B | 0.333 | r | 2 | – |
| objective | voicing_agreement | System B | 0.971 | fraction | 2 | – |
| objective | speaker_similarity_to_reference | System A | 0.105 | cosine | 2 | spectral proxy (LTAS + F0 stats), content-sensitive - not a verification score |
| objective | speaker_similarity_to_reference | System B | -0.331 | cosine | 2 | spectral proxy (LTAS + F0 stats), content-sensitive - not a verification score |
| component | speaker_consistency | System A | -0.5 | cosine | 2 | spectral proxy (LTAS + F0 stats), content-sensitive - not a verification score (GROUND_TRUTH row is the real-speaker calibration) |
| component | speaker_consistency | System B | 0.441 | cosine | 2 | spectral proxy (LTAS + F0 stats), content-sensitive - not a verification score (GROUND_TRUTH row is the real-speaker calibration) |
| component | speaker_consistency | Ground truth (real recordings) | 0.337 | cosine | 4 | spectral proxy (LTAS + F0 stats), content-sensitive - not a verification score (GROUND_TRUTH row is the real-speaker calibration) |
| component | predicted_duration_mean | System A | 1.29 | frames/token | 19 | ~1.0 means the duration predictor has collapsed to its minimum |
| component | tokens_at_minimum_duration | System A | 82.8 | % | 19 | – |
| component | duration_ratio_vs_recording | System A | 0.838 | ratio | 2 | 1.0 = same length as the real recording |
| component | timing_drift | System A | 11.6 | frames | 2 | RMS deviation of the DTW path from the diagonal |
| component | predicted_duration_mean | System B | 1.27 | frames/token | 19 | ~1.0 means the duration predictor has collapsed to its minimum |
| component | tokens_at_minimum_duration | System B | 83.4 | % | 19 | – |
| component | duration_ratio_vs_recording | System B | 0.858 | ratio | 2 | 1.0 = same length as the real recording |
| component | timing_drift | System B | 11.4 | frames | 2 | RMS deviation of the DTW path from the diagonal |
| component | predicted_duration_mean | System C | 3.04 | frames/token | 17 | ~1.0 means the duration predictor has collapsed to its minimum |
| component | tokens_at_minimum_duration | System C | 35.5 | % | 17 | – |
| component | lexicon_fidelity[en_lexicon] | Text front-end | 98.9 | % | 374 | share of lexicon entries the pipeline reproduces exactly |
| component | phoneme_error_rate[en_lexicon] | Text front-end | 0.007 | PER | 374 | – |
| component | lexicon_fidelity[si_lexicon] | Text front-end | 94.8 | % | 115 | share of lexicon entries the pipeline reproduces exactly |
| component | phoneme_error_rate[si_lexicon] | Text front-end | 0.016 | PER | 115 | – |
| component | ipa_vocab_coverage | System A | 100.0 | % | 1,274 | 0 chars silently discarded by the tokenizer (0 distinct) |
| component | ipa_vocab_coverage | System B | 100.0 | % | 1,274 | 0 chars silently discarded by the tokenizer (0 distinct) |
| component | ipa_vocab_coverage | System C | 95.21 | % | 1,274 | 61 chars silently discarded by the tokenizer (15 distinct) |
| component | language_routing_accuracy | Text front-end | 100.0 | % | 22 | token-level router probe (Sinhala / English / other) |
| component | digits_surviving_normalisation | Text front-end | 0 | tokens | 192 | digit tokens still present after normalisation - these reach the model unspoken |
| component | lexicon_coverage[eval_prompts/english] | Text front-end | 17.86 | % of tokens | 28 | type coverage 20.8% |
| component | lexicon_coverage[eval_prompts/sinhala] | Text front-end | 18.57 | % of tokens | 140 | type coverage 16.8% |
| component | lexicon_coverage[dataset_transcripts/english] | Text front-end | 0.0 | % of tokens | 6 | type coverage 0.0% |
| component | lexicon_coverage[dataset_transcripts/sinhala] | Text front-end | 4.16 | % of tokens | 27,083 | type coverage 0.6% |
| component | lexicon_coverage[dataset_code_switch/english] | Text front-end | 9.29 | % of tokens | 8,838 | type coverage 5.5% |
| component | lexicon_coverage[dataset_code_switch/sinhala] | Text front-end | 6.41 | % of tokens | 17,633 | type coverage 1.1% |
| system | peak_throughput | System A | 1.379 | req/s | 2 | at concurrency 2 on cpu |
| system | latency_p95_under_load | System A | 1467.8 | ms | 2 | at concurrency 2 |
| system | peak_throughput | System B | 1.289 | req/s | 2 | at concurrency 2 on cpu |
| system | latency_p95_under_load | System B | 1682.0 | ms | 2 | at concurrency 2 |
| system | peak_throughput | System C | 0.528 | req/s | 2 | at concurrency 2 on cpu |
| system | latency_p95_under_load | System C | 4340.3 | ms | 2 | at concurrency 2 |
| system | robustness_clean_rate | System A | 90.9 | % | 22 | edge cases producing no anomaly flag |
| system | robustness_hard_failures | System A | 2 | cases | 22 | – |
| system | robustness_clean_rate | System B | 90.9 | % | 22 | edge cases producing no anomaly flag |
| system | robustness_hard_failures | System B | 2 | cases | 22 | – |
| system | robustness_clean_rate | System C | 45.5 | % | 22 | edge cases producing no anomaly flag |
| system | robustness_hard_failures | System C | 2 | cases | 22 | – |


## Appendix B — metrics not measured, and why

Nothing requested was dropped silently. Each entry states the reason and what would be required to obtain it.

| group | metric | reason | what would be needed |
|---|---|---|---|
| objective | reference_metrics[System C] | none of the 33 held-out recordings for this run exist locally (metadata points at pathnirwana_metadata.txt, wavs expected under D:\RUSL\Final Project\TTS\data) | copy this dataset's wavs/ folder to this machine |
| objective | PESQ | the 'pesq' package is not installed (it is a C extension with no Python 3.14 wheel on PyPI) | install MS Visual C++ Build Tools then `pip install pesq`, or run this section on Python 3.11 |
| objective | STOI/ESTOI | the 'pystoi' package is not installed | `pip install pystoi` (pure Python, installs cleanly on 3.14) |
| objective | ASR WER/CER | disabled in CONFIG['enable_asr'] | set it to True and re-run |
| objective | Neural MOS (UTMOS) | disabled in CONFIG['enable_utmos'] | set it to True |
| objective | Speaker similarity (ECAPA x-vector) | the 'speechbrain' package is not installed, so a spectral proxy is used instead | `pip install speechbrain` for a real speaker-verification embedding |
| component | phoneme-level duration vs forced alignment | no forced aligner with a Sinhala acoustic model is available on this machine | train/obtain an MFA Sinhala acoustic model, align the corpus, then compare per-phoneme durations against the VITS duration predictor |
| component | G2P accuracy vs a human-verified phonetic gold set | no human-transcribed IPA reference exists for this corpus; the lexicon is the only machine-checkable reference and the pipeline reads that same file | have a phonetician transcribe ~200 sampled words (Sinhala, English loanwords, acronyms) and score the pipeline against that set |
| subjective | MOS score | listening test not yet run - only the blinded material is generated | collect mos_rating_sheet.csv from >=15 listeners, then run the scoring cell in 19.9 |
| subjective | CMOS score | listening test not yet run - only the blinded material is generated | collect cmos_sheet.csv from listeners, then run the scoring cell in 19.9 |
| subjective | AB preference / ABX | listening test not yet run - only the blinded material is generated | collect ab_preference_sheet.csv and abx_sheet.csv from listeners, then run 19.9 |
| subjective | Intelligibility (human transcription) | listening test not yet run - only the blinded material is generated | collect transcription_sheet.csv from listeners, then run the scorer in 19.9 |
| subjective | MUSHRA score | listening test not yet run - only the blinded material is generated | collect mushra_sheet.csv from trained listeners, then run the scorer in 19.9 |
| subjective | Turing-style real-vs-synthetic | listening test not yet run | collect turing_sheet.csv from listeners, then run the scorer in 19.9 |
| subjective | Downstream ICT comprehension accuracy | needs students to answer the quiz; only the audio + quiz instrument is generated here | run the quiz with >=20 students per system, score accuracy and mean replays per item |
| system | A/B deployment test (engagement, replay, skip rates) | requires both models served to real users with telemetry; there is no deployment or event pipeline attached to this project yet | ship both models behind a feature flag, log the events in the spec below, and run the test for at least two weeks or until the required sample size is reached |


## Appendix C — A/B deployment test specification

Online A/B testing cannot be performed from a notebook: it requires both models served to real users with telemetry returning. The specification below defines what to instrument so the test is decidable when VoiceLK ships.

| metric | definition | event | primary | direction | n per arm (5 pp effect) |
|---|---|---|---|---|---|
| listen_through_rate | share of playbacks reaching >=90% of the clip | audio_progress | True | higher is better | 1,377 |
| replay_rate | share of clips replayed at least once | audio_replay | True | lower is better (signals unclear audio) | 1,377 |
| skip_rate | share of playbacks abandoned in the first 5 seconds | audio_skip | True | lower is better | 1,377 |
| session_audio_minutes | audio minutes played per session | audio_progress | False | higher is better | – |
| thumbs_down_rate | explicit negative feedback per 100 clips | feedback_submit | False | lower is better | – |
| p95_synthesis_latency | server-side p95 time to first audio byte | tts_request | False | lower is better | – |
| error_rate | failed synthesis requests per 1000 | tts_request | False | lower is better | – |



## Appendix D — artifacts and reproduction

Every number in this report comes from `model_evaluate/outputs/`, regenerated by running `model_evaluate/model_evaluation_full_suite.ipynb` top to bottom and then `python model_evaluate/build_report.py`.

| artifact | size |
|---|---|
| 00_eval_environment.json | 1 KB |
| 01_model_registry.csv | 1 KB |
| 02_checkpoint_footprint.csv | 1 KB |
| 02_checkpoint_footprint.png | 24 KB |
| 02_speaker_embedding_health.csv | 0 KB |
| 04_eval_corpus.csv | 2 KB |
| 04_reference_pairs.csv | 1 KB |
| 10_rtf.csv | 9 KB |
| 10_rtf.png | 71 KB |
| 11_checkpoint_sweep.csv | 1 KB |
| 11_checkpoint_sweep.png | 73 KB |
| 11_mcd.csv | 1 KB |
| 12_f0.csv | 2 KB |
| 17_speaker_consistency.csv | 0 KB |
| 17_speaker_similarity.csv | 0 KB |
| 18_duration_alignment.png | 42 KB |
| 18_duration_prosody.csv | 9 KB |
| 18_durations_custom.png | 28 KB |
| 18_durations_voicelk_vits_custom-Septembe.png | 38 KB |
| 18_durations_voicelk_vits_pathnirwana-Sep.png | 36 KB |
| 20_g2p_lexicon_fidelity.csv | 29 KB |
| 20_g2p_worst_mismatches.csv | 1 KB |
| 20_ipa_dropped_chars.csv | 1 KB |
| 20_ipa_vocab_coverage.csv | 0 KB |
| 21_language_routing_sentences.csv | 1 KB |
| 21_language_routing_tokens.csv | 1 KB |
| 22_lexicon_coverage.csv | 0 KB |
| 22_lexicon_oov_top300.csv | 14 KB |
| 30_load_latency.csv | 1 KB |
| 30_load_latency.png | 71 KB |
| 31_robustness.csv | 17 KB |
| 40_listening_test/ | 18 files |
| 50_error_taxonomy.csv | 8 KB |
| 50_error_taxonomy_counts.csv | 0 KB |
| 51_ab_deployment_spec.csv | 1 KB |
| 90_summary_metrics.csv | 9 KB |
| 90_summary_pivot.csv | 2 KB |
| 91_skipped_metrics.csv | 3 KB |
| 92_evaluation_report.md | 15 KB |
| 99_digest.txt | 42 KB |


**Artifacts absent from this build** (the corresponding sections say so explicitly): `13_pesq_stoi.csv`, `15_asr_wer.csv`, `16_utmos.csv`, `41_ab_preference_scores.csv`, `41_abx_scores.csv`, `41_cmos_scores.csv`, `41_ict_comprehension_scores.csv`, `41_mos_scores.csv`, `41_mushra_scores.csv`, `41_transcription_scores.csv`, `41_turing_scores.csv`.
