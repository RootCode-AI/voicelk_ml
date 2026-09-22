# model_finetune — Fix A

Training-free repairs for the VoiceLK TTS project. **Nothing here trains, fine-tunes or
modifies a checkpoint.** Existing checkpoints are read only.

## Why this exists

`model_evaluate/EVALUATION_REPORT.md` cannot currently detect an improvement. It scores two
**byte-identical** checkpoints (§1.3 confirms the MD5 match) at 10.374 dB and 11.101 dB, and
quotes 11.91 dB and 10.872 dB for the same weights elsewhere — a ~1.5 dB spread from one model.

The cause is that VITS synthesis is stochastic (`inference_noise_scale = 0.667`,
`noise_scale_dp = 1.0`) and the evaluation notebook seeds neither. Until that is fixed, a
fine-tune cannot be shown to have worked, and §10.1's recommendation to ship
`checkpoint_70000.pth` on a 0.92 dB margin sits inside the noise.

Fix A does not improve audio quality. **It makes the improvement measurable.**

## Which environment

**Use `voicelk_ml/venv` — Python 3.11.9. Locally, on CPU. Not Colab, not Kaggle.**

| | Python 3.11 (`venv`) | Python 3.14 (`venv_evoluation`) |
|---|---|---|
| torch, librosa, pandas, soundfile | yes | yes |
| sinling, eng_to_ipa | yes | yes |
| vendored Coqui TTS imports | yes (verified) | yes |
| `pesq` | installable | **no wheel exists** |
| `pystoi` | installable | installable |
| `speechbrain` (ECAPA x-vectors) | installable | unsupported |
| `transformers` (UTMOS) | installable | unsupported |

Appendix B of the report lists PESQ, STOI, ASR WER/CER, UTMOS and ECAPA speaker similarity as
skipped. Four of the five are blocked by Python 3.14 alone. Running Fix A on 3.11 unblocks
them, and the code for all of them already exists in the evaluation notebook.

Cloud is the wrong choice here: Fix A needs no GPU, and the data is already local — 908 wav
files plus two ~950 MB checkpoints would have to be uploaded for no benefit. Save Kaggle and
Colab for the fine-tune itself.

```powershell
# from voicelk_ml/
.\venv\Scripts\Activate.ps1
pip install -r model_finetune\requirements-fix-a.txt
python -m ipykernel install --user --name voicelk-fix-a --display-name "venv (Python 3.11)"
jupyter notebook model_finetune\fix_a_diagnostics.ipynb
```

Select the **venv (Python 3.11)** kernel before running.

## Where the data is

The wav files are **not** in `voicelk_ml/data/wavs`. They sit one level up, beside the repo:

```
D:\RUSL\Final Project\TTS\
├── data\wavs\          <- 908 wav files, 16 kHz
└── voicelk_ml\
    ├── data\custom_metadata.txt
    └── model_finetune\   <- you are here
```

`fa_core` resolves this from its own file location, so the notebook works regardless of the
kernel's working directory. If your layout differs, edit `WAV_ROOT` in `fa_core.py`.

## Files

| file | contents |
|---|---|
| `fix_a_diagnostics.ipynb` | the notebook — run this |
| `fa_core.py` | paths, phoneme counting, alignment audit, G2P audit, footprint, test split |
| `fa_inference.py` | model loading, deterministic synthesis, MCD, noise floor, length_scale sweep |
| `requirements-fix-a.txt` | dependencies |
| `outputs/` | generated CSVs and figures |

## What each section does

| § | fixes | report reference |
|---|---|---|
| A1 | transcript ↔ audio alignment audit | the real cause behind §3.7 |
| A2 | corrected duration statistics | §3.7 counts blanks and diacritics as phonemes |
| A3 | G2P fallthrough → lexicon candidates | §5.4, reordered by severity |
| A4 | checkpoint footprint | §1.3, §6.3 include the discriminator |
| A5 | deterministic measurement + noise floor | §3, §4 |
| A6 | `length_scale` sweep | §3.7 length ratio 0.838 |
| A7 | leak-free test split | §9.1 |

### A2 in particular

§3.7 reports *83% of tokens at the minimum one-frame duration* and concludes the duration
predictor has collapsed. That counts tokens, and a token is not a phoneme:

- `add_blank = True` (the `VitsConfig` default, confirmed in every saved `config.json`)
  interleaves a `<BLNK>` between every character, so ~50% of tokens are blanks that correctly
  receive one frame.
- ~10% of the IPA characters are combining marks (`t̪` is `t` + U+032A) and length/stress marks,
  none of which occupies time of its own.

A2 recomputes on duration-bearing phonemes and checks the result against the rate the corpus
actually implies. If the two agree, the predictor is reproducing its training data faithfully
and the collapse finding is a counting artefact.

## What Fix A deliberately does not do

These need the fine-tune, or the audio pipeline, and are out of scope here:

- **Re-segmentation.** Clips are capped near 10 s and cut mid-word — `Download (1).wav` ends
  mid-word and `Download (2).wav` opens with the remainder. Needs VAD over the source audio.
- **The generalisation gap** (+2.6 to +3.9 dB). 860 utterances trained for ~1,170 epochs.
  Only more data fixes this, which is what the pooled multi-speaker stage is for.
- **Dropped phonemes in the pathnirwana model.** Its 41-symbol vocabulary cannot represent
  English at all; that needs a unified vocabulary and a re-train.
- **Sample rate.** 16 kHz is the ceiling unless the original recordings are re-extracted.

## Output

`outputs/A_baseline_summary.csv` is the deliverable. Keep it — it is the baseline the
fine-tune is measured against. Re-run this notebook after the fine-tune and compare like
for like, with the same seeds and the same reference pairs.
