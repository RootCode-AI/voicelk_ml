"""
Fix A — training-free diagnostics for VoiceLK TTS.

"Fix A" is the set of repairs that do NOT touch model weights: measurement
corrections, data audits and text front-end checks. Nothing here trains,
fine-tunes or modifies a checkpoint.

Run this before any fine-tuning. Its job is to produce a baseline you can
actually trust, because the numbers in model_evaluate/EVALUATION_REPORT.md
currently cannot be compared against anything (§4 scores byte-identical
checkpoints 0.7 dB apart, which is measurement noise, not model difference).

Paths are resolved from this file's location, so the module works regardless
of the notebook's working directory.
"""

from __future__ import annotations

import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    import sys

    raise ImportError(
        f"{exc}\n\n"
        f"  This kernel cannot run Fix A.\n"
        f"  interpreter: {sys.executable}\n"
        f"  python     : {sys.version.split()[0]}\n\n"
        "  Select a kernel whose interpreter is inside one of the project venvs:\n"
        "    'VoiceLK (venv_evoluation 3.14)'  - works for every Fix A section\n"
        "    'VoiceLK Fix A (venv 3.11)'       - also unlocks pesq/speechbrain/UTMOS\n\n"
        "  In VS Code the kernel picker is at the top right of the notebook.\n"
        "  Choosing a global Python instead is what produces this error."
    ) from exc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent          # voicelk_ml/model_finetune
PROJECT_ROOT = MODULE_DIR.parent                      # voicelk_ml
WORKSPACE_ROOT = PROJECT_ROOT.parent                  # .../TTS

METADATA_FILE = PROJECT_ROOT / "data" / "custom_metadata.txt"
DATASET_CSV = PROJECT_ROOT / "data" / "custom_dataset.csv"

# The wavs live one level above the repo, next to it — not in voicelk_ml/data.
WAV_ROOT = WORKSPACE_ROOT / "data" / "wavs"

MODEL_DIR = PROJECT_ROOT / "model_evaluate" / "model"
OUTPUT_DIR = MODULE_DIR / "outputs"

# ---------------------------------------------------------------------------
# Audio / frame constants — must match the training config, not be guessed.
# See model_evaluate/model/custom/config.json
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_FFT = 1024
WIN_LENGTH = 1024
N_MELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000

FRAME_SECONDS = HOP_LENGTH / SAMPLE_RATE              # 0.016 s = 16 ms

# Deterministic-measurement defaults (see fa_inference.py).
SEED = 1234
N_REPEATS = 5

# ---------------------------------------------------------------------------
# Phoneme counting
# ---------------------------------------------------------------------------
# The report's §3.7 "duration predictor has collapsed" finding counts *tokens*,
# but a token is not a phoneme. Two things inflate the token count:
#
#   1. add_blank=True (VitsConfig default) interleaves a <BLNK> between every
#      character, so ~50% of tokens are blanks that correctly get 1 frame.
#   2. IPA is stored as raw codepoints, so "t̪" is t + U+032A (combining) and
#      "aː" is a + U+02D0. Those marks have no duration of their own.
#
# Counting only duration-bearing characters gives frames-per-phoneme, which is
# the quantity that can actually be compared against human speech.

NON_DURATION_MARKS = frozenset(
    "ː"  # ː  length
    "ˈ"  # ˈ  primary stress
    "ˌ"  # ˌ  secondary stress
    "ʰ"  # ʰ  aspiration
)


def is_duration_bearing(ch: str) -> bool:
    """True if *ch* is a phoneme that occupies time in the signal.

    Excludes whitespace, Unicode combining marks (the dental diacritic in t̪,
    d̪) and the length/stress/aspiration modifiers. A long vowel "aː" counts
    once, with its extra duration attributed to the base vowel.
    """
    if ch.isspace():
        return False
    if unicodedata.combining(ch):
        return False
    return ch not in NON_DURATION_MARKS


def count_phonemes(ipa: str) -> int:
    """Number of duration-bearing phonemes in an IPA string."""
    return sum(1 for ch in ipa if is_duration_bearing(ch))


def token_count_with_blank(ipa: str) -> int:
    """Token count as the VITS tokenizer sees it, with add_blank=True.

    Coqui interleaves a blank between every character and adds one at each
    end, giving 2N+1 for an N-character sequence.
    """
    return 2 * len(ipa) + 1


# ---------------------------------------------------------------------------
# Speaking-rate thresholds
# ---------------------------------------------------------------------------
# Conversational speech runs ~12-14 phonemes/s; fast speech reaches ~16-18.
# Sustained rates above ~22 are not physically plausible and indicate the
# transcript covers more speech than the clip contains.

RATE_COMFORTABLE = 14.0
RATE_FAST = 18.0
RATE_IMPLAUSIBLE = 22.0


def rate_verdict(phonemes_per_sec: float) -> str:
    if phonemes_per_sec > RATE_IMPLAUSIBLE:
        return "implausible"
    if phonemes_per_sec > RATE_FAST:
        return "very_fast"
    if phonemes_per_sec > RATE_COMFORTABLE:
        return "fast"
    return "ok"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    """Resolved input paths, with a human-readable validation report."""

    metadata: Path
    dataset_csv: Path
    wav_root: Path
    model_dir: Path
    output_dir: Path

    def check(self) -> pd.DataFrame:
        rows = []
        for name, path, kind in [
            ("metadata", self.metadata, "file"),
            ("dataset_csv", self.dataset_csv, "file"),
            ("wav_root", self.wav_root, "dir"),
            ("model_dir", self.model_dir, "dir"),
        ]:
            ok = path.is_file() if kind == "file" else path.is_dir()
            extra = ""
            if ok and kind == "dir" and name == "wav_root":
                extra = f"{len(list(path.glob('*.wav')))} wav files"
            rows.append({"name": name, "path": str(path), "found": ok, "note": extra})
        return pd.DataFrame(rows)


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "librosa",
    "soundfile",
    "torch",
    "matplotlib",
    "mutagen",      # imported by TTS/tts/datasets/dataset.py
    "coqpit",       # config objects for the vendored Coqui tree
)

OPTIONAL_MODULES = ("sinling", "eng_to_ipa")   # A3 only


def check_environment() -> pd.DataFrame:
    """Report which interpreter is running and whether it can do the work.

    Run this first. A missing dependency otherwise surfaces deep inside the
    vendored Coqui import chain as an unrelated-looking ModuleNotFoundError —
    'No module named mutagen' raised from vits_config.py, for instance, which
    says nothing about the real problem being the selected kernel.
    """
    import importlib
    import sys

    rows = []
    for name in REQUIRED_MODULES + OPTIONAL_MODULES:
        required = name in REQUIRED_MODULES
        try:
            mod = importlib.import_module(name)
            rows.append(
                {
                    "module": name,
                    "required": required,
                    "found": True,
                    "version": getattr(mod, "__version__", ""),
                }
            )
        except ImportError:
            rows.append({"module": name, "required": required, "found": False, "version": ""})

    df = pd.DataFrame(rows)
    missing = df[df["required"] & ~df["found"]]["module"].tolist()

    print(f"interpreter : {sys.executable}")
    print(f"python      : {sys.version.split()[0]}")
    if missing:
        print()
        print("  MISSING REQUIRED MODULES:", ", ".join(missing))
        print("  The notebook is almost certainly running on the wrong kernel.")
        print("  Pick a kernel whose interpreter is inside one of the project venvs:")
        print("    VoiceLK (venv_evoluation 3.14)  — works for every Fix A section")
        print("    VoiceLK Fix A (venv 3.11)       — also unlocks pesq/speechbrain/UTMOS")
    else:
        print("  all required modules present")
    return df


def default_paths() -> Paths:
    return Paths(
        metadata=METADATA_FILE,
        dataset_csv=DATASET_CSV,
        wav_root=WAV_ROOT,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
    )


def wav_duration(path: Path) -> float:
    """Clip length in seconds, read from the header only (no decoding)."""
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def wav_sample_rate(path: Path) -> int:
    with wave.open(str(path)) as w:
        return w.getframerate()


def load_metadata(metadata: Path | None = None, wav_root: Path | None = None) -> pd.DataFrame:
    """Read custom_metadata.txt into a frame and attach wav paths.

    Splits on the first '|' only. A plain split('|') would truncate the IPA on
    lines where it legitimately contains '|' — that is the exact bug that gave
    System C its 234 phantom speakers (EVALUATION_REPORT.md §1.5).
    """
    metadata = metadata or METADATA_FILE
    wav_root = wav_root or WAV_ROOT

    rows = []
    with open(metadata, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rel, sep, ipa = line.partition("|")
            if not sep:
                continue
            name = rel.replace("wavs/", "").replace("wavs\\", "")
            path = wav_root / name
            rows.append(
                {
                    "line": lineno,
                    "wav": name,
                    "wav_path": path,
                    "wav_exists": path.is_file(),
                    "ipa": ipa,
                }
            )
    return pd.DataFrame(rows)


def ensure_output_dir(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save(df: pd.DataFrame, name: str, output_dir: Path | None = None) -> Path:
    """Write a result table and return where it landed."""
    out = ensure_output_dir(output_dir) / name
    df.to_csv(out, index=False, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# A1 — transcript / audio alignment audit
# ---------------------------------------------------------------------------


def audit_alignment(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-utterance speaking rate, to find transcripts that outrun their audio.

    A transcript longer than its clip forces the aligner to squeeze every
    phoneme toward the one-frame minimum. That is the mechanism behind the
    "duration collapse" reported in §3.7, so it has to be measured directly
    rather than inferred from the token histogram.
    """
    records = []
    for row in meta.itertuples():
        if not row.wav_exists:
            continue
        seconds = wav_duration(row.wav_path)
        phonemes = count_phonemes(row.ipa)
        chars = len(row.ipa)
        tokens = token_count_with_blank(row.ipa)
        frames = seconds / FRAME_SECONDS
        records.append(
            {
                "wav": row.wav,
                "seconds": round(seconds, 3),
                "sample_rate": wav_sample_rate(row.wav_path),
                "ipa_chars": chars,
                "phonemes": phonemes,
                "tokens_with_blank": tokens,
                "frames": round(frames, 1),
                "phonemes_per_sec": round(phonemes / seconds, 2) if seconds else 0.0,
                "frames_per_phoneme": round(frames / phonemes, 3) if phonemes else 0.0,
                "frames_per_token": round(frames / tokens, 3) if tokens else 0.0,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["verdict"] = df["phonemes_per_sec"].map(rate_verdict)
    return df.sort_values("phonemes_per_sec", ascending=False).reset_index(drop=True)


def alignment_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Percentiles and verdict counts for the alignment audit."""
    if audit.empty:
        return pd.DataFrame()
    q = audit["phonemes_per_sec"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    rows = [{"metric": f"phonemes_per_sec p{int(p * 100)}", "value": round(v, 2)} for p, v in q.items()]
    rows += [
        {"metric": "frames_per_phoneme (median)", "value": round(audit["frames_per_phoneme"].median(), 3)},
        {"metric": "frames_per_token (median)", "value": round(audit["frames_per_token"].median(), 3)},
        {"metric": "clip seconds (median)", "value": round(audit["seconds"].median(), 2)},
        {"metric": "clip seconds (max)", "value": round(audit["seconds"].max(), 2)},
        {"metric": "utterances", "value": len(audit)},
    ]
    for verdict, n in audit["verdict"].value_counts().items():
        rows.append({"metric": f"verdict: {verdict}", "value": f"{n} ({100 * n / len(audit):.1f}%)"})
    return pd.DataFrame(rows)


def clean_subset(audit: pd.DataFrame, max_rate: float = RATE_IMPLAUSIBLE) -> pd.DataFrame:
    """Utterances whose transcript plausibly matches their audio."""
    return audit[audit["phonemes_per_sec"] <= max_rate].reset_index(drop=True)


# ---------------------------------------------------------------------------
# A2 — G2P fallthrough audit
# ---------------------------------------------------------------------------


def audit_g2p_fallthrough(meta: pd.DataFrame, dataset_csv: Path | None = None) -> pd.DataFrame:
    """English words that reached the IPA column as raw spelling.

    When eng_to_ipa has no entry for a word (brand names, compounds) it returns
    the word unchanged, and prepare_dataset.py writes it through silently. The
    model then learns Latin letters as if they were phonemes — which is where
    'A B C G H I M P S T' in the 90-symbol vocabulary came from.

    These are the highest-value lexicon entries to add, ahead of the
    frequency-ordered OOV list the report recommends in §10.5, because an OOV
    word merely has a guessed pronunciation while these have none at all.
    """
    import csv
    import re

    dataset_csv = dataset_csv or DATASET_CSV
    ipa_by_wav = {r.wav: r.ipa for r in meta.itertuples()}

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}

    with open(dataset_csv, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            wav = row["file_name"].strip()
            ipa = ipa_by_wav.get(wav)
            if not ipa:
                continue
            ipa_tokens = set(ipa.lower().split())
            source = row.get("sinhala_transcript_with_code_switch", "")
            for word in set(re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", source)):
                if word.lower() in ipa_tokens:
                    counts[word] = counts.get(word, 0) + 1
                    examples.setdefault(word, wav)

    df = pd.DataFrame(
        [{"word": w, "count": c, "example_wav": examples[w]} for w, c in counts.items()]
    )
    if df.empty:
        return df
    return df.sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# A3 — checkpoint footprint, discriminator excluded
# ---------------------------------------------------------------------------

# Parameter-name prefixes that exist only for adversarial training and are
# never loaded at synthesis time.
TRAINING_ONLY_PREFIXES = ("disc",)


def audit_footprint(checkpoint: Path) -> pd.DataFrame:
    """Per-module parameter counts, separating deployable from training-only.

    EVALUATION_REPORT.md §1.3 and §6.3 quote "weights only, fp32 = 316.8 MB",
    but that figure includes the discriminator, which is ~56% of the
    checkpoint and is not part of the served model.
    """
    import torch

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    weights = state["model"]

    groups: dict[str, int] = {}
    for key, tensor in weights.items():
        module = key.split(".")[0]
        groups[module] = groups.get(module, 0) + tensor.numel()

    total = sum(groups.values())
    rows = []
    for module, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        deployable = not module.startswith(TRAINING_ONLY_PREFIXES)
        rows.append(
            {
                "module": module,
                "parameters": n,
                "share_%": round(100 * n / total, 2),
                "fp32_MB": round(n * 4 / 1024**2, 1),
                "deployed": deployable,
            }
        )
    return pd.DataFrame(rows)


def footprint_summary(footprint: pd.DataFrame) -> pd.DataFrame:
    deployed = footprint[footprint["deployed"]]["parameters"].sum()
    training = footprint[~footprint["deployed"]]["parameters"].sum()
    total = deployed + training
    return pd.DataFrame(
        [
            {"quantity": "total parameters in checkpoint", "value": f"{total:,}"},
            {"quantity": "training-only (discriminator)", "value": f"{training:,}"},
            {"quantity": "deployable parameters", "value": f"{deployed:,}"},
            {"quantity": "deployable fp32", "value": f"{deployed * 4 / 1024**2:.1f} MB"},
            {"quantity": "deployable fp16", "value": f"{deployed * 2 / 1024**2:.1f} MB"},
        ]
    )


# ---------------------------------------------------------------------------
# A4 — leak-free test split
# ---------------------------------------------------------------------------


def source_id(wav_name: str) -> str:
    """Group key for a clip.

    Clips cut from one recording are highly correlated, so a random split
    leaks training audio into the test set. Grouping by source and holding out
    whole sources removes that leak.

    The current corpus is named 'Download (N).wav' with no recoverable source
    id, so every clip becomes its own group and this function is a no-op until
    re-segmentation attaches a real source. Replace the body once the
    segmentation step names clips '<source>_<index>.wav'.
    """
    stem = Path(wav_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def build_test_split(audit: pd.DataFrame, fraction: float = 0.05, seed: int = SEED) -> pd.DataFrame:
    """Hold out whole sources, not random clips.

    Returns the audit frame with a 'split' column. Uses only clips that passed
    the alignment audit, so the test set measures the model rather than the
    transcript defects.
    """
    import numpy as np

    usable = clean_subset(audit).copy()
    usable["source"] = usable["wav"].map(source_id)

    sources = sorted(usable["source"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(sources)

    n_test = max(1, int(round(fraction * len(sources))))
    test_sources = set(sources[:n_test])

    usable["split"] = ["test" if s in test_sources else "train" for s in usable["source"]]
    return usable.reset_index(drop=True)
