"""
Fix A — deterministic synthesis and spectral measurement.

The existing evaluation cannot detect a real improvement because its numbers
move on their own. EVALUATION_REPORT.md scores two byte-identical checkpoints
(§1.3 confirms the MD5 match) at 10.374 dB and 11.101 dB, and quotes 11.91 dB
and 10.872 dB for the same weights elsewhere — a ~1.5 dB spread from one model.

The cause is that VITS synthesis is stochastic: inference_noise_scale = 0.667
and noise_scale_dp = 1.0, with nothing seeded. This module removes that
variance so a fine-tune can be shown to have worked.

Nothing here modifies a checkpoint.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import fa_core as fa


# ---------------------------------------------------------------------------
# Imports from the vendored Coqui tree
# ---------------------------------------------------------------------------


def _add_vendored_tts_to_path() -> None:
    """Put the vendored TTS and its sibling modules on sys.path.

    model_training/ must come first because formatters.py and vocab_utils.py
    use flat sibling imports, matching how train_custom.py runs.
    """
    training = fa.PROJECT_ROOT / "model_training"
    for path in (training / "TTS", training, fa.PROJECT_ROOT / "model_engine"):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


@contextmanager
def _quiet_torch_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        yield


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A VITS checkpoint ready for measurement, with its tokenizer."""

    name: str
    model: object
    tokenizer: object
    config: object
    checkpoint: Path
    sample_rate: int = fa.SAMPLE_RATE
    _rng_state: dict = field(default_factory=dict, repr=False)


def load_model(run_dir: Path, checkpoint_name: str = "best_model.pth") -> LoadedModel:
    """Load one run's checkpoint on CPU for evaluation.

    *run_dir* is a folder under model_evaluate/model/ containing config.json
    and the checkpoint files.
    """
    _add_vendored_tts_to_path()
    import torch

    with _quiet_torch_warnings():
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    checkpoint = run_dir / checkpoint_name
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json in {run_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No {checkpoint_name} in {run_dir}")

    config = VitsConfig()
    config.load_json(str(config_path))

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    model = Vits(config, ap, tokenizer, speaker_manager=None)
    model.load_checkpoint(config, str(checkpoint), eval=True)
    model.eval()
    torch.set_grad_enabled(False)

    return LoadedModel(
        name=run_dir.name,
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint=checkpoint,
        sample_rate=int(config.audio.sample_rate),
    )


def available_runs(model_dir: Path | None = None) -> pd.DataFrame:
    """List runs and their checkpoints under model_evaluate/model/."""
    model_dir = Path(model_dir or fa.MODEL_DIR)
    rows = []
    for run in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        ckpts = sorted(p.name for p in run.glob("*.pth"))
        cfg = run / "config.json"
        sr = None
        if cfg.is_file():
            try:
                sr = json.loads(cfg.read_text(encoding="utf-8"))["audio"]["sample_rate"]
            except Exception:
                pass
        rows.append(
            {
                "run": run.name,
                "sample_rate": sr,
                "n_checkpoints": len(ckpts),
                "checkpoints": ", ".join(ckpts),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deterministic synthesis
# ---------------------------------------------------------------------------


def synthesize(
    loaded: LoadedModel,
    ipa: str,
    seed: int = fa.SEED,
    deterministic_duration: bool = True,
    length_scale: float = 1.0,
) -> dict:
    """Synthesise one IPA string and return audio plus predicted durations.

    *deterministic_duration* sets noise_scale_dp to 0, which fixes the
    stochastic duration predictor. That is the single largest source of the
    run-to-run spread, and it must be off for any measurement you intend to
    compare across checkpoints.

    Returns the waveform, the per-token durations in frames (w_ceil from
    vits.py inference), and the token ids.
    """
    import torch

    model = loaded.model

    model.length_scale = length_scale
    model.inference_noise_scale_dp = 0.0 if deterministic_duration else 1.0
    model.noise_scale_dp = model.inference_noise_scale_dp

    token_ids = loaded.tokenizer.text_to_ids(ipa)
    x = torch.LongTensor(token_ids).unsqueeze(0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        outputs = model.inference(x)

    wav = outputs["model_outputs"].squeeze().cpu().numpy()
    durations = outputs["durations"].squeeze().cpu().numpy().astype(int)

    return {
        "wav": wav,
        "sample_rate": loaded.sample_rate,
        "durations": durations,
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "seconds": len(wav) / loaded.sample_rate,
    }


def synthesize_repeats(
    loaded: LoadedModel,
    ipa: str,
    n_repeats: int = fa.N_REPEATS,
    base_seed: int = fa.SEED,
    deterministic_duration: bool = True,
) -> list[dict]:
    """Synthesise the same text several times under different seeds.

    With deterministic_duration=True the durations are identical across
    repeats and only the spectral sampling varies, which is what you want
    when estimating the noise floor of a spectral metric.
    """
    return [
        synthesize(loaded, ipa, seed=base_seed + i, deterministic_duration=deterministic_duration)
        for i in range(n_repeats)
    ]


# ---------------------------------------------------------------------------
# Duration statistics, corrected
# ---------------------------------------------------------------------------


def duration_stats(loaded: LoadedModel, ipa: str, deterministic_duration: bool = False, **kwargs) -> dict:
    """Per-phoneme duration statistics for one utterance.

    The report's §3.7 figure ("83% of tokens at the minimum duration") counts
    blanks and combining marks, which legitimately occupy one frame. This
    recomputes on duration-bearing phonemes so the number can be compared
    against real speech.

    Note the default differs from measure_noise_floor: durations are left
    **stochastic** here. Setting noise_scale_dp to 0 makes the stochastic
    duration predictor emit its mode, which collapses the duration spread and
    drives tokens-at-minimum toward 100% — reproducible, but no longer a
    description of the distribution the model actually produces. Use seeds and
    repeats for reproducibility instead, and keep noise_scale_dp = 0 for
    cross-checkpoint spectral comparison only.
    """
    result = synthesize(loaded, ipa, deterministic_duration=deterministic_duration, **kwargs)
    durations = result["durations"]

    phonemes = fa.count_phonemes(ipa)
    total_frames = int(durations.sum())

    return {
        "ipa_chars": len(ipa),
        "phonemes": phonemes,
        "n_tokens": result["n_tokens"],
        "total_frames": total_frames,
        "audio_seconds": round(result["seconds"], 3),
        "frames_per_token": round(total_frames / len(durations), 3) if len(durations) else 0.0,
        "frames_per_phoneme": round(total_frames / phonemes, 3) if phonemes else 0.0,
        "tokens_at_minimum_%": round(100 * float((durations <= 1).mean()), 1),
        "phonemes_per_sec": round(phonemes / result["seconds"], 2) if result["seconds"] else 0.0,
    }


def duration_report(loaded: LoadedModel, ipa_list: list[str], **kwargs) -> pd.DataFrame:
    return pd.DataFrame([duration_stats(loaded, ipa, **kwargs) for ipa in ipa_list])


# ---------------------------------------------------------------------------
# Mel-cepstral distortion
# ---------------------------------------------------------------------------
# Reimplements the definition in EVALUATION_REPORT.md §2.2 so the new numbers
# stay on the same scale as the old ones. This is a filterbank cepstrum, not
# an SPTK mcep envelope, so the values are internally comparable but should
# not be quoted against published MCD figures.


def _mel_cepstrum(wav: np.ndarray, sample_rate: int, n_coeff: int = 13) -> np.ndarray:
    import librosa
    from scipy.fftpack import dct

    wav = np.asarray(wav, dtype=np.float64)
    rms = np.sqrt(np.mean(wav**2)) or 1e-9
    target = 10 ** (-27 / 20)              # normalise both signals to -27 dBFS
    wav = wav * (target / rms)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=fa.N_FFT,
        hop_length=fa.HOP_LENGTH,
        win_length=fa.WIN_LENGTH,
        n_mels=fa.N_MELS,
        fmin=fa.MEL_FMIN,
        fmax=min(fa.MEL_FMAX, sample_rate // 2),
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max, top_db=40)
    ln = db * (np.log(10) / 10.0)          # dB -> natural log

    keep = db.max(axis=0) > -40            # drop frames below -40 dB of peak
    if keep.sum() < 2:
        keep = np.ones(db.shape[1], dtype=bool)
    ln = ln[:, keep]

    cep = dct(ln, type=2, axis=0, norm=None) * (2.0 / fa.N_MELS)
    return cep[1 : 1 + n_coeff]            # c1..c13, energy term c0 discarded


def mcd(reference: np.ndarray, synthesis: np.ndarray, sample_rate: int = fa.SAMPLE_RATE) -> float:
    """Mel-cepstral distortion in dB between a recording and a synthesis.

    The two signals have independent timing, so the cepstra are DTW-aligned
    before the distance is taken.
    """
    import librosa

    ref_c = _mel_cepstrum(reference, sample_rate)
    syn_c = _mel_cepstrum(synthesis, sample_rate)
    if ref_c.shape[1] < 2 or syn_c.shape[1] < 2:
        return float("nan")

    _, path = librosa.sequence.dtw(X=ref_c, Y=syn_c, metric="euclidean")
    diffs = ref_c[:, path[:, 0]] - syn_c[:, path[:, 1]]
    frame_distance = np.sqrt(2.0 * np.sum(diffs**2, axis=0))
    return float((10.0 / np.log(10)) * frame_distance.mean())


def load_wav(path: Path, sample_rate: int = fa.SAMPLE_RATE) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return wav


# ---------------------------------------------------------------------------
# Noise-floor measurement
# ---------------------------------------------------------------------------


def measure_noise_floor(
    loaded: LoadedModel,
    pairs: pd.DataFrame,
    n_repeats: int = fa.N_REPEATS,
    deterministic_duration: bool = True,
) -> pd.DataFrame:
    """Score the same checkpoint repeatedly to find the measurement noise floor.

    *pairs* needs 'wav_path' and 'ipa' columns. Any improvement smaller than
    the spread this reports is not an improvement — it is the metric moving on
    its own. Run this before and after a fine-tune.
    """
    rows = []
    for row in pairs.itertuples():
        reference = load_wav(Path(row.wav_path), loaded.sample_rate)
        for i in range(n_repeats):
            out = synthesize(
                loaded,
                row.ipa,
                seed=fa.SEED + i,
                deterministic_duration=deterministic_duration,
            )
            rows.append(
                {
                    "wav": Path(row.wav_path).name,
                    "repeat": i,
                    "seed": fa.SEED + i,
                    "mcd_db": round(mcd(reference, out["wav"], loaded.sample_rate), 3),
                    "ref_seconds": round(len(reference) / loaded.sample_rate, 3),
                    "syn_seconds": round(out["seconds"], 3),
                    "length_ratio": round(
                        out["seconds"] / (len(reference) / loaded.sample_rate), 3
                    ),
                }
            )
    return pd.DataFrame(rows)


def noise_floor_summary(measurements: pd.DataFrame) -> pd.DataFrame:
    """Per-utterance spread, and the overall floor below which nothing counts."""
    if measurements.empty:
        return pd.DataFrame()
    per_utt = (
        measurements.groupby("wav")["mcd_db"]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .reset_index()
    )
    per_utt["spread"] = (per_utt["max"] - per_utt["min"]).round(3)
    return per_utt


def length_scale_sweep(
    loaded: LoadedModel,
    pairs: pd.DataFrame,
    scales: tuple[float, ...] = (0.9, 1.0, 1.1, 1.2, 1.3),
) -> pd.DataFrame:
    """Find the length_scale that best matches the reference recording length.

    A training-free partial remedy for the 0.838 length ratio in §3.7: the
    synthesis is systematically ~16% shorter than the recording, and
    length_scale is an inference-time multiplier on predicted durations.

    This changes rate uniformly, so it cannot restore rhythm that the duration
    predictor never learned — but it costs nothing and it is measurable.
    """
    rows = []
    for scale in scales:
        ratios, mcds = [], []
        for row in pairs.itertuples():
            reference = load_wav(Path(row.wav_path), loaded.sample_rate)
            ref_seconds = len(reference) / loaded.sample_rate
            out = synthesize(loaded, row.ipa, length_scale=scale)
            ratios.append(out["seconds"] / ref_seconds)
            mcds.append(mcd(reference, out["wav"], loaded.sample_rate))
        rows.append(
            {
                "length_scale": scale,
                "mean_length_ratio": round(float(np.mean(ratios)), 3),
                "mean_mcd_db": round(float(np.mean(mcds)), 3),
                "n": len(ratios),
            }
        )
    return pd.DataFrame(rows)
