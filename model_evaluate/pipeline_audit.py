"""
Train → checkpoint → inference audit for a VoiceLK VITS run.

Answers one question: can an exact training sample pass through the whole pipeline
and be reproduced? Each subcommand isolates one stage, so a failure points at it.

  data           every metadata line has its wav; sample rate, channels, duration,
                 clipping, loudness, silence, and speaking rate (IPA characters per
                 second) to catch transcripts paired with the wrong recording
  equivalence    token IDs from the training DataLoader path vs the inference path,
                 for every metadata line (no audio needed)
  deterministic  synthesise one text with all sampling noise off, several times, and
                 check the waveform is identical every run (no audio needed)
  reconstruct    ground-truth audio → posterior encoder → decoder.  Bypasses the text
                 encoder, prior and duration predictor entirely
  prior          for one training clip: MAS alignment and durations vs the duration
                 predictor, prior vs posterior statistics, and synthesis with the
                 ground-truth durations (isolates the duration predictor)
  all            every test that the given inputs allow

Run from voicelk_ml/model_training (so the vendored TTS package resolves), e.g.

  python ../model_evaluate/pipeline_audit.py all \
      --run ../../Models/speaker_01 \
      --metadata ../data/prepared/speaker_01_metadata.txt \
      --csv ../data/prepared/speaker_01.csv \
      --wav-dir <folder with sin_s1_0001.wav ...> \
      --sample sin_s1_0001 --out audit_out

Exit status is 1 when a check that must hold (identical tokens, deterministic output,
clean checkpoint load, matching sample rate) fails.
"""

import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ML_ROOT, "model_training"))
sys.path.insert(0, os.path.join(ML_ROOT, "model_engine"))

from TTS.tts.configs.vits_config import VitsConfig  # noqa: E402
from TTS.tts.models.vits import Vits, spec_to_mel, wav_to_spec  # noqa: E402
from TTS.tts.utils.helpers import sequence_mask  # noqa: E402
from TTS.tts.utils.text.tokenizer import TTSTokenizer  # noqa: E402
from TTS.utils.audio import AudioProcessor  # noqa: E402

from formatters import custom_formatter  # noqa: E402
from pipeline import TextProcessingPipeline  # noqa: E402
from vocab_utils import build_characters_config  # noqa: E402

FAILURES = []


def check(ok: bool, message: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + message)
    if not ok:
        FAILURES.append(message)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_checkpoint(run_dir: str, name: str = None) -> str:
    if name:
        return os.path.join(run_dir, name)
    pths = [f for f in os.listdir(run_dir) if f.endswith(".pth") and "speakers" not in f.lower()]
    if "best_model.pth" in pths:
        return os.path.join(run_dir, "best_model.pth")
    return os.path.join(run_dir, sorted(pths)[-1])


def load_run(run_dir: str, checkpoint: str = None):
    """Loads a run exactly as bridge/app/engine.py and synthesize.py do, and reports
    the checkpoint identity and any key mismatch instead of trusting strict loading."""
    config = VitsConfig()
    config.load_json(os.path.join(run_dir, "config.json"))
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    path = find_checkpoint(run_dir, checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state["model"], strict=False)
    model.eval()

    print(f"Checkpoint : {path}")
    print(f"Step/epoch : {state.get('step')} / {state.get('epoch')}   saved losses: {state.get('model_loss')}")
    check(not result.missing_keys, f"no missing keys ({len(result.missing_keys)})")
    check(not result.unexpected_keys, f"no unexpected keys ({len(result.unexpected_keys)})")
    check(not model.training, "model is in eval mode")
    return config, ap, tokenizer, model


def read_transcripts(csv_path: str) -> dict:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column = [c for c in reader.fieldnames if c != "file_name"][0]
        return {row["file_name"].removesuffix(".wav"): row[column] for row in reader}


def deterministic(model, on: bool):
    """All sampling noise off (prior noise and duration-predictor noise), unit length."""
    if on:
        model.inference_noise_scale, model.inference_noise_scale_dp, model.length_scale = 0.0, 0.0, 1.0
    else:
        args = model.args
        model.inference_noise_scale = args.inference_noise_scale
        model.inference_noise_scale_dp = args.inference_noise_scale_dp
        model.length_scale = args.length_scale


def synth(model, token_ids, durations=None):
    x = torch.LongTensor(token_ids).unsqueeze(0)
    aux = {"x_lengths": None, "d_vectors": None, "speaker_ids": None, "language_ids": None, "durations": durations}
    with torch.no_grad():
        return model.inference(x, aux_input=aux)


def mel_of(config, wav: torch.Tensor) -> torch.Tensor:
    """Log-mel exactly as training computes it for the mel loss (vits.py format_batch_on_device)."""
    ac = config.audio
    spec = wav_to_spec(wav.view(1, 1, -1), ac.fft_size, ac.hop_length, ac.win_length, center=False)
    return spec_to_mel(spec, ac.fft_size, ac.num_mels, ac.sample_rate, ac.mel_fmin, ac.mel_fmax)


def mel_l1(config, a: torch.Tensor, b: torch.Tensor) -> float:
    ma, mb = mel_of(config, a), mel_of(config, b)
    n = min(ma.shape[-1], mb.shape[-1])
    return float(torch.nn.functional.l1_loss(ma[..., :n], mb[..., :n]))


def load_wav(path: str, config) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    info = sf.info(path)
    print(f"  wav: {os.path.basename(path)}  {sr} Hz  {data.shape[1]} ch  {info.subtype}  "
          f"{len(data) / sr:.2f}s  peak {np.abs(data).max():.3f}")
    check(sr == config.audio.sample_rate, f"wav sample rate {sr} == config {config.audio.sample_rate}")
    check(data.shape[1] == 1, "wav is mono (training reads only what the file holds)")
    return torch.from_numpy(data[:, 0].copy())


def save(path: str, wav, sr: int) -> None:
    wav = wav.detach().cpu().numpy().reshape(-1) if torch.is_tensor(wav) else np.asarray(wav).reshape(-1)
    sf.write(path, wav, sr)
    print(f"  saved {path}")


def save_alignment(path: str, attn: torch.Tensor, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing; alignment image skipped)")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(attn.cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
    ax.set_xlabel("spectrogram frame")
    ax.set_ylabel("token (blanks included)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_equivalence(args, tokenizer):
    """Training side: the metadata text (what the DataLoader read) through the tokenizer
    the run saved in config.json (the one the trainer used). Inference side: the raw
    transcript through the front-end and the same tokenizer, as the bridge does."""
    print("\n== Training vs inference preprocessing ==")
    meta_dir, meta_file = os.path.split(os.path.abspath(args.metadata))
    cfg_now = VitsConfig(text_cleaner="basic_cleaners", use_phonemes=False,
                         characters=build_characters_config(args.metadata))
    tok_now, _ = TTSTokenizer.init_from_config(cfg_now)
    if tok_now.characters.vocab != tokenizer.characters.vocab:
        added = sorted(set(tok_now.characters.vocab) - set(tokenizer.characters.vocab))
        print(f"  INFO  vocab_utils.py would build a different vocabulary today (adds {added}); "
              "a run trained after that change uses it, this checkpoint does not")

    pipeline = TextProcessingPipeline()
    raw = read_transcripts(args.csv)
    mismatches = 0
    samples = custom_formatter(meta_dir, meta_file)
    for sample in samples:
        name = os.path.basename(sample["audio_file"]).removesuffix(".wav")
        ids_train = tokenizer.text_to_ids(sample["text"])
        ids_infer = tokenizer.text_to_ids(pipeline.process(raw[name])["ipa_sequence"])
        if ids_train != ids_infer:
            mismatches += 1
            print(f"    differs: {name}")
    check(mismatches == 0, f"token IDs identical for {len(samples) - mismatches}/{len(samples)} metadata lines "
                           "(metadata IPA vs front-end on the raw transcript)")


def test_data(args, config):
    print("\n== Data quality ==")
    meta_dir, meta_file = os.path.split(os.path.abspath(args.metadata))
    rows, missing = [], []
    for sample in custom_formatter(meta_dir, meta_file):
        name = os.path.basename(sample["audio_file"])
        path = os.path.join(args.wav_dir, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        frame = max(1, sr // 100)                                   # 10 ms frames
        energy = np.array([np.sqrt(np.mean(mono[i:i + frame] ** 2) + 1e-12) for i in range(0, len(mono), frame)])
        db = 20 * np.log10(energy)
        voiced = np.where(db > db.max() - 40)[0]
        lead = voiced[0] * 10 if len(voiced) else 0
        trail = (len(db) - 1 - voiced[-1]) * 10 if len(voiced) else 0
        seconds = len(mono) / sr
        rows.append((name, sr, data.shape[1], seconds, float(np.abs(mono).max()),
                     float(np.mean(np.abs(mono) >= 0.999)), float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-12)),
                     float(np.mean(db < db.max() - 40)), lead, trail, len(sample["text"].replace(" ", "")) / seconds))

    check(not missing, f"every metadata line has its wav ({len(missing)} missing{': ' + ', '.join(missing[:5]) if missing else ''})")
    if not rows:
        return
    print(f"  {'clip':18s} {'Hz':>6s} {'ch':>2s} {'sec':>6s} {'peak':>5s} {'clip%':>6s} {'rmsdB':>6s} "
          f"{'sil%':>5s} {'leadms':>6s} {'trailms':>7s} {'ipa/s':>6s}")
    rates = np.array([r[-1] for r in rows])
    median = float(np.median(rates))
    for r in rows:
        flag = "  <- text/audio length mismatch?" if abs(r[-1] - median) > 0.35 * median else ""
        print(f"  {r[0]:18s} {r[1]:6d} {r[2]:2d} {r[3]:6.2f} {r[4]:5.2f} {100 * r[5]:6.2f} {r[6]:6.1f} "
              f"{100 * r[7]:5.1f} {r[8]:6d} {r[9]:7d} {r[10]:6.1f}{flag}")
    check(all(r[1] == config.audio.sample_rate for r in rows), f"all clips at {config.audio.sample_rate} Hz")
    check(all(r[5] < 0.001 for r in rows), "no clip has more than 0.1% clipped samples")
    check(all(r[6] > -35 for r in rows), "no clip quieter than -35 dBFS RMS")
    durations = [r[3] for r in rows]
    print(f"  duration min/median/max {min(durations):.2f}/{np.median(durations):.2f}/{max(durations):.2f}s, "
          f"total {sum(durations) / 60:.1f} min; IPA characters per second median {median:.1f}")


def test_deterministic(args, config, tokenizer, model, text: str, tag: str):
    print("\n== Deterministic inference ==")
    ids = tokenizer.text_to_ids(TextProcessingPipeline().process(text)["ipa_sequence"])
    deterministic(model, True)
    runs = [synth(model, ids)["model_outputs"][0, 0] for _ in range(3)]
    diff = max(float((runs[0] - r).abs().max()) for r in runs[1:])
    check(diff == 0.0, f"3 runs with noise 0 are identical (max abs diff {diff:.2e})")
    save(os.path.join(args.out, f"{tag}_deterministic.wav"), runs[0], config.audio.sample_rate)

    deterministic(model, False)
    torch.manual_seed(0)
    noisy = [synth(model, ids)["model_outputs"][0, 0] for _ in range(3)]
    n = min(len(w) for w in noisy)
    spread = max(mel_l1(config, noisy[0][:n], w[:n]) for w in noisy[1:])
    lengths = [f"{len(w) / config.audio.sample_rate:.2f}s" for w in noisy]
    print(f"  default noise ({model.args.inference_noise_scale}/{model.args.inference_noise_scale_dp}): "
          f"durations {lengths}, mel-L1 spread between runs {spread:.3f}")
    save(os.path.join(args.out, f"{tag}_default_noise.wav"), noisy[0], config.audio.sample_rate)
    return runs[0]


def test_reconstruct(args, config, model, wav: torch.Tensor, tag: str):
    print("\n== Ground-truth reconstruction (posterior encoder → decoder) ==")
    ac = config.audio
    spec = wav_to_spec(wav.view(1, 1, -1), ac.fft_size, ac.hop_length, ac.win_length, center=False)
    lengths = torch.LongTensor([spec.shape[-1]])
    with torch.no_grad():
        _, m_q, _, y_mask = model.posterior_encoder(spec, lengths, g=None)
        recon = model.waveform_decoder(m_q * y_mask, g=None)[0, 0]
    l1 = mel_l1(config, wav, recon)
    print(f"  mel L1 (log-mel, unweighted; training logs this ×{config.mel_loss_alpha}): {l1:.3f}"
          f"  ≙ loss_mel {l1 * config.mel_loss_alpha:.1f}")
    save(os.path.join(args.out, f"{tag}_reconstructed.wav"), recon, ac.sample_rate)
    save(os.path.join(args.out, f"{tag}_ground_truth.wav"), wav, ac.sample_rate)
    return l1


def test_prior(args, config, tokenizer, model, wav: torch.Tensor, ipa: str, tag: str):
    print("\n== Prior vs posterior, alignment and durations ==")
    ac = config.audio
    ids = tokenizer.text_to_ids(ipa)
    x = torch.LongTensor(ids).unsqueeze(0)
    x_lengths = torch.LongTensor([len(ids)])
    spec = wav_to_spec(wav.view(1, 1, -1), ac.fft_size, ac.hop_length, ac.win_length, center=False)
    y_lengths = torch.LongTensor([spec.shape[-1]])
    with torch.no_grad():
        out = model.forward(x, x_lengths, spec, y_lengths, wav.view(1, 1, -1),
                            aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None})
        attn = out["alignments"][0]                       # [T_seq, T_dec]
        mas_dur = attn.sum(1)                              # frames per token from MAS

        x_enc, _, _, x_mask = model.text_encoder(x, x_lengths)
        if model.args.use_sdp:
            logw = model.duration_predictor(x_enc, x_mask, g=None, reverse=True, noise_scale=0.0)
        else:
            logw = model.duration_predictor(x_enc, x_mask, g=None)
        pred_dur = torch.ceil(torch.exp(logw) * x_mask)[0, 0]

    blank = tokenizer.characters.blank_id
    is_blank = torch.tensor([i == blank for i in ids])
    print(f"  tokens {len(ids)} ({int(is_blank.sum())} blanks)  frames {spec.shape[-1]}  "
          f"MAS total {int(mas_dur.sum())}  predicted total {int(pred_dur.sum())}")
    print(f"  tokens MAS gave 0 frames: {int((mas_dur == 0).sum())} "
          f"(non-blank: {int(((mas_dur == 0) & ~is_blank).sum())})")
    print(f"  mean frames/token  MAS: blank {mas_dur[is_blank].mean():.2f}  phoneme {mas_dur[~is_blank].mean():.2f}"
          f"   predicted: blank {pred_dur[is_blank].mean():.2f}  phoneme {pred_dur[~is_blank].mean():.2f}")
    print(f"  duration MAE predicted vs MAS: {float((pred_dur - mas_dur).abs().mean()):.2f} frames/token")

    z_p, m_p, logs_p = out["z_p"], out["m_p"], out["logs_p"]
    mask = sequence_mask(y_lengths, None).unsqueeze(1).to(z_p.dtype)
    gap = float((((z_p - m_p) * torch.exp(-logs_p)) ** 2 * mask).sum() / (mask.sum() * z_p.shape[1]))
    print(f"  flow(posterior) vs prior, mean squared z-score: {gap:.3f}  (≈1 when the prior has learned the posterior)")
    print(f"  posterior m_q mean/std {float(out['m_q'].mean()):.3f}/{float(out['m_q'].std()):.3f}   "
          f"prior m_p mean/std {float(m_p.mean()):.3f}/{float(m_p.std()):.3f}")

    save_alignment(os.path.join(args.out, f"{tag}_alignment_mas.png"), attn, f"{tag}: MAS alignment")
    deterministic(model, True)
    gt = synth(model, ids, durations=mas_dur.view(1, -1))["model_outputs"][0, 0]
    l1 = mel_l1(config, wav, gt)
    print(f"  synthesis with ground-truth (MAS) durations: mel L1 {l1:.3f}")
    save(os.path.join(args.out, f"{tag}_gt_durations.wav"), gt, ac.sample_rate)
    pred = synth(model, ids)
    print(f"  synthesis with predicted durations: mel L1 {mel_l1(config, wav, pred['model_outputs'][0, 0]):.3f}"
          " (length differs, so only a rough figure)")
    save_alignment(os.path.join(args.out, f"{tag}_alignment_predicted.png"), pred["alignments"][0],
                   f"{tag}: predicted durations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=["equivalence", "data", "deterministic", "reconstruct", "prior", "all"])
    parser.add_argument("--run", required=True, help="Run folder with config.json and a checkpoint")
    parser.add_argument("--checkpoint", help="Checkpoint file name inside --run (default best_model.pth)")
    parser.add_argument("--metadata", help="Training metadata (wavs/name.wav|IPA)")
    parser.add_argument("--csv", help="Transcript CSV the metadata was made from")
    parser.add_argument("--wav-dir", help="Folder holding the training wavs")
    parser.add_argument("--sample", help="File name (without .wav) of the training clip to test")
    parser.add_argument("--text", help="Text for the deterministic test (default: the sample's transcript)")
    parser.add_argument("--out", default="audit_out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    config, _, tokenizer, model = load_run(args.run, args.checkpoint)
    wanted = {args.test} if args.test != "all" else {"equivalence", "data", "deterministic", "reconstruct", "prior"}
    tag = args.sample or "text"

    if "equivalence" in wanted and args.metadata and args.csv:
        test_equivalence(args, tokenizer)

    if "data" in wanted and args.metadata and args.wav_dir:
        test_data(args, config)

    text = args.text or (read_transcripts(args.csv)[args.sample] if args.csv and args.sample else None)
    if "deterministic" in wanted and text:
        test_deterministic(args, config, tokenizer, model, text, tag)

    wav = None
    if wanted & {"reconstruct", "prior"} and args.wav_dir and args.sample:
        wav = load_wav(os.path.join(args.wav_dir, f"{args.sample}.wav"), config)
    if "reconstruct" in wanted and wav is not None:
        test_reconstruct(args, config, model, wav, tag)
    if "prior" in wanted and wav is not None and args.csv:
        ipa = TextProcessingPipeline().process(read_transcripts(args.csv)[args.sample])["ipa_sequence"]
        test_prior(args, config, tokenizer, model, wav, ipa, tag)

    print(f"\n{len(FAILURES)} failed check(s)" + ("".join(f"\n  - {f}" for f in FAILURES) if FAILURES else ""))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
