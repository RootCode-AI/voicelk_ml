"""
Trains a single-speaker VITS model on one of VoiceLK's recorded speakers (speaker_01,
speaker_02, ...).

Unlike train_custom.py / train_pathnirwana.py, the data location, run name and output
folder are command-line options, because this script is meant to run on Kaggle, where
the audio is a read-only input dataset and outputs must go under /kaggle/working.

Expects:
  <data-path>/<metadata>   -> produced by data_preparation/prepare_speaker_dataset.py
  <data-path>/wavs/*.wav   -> the audio referenced by the metadata, all at one sample rate

Usage (from the project root):
  python model_training/train_speaker.py --data-path data --metadata speaker_01_metadata.txt
  # resume an interrupted run (the trainer reads the run folder's config and last checkpoint):
  python model_training/train_speaker.py ... --continue_path <run folder>
  # memorisation test on a metadata file with one or a few clips, full precision:
  python model_training/train_speaker.py ... --metadata one_clip.txt --memorize --fp32 \
      --deterministic-durations --epochs 10000 --save-step 500 --keep-checkpoints 20
"""

import argparse
import os

# Must be set before any CUDA-touching import — see train_custom.py. This script isn't
# set up for multi-GPU/DDP training, so it uses one GPU of Kaggle's "GPU T4 x2".
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from trainer import Trainer, TrainerArgs  # noqa: E402

from TTS.tts.configs.shared_configs import BaseDatasetConfig  # noqa: E402
from TTS.tts.configs.vits_config import VitsConfig  # noqa: E402
from TTS.tts.datasets import load_tts_samples  # noqa: E402
from TTS.tts.models.vits import Vits, VitsAudioConfig  # noqa: E402
from TTS.tts.utils.text.tokenizer import TTSTokenizer  # noqa: E402
from TTS.utils.audio import AudioProcessor  # noqa: E402

from formatters import custom_formatter  # noqa: E402
from vocab_utils import build_characters_config  # noqa: E402

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def detect_sample_rate(metadata: str, data_path: str) -> int:
    """Sample rate shared by every clip. The VITS data loader reads each wav without
    checking its rate against the config (vits.py VitsDataset.__getitem__), so a clip at
    another rate would train on wrongly timed spectrograms; refuse to start instead."""
    rates = {}
    with open(metadata, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                path = line.split("|", 1)[0]
                info = sf.info(os.path.join(data_path, path))
                if info.channels != 1:
                    raise SystemExit(f"{path} has {info.channels} channels; convert the audio to mono first")
                rates.setdefault(info.samplerate, []).append(path)
    if len(rates) != 1:
        summary = ", ".join(f"{rate} Hz ×{len(paths)} (e.g. {paths[0]})" for rate, paths in rates.items())
        raise SystemExit(f"Clips have different sample rates: {summary}. Resample them with "
                         "prepare_speaker_dataset.py --wav-out so they share one rate.")
    return next(iter(rates))


def ipa_test_sentences(metadata: str, count: int = 2) -> list:
    """The first metadata lines as Coqui test sentences. The default test sentences are
    English text, which this IPA vocabulary cannot spell, so the logged test audio was noise."""
    with open(metadata, "r", encoding="utf-8") as f:
        return [[line.split("|", 1)[1].strip()] for line in f if "|" in line][:count]


def main():
    parser = argparse.ArgumentParser(description="Single-speaker VITS training")
    parser.add_argument("--data-path", default=os.path.join(project_root, "data"))
    parser.add_argument("--metadata", default=os.path.join("prepared", "speaker_01_metadata.txt"),
                        help="Metadata file relative to --data-path (prepare_speaker_dataset.py writes data/prepared/)")
    parser.add_argument("--run-name", default="voicelk_vits_speaker_01")
    parser.add_argument("--output-path", default=os.path.join(current_dir, "runs", "speaker_01"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-step", type=int, default=5000)
    parser.add_argument("--keep-checkpoints", type=int, default=2,
                        help="How many step checkpoints to keep (raise it to compare milestones)")
    parser.add_argument("--memorize", action="store_true",
                        help="Memorisation test: train on every clip, hold none out, keep the "
                             "checkpoint with the lowest training loss")
    parser.add_argument("--fp32", action="store_true", help="Train in full precision (no AMP)")
    parser.add_argument("--deterministic-durations", action="store_true",
                        help="Plain duration predictor (MSE on log durations) instead of the stochastic "
                             "one; learns from little data far faster and has no sampling noise")
    args, _ = parser.parse_known_args()

    metadata_path = os.path.join(args.data_path, args.metadata)
    sample_rate = detect_sample_rate(metadata_path, args.data_path)
    print(f"Sample rate: {sample_rate} Hz")

    dataset_config = BaseDatasetConfig(
        formatter="custom_formatter",
        dataset_name=args.run_name,
        meta_file_train=args.metadata,
        path=args.data_path,
    )

    characters_config = build_characters_config(metadata_path)

    audio_config = VitsAudioConfig(
        sample_rate=sample_rate, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name=args.run_name,
        batch_size=args.batch_size,
        eval_batch_size=8,
        batch_group_size=5,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        # fp16 on GPU; on CPU autocast would fall back to bfloat16, which the loggers cannot handle
        mixed_precision=torch.cuda.is_available() and not args.fp32,
        test_sentences=ipa_test_sentences(metadata_path),
        output_path=args.output_path,
        datasets=[dataset_config],
        characters=characters_config,
        cudnn_benchmark=True,
        # A full VITS checkpoint with optimizer state is close to 1 GB; Kaggle keeps 20 GB of
        # /kaggle/working, so only the newest two are kept alongside the best one.
        save_step=args.save_step,
        save_n_checkpoints=args.keep_checkpoints,
    )

    if args.deterministic_durations:
        config.model_args.use_sdp = False
    print(f"Duration predictor: {'stochastic' if config.model_args.use_sdp else 'deterministic'}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        clip_count = sum(1 for line in f if line.strip())
    if args.memorize:
        # Every clip is trained on, so any of them can be checked against its recording;
        # with no eval set the best checkpoint is chosen by training loss.
        config.run_eval = False
        print(f"Clips: {clip_count} (memorize mode: all trained on, none held out)")
    else:
        # The default 1% eval split rounds down to no clips on a small recording set (under
        # 100 clips), which Coqui rejects; hold out two clips instead. Values above 1 are a count.
        if clip_count * config.eval_split_size < 1:
            config.eval_split_size = 2
        print(f"Clips: {clip_count} ({config.eval_split_size} held out for evaluation)")
    print(f"Mixed precision: {config.mixed_precision}")

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=custom_formatter,
        eval_split=not args.memorize,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # Parses --continue_path / --restore_path and the other TrainerArgs flags.
    train_args = TrainerArgs()
    trainer_parser = train_args.init_argparse(arg_prefix="")
    cli_args, _ = trainer_parser.parse_known_args()
    train_args.parse_args(cli_args)

    trainer = Trainer(
        train_args,
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


# Guarded so DataLoader workers, which re-import this module on Windows (spawn), do not
# start a second training run.
if __name__ == "__main__":
    main()
