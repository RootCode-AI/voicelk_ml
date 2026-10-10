"""
Turns a recorded-speaker transcript CSV (e.g. data/prepared/speaker_01.csv) into the
VITS metadata file the training scripts read, and optionally a training-ready copy of
the audio.

CSV columns: file_name plus one transcript column (detected automatically, or --column).
Output line format, the same as data_preparation/prepare_dataset.py:
    wavs/{file_name}.wav|{ipa_sequence}

With --wav-dir, rows whose recording is missing are skipped (and listed), so a
partially recorded script still trains on what exists. With --wav-out, every kept
recording is written there as mono 16-bit PCM at --sample-rate, so the training config
can rely on a single, known sample rate whatever the recorder produced.

Usage:
    python data_preparation/prepare_speaker_dataset.py --csv data/prepared/speaker_01.csv
    python data_preparation/prepare_speaker_dataset.py --csv data/prepared/speaker_01.csv \
        --wav-dir /kaggle/input/voicelk-speaker-01/wavs \
        --wav-out /kaggle/working/data/wavs --sample-rate 22050 \
        --out /kaggle/working/data/speaker_01_metadata.txt
"""

import argparse
import csv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "model_engine"))

from pipeline import TextProcessingPipeline  # noqa: E402


def detect_column(fieldnames: list) -> str:
    candidates = [name for name in fieldnames if name != "file_name"]
    if len(candidates) != 1:
        raise SystemExit(f"Pass --column; the CSV has several text columns: {candidates}")
    return candidates[0]


def wav_name(file_name: str) -> str:
    return file_name if file_name.lower().endswith(".wav") else f"{file_name}.wav"


def convert_wav(src: str, dst: str, sample_rate: int) -> float:
    """Writes src to dst as mono 16-bit PCM at sample_rate; returns the duration in seconds."""
    import librosa
    import soundfile as sf

    audio, _ = librosa.load(src, sr=sample_rate, mono=True)
    sf.write(dst, audio, sample_rate, subtype="PCM_16")
    return len(audio) / sample_rate


def prepare(args):
    if args.wav_out and not args.wav_dir:
        raise SystemExit("--wav-out needs --wav-dir to read the recordings from")

    stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = args.out or os.path.join(project_root, "data", "prepared", f"{stem}_metadata.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if args.wav_out:
        os.makedirs(args.wav_out, exist_ok=True)

    print("Loading NLP Pipeline...")
    pipeline = TextProcessingPipeline()

    written, missing, errors = 0, [], 0
    total_seconds = 0.0

    with open(args.csv, "r", encoding="utf-8") as csvfile, open(out_path, "w", encoding="utf-8") as outfile:
        reader = csv.DictReader(csvfile)
        column = args.column or detect_column(reader.fieldnames)
        print(f"Transcript column: {column}")

        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            transcript = (row.get(column) or "").strip()
            if not file_name or not transcript:
                continue

            name = wav_name(file_name)
            if args.wav_dir:
                src = os.path.join(args.wav_dir, name)
                if not os.path.exists(src):
                    missing.append(name)
                    continue

            try:
                ipa_seq = pipeline.process(transcript)["ipa_sequence"]
                if args.wav_out:
                    total_seconds += convert_wav(src, os.path.join(args.wav_out, name), args.sample_rate)
            except Exception as e:
                print(f"Error processing row '{file_name}': {e}")
                errors += 1
                continue

            outfile.write(f"wavs/{name}|{ipa_seq}\n")
            written += 1
            if written % 100 == 0:
                print(f"Processed {written} rows...")

    print("\n--- Data Preprocessing Complete ---")
    print(f"Rows written: {written}")
    print(f"Errors encountered: {errors}")
    if args.wav_dir:
        print(f"Rows skipped, recording missing: {len(missing)}")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if args.wav_out:
        print(f"Audio written to {args.wav_out} at {args.sample_rate} Hz mono: {total_seconds / 3600:.2f} h")
    print(f"VITS metadata file saved at: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Transcript CSV with a file_name column")
    parser.add_argument("--column", help="Transcript column; detected when the CSV has only one")
    parser.add_argument("--out", help="Metadata file to write (default: data/prepared/<csv name>_metadata.txt)")
    parser.add_argument("--wav-dir", help="Folder holding the recordings; rows without one are skipped")
    parser.add_argument("--wav-out", help="Write mono PCM copies of the kept recordings here")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate for --wav-out")
    prepare(parser.parse_args())
