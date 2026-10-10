"""
Turns a recorded-speaker transcript CSV (e.g. data/prepared/speaker_01.csv) into the
VITS metadata file the training scripts read, and optionally a training-ready copy of
the audio.

CSV columns: file_name plus one transcript column (detected automatically, or --column).
Output line format, the same as data_preparation/prepare_dataset.py:
    wavs/{file_name}.wav|{ipa_sequence}

Instead of --csv, --metadata takes an already converted metadata file (path|ipa_sequence,
e.g. uploaded to Kaggle next to the audio): its IPA is kept as it is and only the audio
is checked and converted, with each clip looked up in --wav-dir by its file name.

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
    python data_preparation/prepare_speaker_dataset.py         --metadata /kaggle/input/.../speaker_01_metadata.txt --wav-dir /kaggle/input/.../audio         --wav-out /kaggle/working/data/wavs --out /kaggle/working/data/speaker_01_metadata.txt
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


def read_csv_rows(csv_path: str, column: str):
    """Yields (wav name, IPA) for each transcript row, converting the text with the front-end."""
    print("Loading NLP Pipeline...")
    pipeline = TextProcessingPipeline()
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        column = column or detect_column(reader.fieldnames)
        print(f"Transcript column: {column}")
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            transcript = (row.get(column) or "").strip()
            if file_name and transcript:
                yield wav_name(file_name), lambda text=transcript: pipeline.process(text)["ipa_sequence"]


def read_metadata_rows(metadata_path: str):
    """Yields (wav name, IPA) for each line of an already converted metadata file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("|", 1)
            if len(cols) == 2 and cols[1].strip():
                yield os.path.basename(cols[0]), lambda ipa=cols[1].strip(): ipa


def prepare(args):
    if args.wav_out and not args.wav_dir:
        raise SystemExit("--wav-out needs --wav-dir to read the recordings from")

    source = args.csv or args.metadata
    stem = os.path.splitext(os.path.basename(source))[0].replace("_metadata", "")
    out_path = args.out or os.path.join(project_root, "data", "prepared", f"{stem}_metadata.txt")
    if os.path.abspath(out_path) == os.path.abspath(source):
        raise SystemExit("--out must differ from the input file")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if args.wav_out:
        os.makedirs(args.wav_out, exist_ok=True)

    rows = read_csv_rows(args.csv, args.column) if args.csv else read_metadata_rows(args.metadata)

    written, missing, errors = 0, [], 0
    total_seconds = 0.0

    with open(out_path, "w", encoding="utf-8") as outfile:
        for name, to_ipa in rows:
            if args.wav_dir:
                src = os.path.join(args.wav_dir, name)
                if not os.path.exists(src):
                    missing.append(name)
                    continue

            try:
                ipa_seq = to_ipa()
                if args.wav_out:
                    total_seconds += convert_wav(src, os.path.join(args.wav_out, name), args.sample_rate)
            except Exception as e:
                print(f"Error processing row '{name}': {e}")
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
        print(f"Audio written to {args.wav_out} at {args.sample_rate} Hz mono: {total_seconds / 60:.1f} min")
    print(f"VITS metadata file saved at: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Transcript CSV with a file_name column; converted to IPA here")
    source.add_argument("--metadata", help="Already converted metadata (path|ipa); IPA kept as it is")
    parser.add_argument("--column", help="Transcript column; detected when the CSV has only one")
    parser.add_argument("--out", help="Metadata file to write (default: data/prepared/<input name>_metadata.txt)")
    parser.add_argument("--wav-dir", help="Folder holding the recordings; rows without one are skipped")
    parser.add_argument("--wav-out", help="Write mono PCM copies of the kept recordings here")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate for --wav-out")
    prepare(parser.parse_args())
