import argparse
import os
import re
import sys
import wave
from collections import Counter


def validate_metadata(metadata_path, wavs_dir, expect_speaker_id=False, show_examples=10):
    """
    Validates a prepared metadata.txt file before it's used for VITS training.

    Checks:
      - Correct column count (2 for single-speaker, 3 for multi-speaker with speaker_id)
      - No empty audio path / IPA sequence fields
      - Referenced audio files actually exist in wavs_dir
      - No duplicate audio file references
      - Basic stats: total lines, unique speakers (if applicable), IPA token-length distribution
    """
    expected_cols = 3 if expect_speaker_id else 2

    total_lines = 0
    valid_lines = 0
    malformed_lines = []
    empty_field_lines = []
    missing_audio = []
    duplicate_audio = []
    seen_audio_paths = set()
    speaker_counter = Counter()
    ipa_token_counts = []

    print(f"\n=== Validating: {metadata_path} ===")

    if not os.path.exists(metadata_path):
        print(f"  NOT FOUND — skipping ({metadata_path})")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            total_lines += 1

            parts = line.split("|")
            if len(parts) != expected_cols:
                malformed_lines.append((line_num, len(parts)))
                continue

            audio_path = parts[0].strip()
            ipa_seq = parts[1].strip()
            speaker_id = parts[2].strip() if expect_speaker_id else None

            if not audio_path or not ipa_seq or (expect_speaker_id and not speaker_id):
                empty_field_lines.append(line_num)
                continue

            if audio_path in seen_audio_paths:
                duplicate_audio.append((line_num, audio_path))
            seen_audio_paths.add(audio_path)

            # audio_path is stored as "wavs/filename.wav" -> resolve against wavs_dir
            filename = os.path.basename(audio_path)
            if not os.path.exists(os.path.join(wavs_dir, filename)):
                missing_audio.append((line_num, filename))

            if expect_speaker_id:
                speaker_counter[speaker_id] += 1

            ipa_token_counts.append(len(ipa_seq.split()))
            valid_lines += 1

    # ---- Report ----
    print(f"  Total non-empty lines:   {total_lines}")
    print(f"  Valid lines:             {valid_lines}")
    print(f"  Malformed (wrong cols):  {len(malformed_lines)}")
    print(f"  Empty-field lines:       {len(empty_field_lines)}")
    print(f"  Missing audio files:     {len(missing_audio)}")
    print(f"  Duplicate audio refs:    {len(duplicate_audio)}")

    if ipa_token_counts:
        avg_len = sum(ipa_token_counts) / len(ipa_token_counts)
        print(f"  IPA tokens min/avg/max:  {min(ipa_token_counts)} / {avg_len:.1f} / {max(ipa_token_counts)}")

    if expect_speaker_id and speaker_counter:
        print(f"  Unique speakers:         {len(speaker_counter)}")
        least_common = speaker_counter.most_common()[-5:]
        print(f"  Speakers with fewest utterances (speaker, count): {least_common}")

    if malformed_lines:
        print(f"\n  First malformed lines (line_num, column_count): {malformed_lines[:show_examples]}")
    if empty_field_lines:
        print(f"  First empty-field lines: {empty_field_lines[:show_examples]}")
    if missing_audio:
        print(f"  First missing audio files (line_num, filename): {missing_audio[:show_examples]}")
    if duplicate_audio:
        print(f"  First duplicate audio refs (line_num, path): {duplicate_audio[:show_examples]}")

    has_issues = bool(malformed_lines or empty_field_lines or missing_audio)
    print("\n  RESULT: " + ("issues found above — fix before training." if has_issues
                             else "looks clean, ready for training."))


def audio_stats(wavs_dir, label, speaker_pattern=None, show_examples=5):
    """
    Reports total audio length, sample-rate distribution, and (if speaker_pattern
    matches filenames) per-speaker utterance/duration counts for a wavs directory.

    speaker_pattern: an optional regex with one capture group that extracts the
    speaker id from a filename (e.g. r"^(sin_\\d+)_" for "sin_01_00001.wav").
    """
    print(f"\n=== Audio stats: {label} ({wavs_dir}) ===")

    if not os.path.isdir(wavs_dir):
        print(f"  NOT FOUND — skipping ({wavs_dir})")
        return

    filenames = sorted(f for f in os.listdir(wavs_dir) if f.lower().endswith(".wav"))
    if not filenames:
        print("  No .wav files found.")
        return

    total_duration = 0.0
    durations = []
    sample_rates = Counter()
    speaker_counter = Counter()
    speaker_duration = Counter()
    unreadable = []

    for filename in filenames:
        path = os.path.join(wavs_dir, filename)
        try:
            with wave.open(path, "rb") as wf:
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(framerate) if framerate else 0.0
        except Exception as exc:
            unreadable.append((filename, str(exc)))
            continue

        total_duration += duration
        durations.append(duration)
        sample_rates[framerate] += 1

        if speaker_pattern:
            match = re.match(speaker_pattern, filename)
            speaker_id = match.group(1) if match else "UNMATCHED"
            speaker_counter[speaker_id] += 1
            speaker_duration[speaker_id] += duration

    readable_count = len(durations)
    print(f"  Total files:             {len(filenames)}")
    print(f"  Readable files:          {readable_count}")
    if unreadable:
        print(f"  Unreadable files:        {len(unreadable)} (first {show_examples}): {unreadable[:show_examples]}")

    if durations:
        hours = total_duration / 3600.0
        print(f"  Total audio length:      {total_duration:.1f}s  (~{hours:.2f} hours)")
        print(f"  Duration min/avg/max:    {min(durations):.2f}s / {sum(durations)/len(durations):.2f}s / {max(durations):.2f}s")

    if sample_rates:
        print(f"  Sample rate distribution (Hz: file count): {dict(sample_rates.most_common())}")

    if speaker_pattern:
        print(f"  Unique speakers matched: {len(speaker_counter)}")
        for speaker_id, count in speaker_counter.most_common():
            hrs = speaker_duration[speaker_id] / 3600.0
            print(f"    {speaker_id}: {count} files, {speaker_duration[speaker_id]:.1f}s (~{hrs:.2f}h)")
    else:
        print("  Speaker breakdown:       not requested (no speaker_pattern given)")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    wavs_dir = os.path.join(data_dir, "wavs")

    parser = argparse.ArgumentParser(description="Validate metadata files and report audio stats for VoiceLK datasets.")
    parser.add_argument("--pathnirwana-wavs", default=wavs_dir,
                         help="Directory containing Pathnirwana .wav files (defaults to data/wavs).")
    parser.add_argument("--custom-wavs", default=wavs_dir,
                         help="Directory containing custom-dataset .wav files (defaults to data/wavs).")
    parser.add_argument("--skip-audio-stats", action="store_true",
                         help="Skip the duration/sample-rate/speaker scan (metadata checks only).")
    args = parser.parse_args()

    # Validate all 3 datasets that share the same data/wavs/ folder.
    # Adjust the wavs_dir argument per dataset if you end up storing audio in
    # separate subfolders (e.g. data/custom/wavs/, data/openslr/wavs/) instead.
    validate_metadata(os.path.join(data_dir, "metadata.txt"), wavs_dir, expect_speaker_id=False)
    validate_metadata(os.path.join(data_dir, "openslr_metadata.txt"), wavs_dir, expect_speaker_id=True)
    validate_metadata(os.path.join(data_dir, "pathnirwana_metadata.txt"), wavs_dir, expect_speaker_id=True)

    if not args.skip_audio_stats:
        # Pathnirwana filenames follow sin_<speaker>_<utterance>.wav — speaker id is a real field here.
        audio_stats(args.pathnirwana_wavs, "Pathnirwana", speaker_pattern=r"^(sin_\d+)_")
        # Custom dataset filenames (Download (N).wav) carry no speaker id, so no speaker_pattern is passed.
        audio_stats(args.custom_wavs, "Custom (code-switch)")
