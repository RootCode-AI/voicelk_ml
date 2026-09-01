import os
import sys
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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    wavs_dir = os.path.join(data_dir, "wavs")

    # Validate all 3 datasets that share the same data/wavs/ folder.
    # Adjust the wavs_dir argument per dataset if you end up storing audio in
    # separate subfolders (e.g. data/custom/wavs/, data/openslr/wavs/) instead.
    validate_metadata(os.path.join(data_dir, "metadata.txt"), wavs_dir, expect_speaker_id=False)
    validate_metadata(os.path.join(data_dir, "openslr_metadata.txt"), wavs_dir, expect_speaker_id=True)
    validate_metadata(os.path.join(data_dir, "pathnirwana_metadata.txt"), wavs_dir, expect_speaker_id=True)
