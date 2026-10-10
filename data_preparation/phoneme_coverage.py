"""
Phoneme coverage report for a transcript CSV, run before training.

The model only ever sees the IPA characters the front-end produces, so an unseen
word is safe as long as its phonemes, and the transitions between them, were heard
often enough in training. This report runs every transcript through the same
pipeline as dataset preparation and lists:

  - every character the model's vocabulary would be built from, with counts
  - phonemes (a base symbol with its length / dental / aspiration marks) that are rare
  - symbols the front-end can produce that never occur in the data — the vocabulary
    includes them, but the model never hears them, so they stay untrained
  - Sinhala letters that never or rarely occur in the text
  - phoneme pairs inside words, how many distinct ones the data covers
  - example rows for each rare phoneme, so you know what to record more of

Usage:
    python data_preparation/phoneme_coverage.py --csv data/prepared/speaker_01.csv
    python data_preparation/phoneme_coverage.py --csv ... --min-count 30 --out report.md
"""

import argparse
import collections
import csv
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "model_engine"))

from pipeline import TextProcessingPipeline

# Marks that belong to the preceding symbol rather than standing alone
PHONEME_MARKS = "ː̪ʰ"
STRESS_MARKS = "ˈˌ"
PUNCTUATION = set(".,?!'\"")
PHONEME_PATTERN = re.compile(f"[^{PHONEME_MARKS}{STRESS_MARKS}\\s][{PHONEME_MARKS}]*")


def split_phonemes(word: str) -> list:
    """Splits one IPA word into phonemes (base symbol plus its marks), dropping stress."""
    return [p for p in PHONEME_PATTERN.findall(word) if p not in PUNCTUATION]


def detect_column(fieldnames: list) -> str:
    candidates = [name for name in fieldnames if name != "file_name"]
    if len(candidates) != 1:
        raise SystemExit(f"Pass --column; the CSV has several text columns: {candidates}")
    return candidates[0]


def build_report(csv_path: str, column: str, min_count: int, examples: int) -> str:
    pipeline = TextProcessingPipeline()
    g2p = pipeline.g2p_engine

    chars = collections.Counter()
    phonemes = collections.Counter()
    pairs = collections.Counter()
    si_letters = collections.Counter()
    rows_with = collections.defaultdict(list)
    row_count = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column = column or detect_column(reader.fieldnames)
        for row in reader:
            text = (row.get(column) or "").strip()
            if not text:
                continue
            row_count += 1
            for ch in text:
                if "඀" <= ch <= "෿":
                    si_letters[ch] += 1

            ipa = pipeline.process(text)["ipa_sequence"]
            chars.update(ch for ch in ipa if not ch.isspace())
            seen_here = set()
            for word in ipa.split():
                units = split_phonemes(word)
                phonemes.update(units)
                pairs.update(zip(units, units[1:]))
                seen_here.update(units)
            for unit in seen_here:
                if len(rows_with[unit]) < examples:
                    rows_with[unit].append((row.get("file_name", "?"), text))

    lines = [f"# Phoneme coverage — {os.path.basename(csv_path)}", ""]
    lines.append(f"Rows: {row_count}  ·  phoneme tokens: {sum(phonemes.values())}  ·  "
                 f"distinct phonemes: {len(phonemes)}  ·  distinct pairs: {len(pairs)}")
    lines.append(f"Rare means fewer than {min_count} occurrences.")
    lines.append("")

    missing = sorted(g2p.phoneme_inventory() - set(chars))
    lines.append("## Symbols the front-end can produce that never occur in this data")
    lines.append("")
    lines.append("The vocabulary still includes them, but the model never hears them in training, "
                 "so their embeddings stay untrained — record sentences that contain them.")
    lines.append("")
    lines.append(" ".join(f"`{s}` (U+{ord(s):04X})" for s in missing) if missing else "None.")
    lines.append("")

    lines.append("## Rare phonemes")
    lines.append("")
    rare = [(p, c) for p, c in phonemes.items() if c < min_count]
    if rare:
        lines.append("| Phoneme | Count | Example rows |")
        lines.append("| --- | ---: | --- |")
        for unit, count in sorted(rare, key=lambda x: x[1]):
            sample = "; ".join(name for name, _ in rows_with[unit])
            lines.append(f"| `{unit}` | {count} | {sample} |")
    else:
        lines.append("None.")
    lines.append("")

    inventory = set(g2p.si_consonants) | set(g2p.si_vowels)
    lines.append("## Sinhala letters")
    lines.append("")
    absent = sorted(ch for ch in inventory if si_letters[ch] == 0)
    scarce = sorted((ch for ch in inventory if 0 < si_letters[ch] < min_count), key=lambda ch: si_letters[ch])
    lines.append("Never in the text: " + (" ".join(f"{ch} ({g2p.si_consonants.get(ch) or g2p.si_vowels.get(ch)})"
                                                  for ch in absent) or "none"))
    lines.append("")
    lines.append("Rare: " + (", ".join(f"{ch} ×{si_letters[ch]}" for ch in scarce) or "none"))
    lines.append("")

    lines.append("## All phonemes by frequency")
    lines.append("")
    lines.append(" · ".join(f"`{p}` {c}" for p, c in phonemes.most_common()))
    lines.append("")

    lines.append("## Model vocabulary (characters)")
    lines.append("")
    lines.append(" · ".join(f"`{ch}` {c}" for ch, c in sorted(chars.items(), key=lambda x: -x[1])))
    lines.append("")

    lines.append("## Rarest phoneme pairs inside words")
    lines.append("")
    singletons = sum(1 for c in pairs.values() if c == 1)
    lines.append(f"{singletons} of {len(pairs)} distinct pairs occur only once.")
    lines.append("")
    lines.append(" · ".join(f"`{a}{b}` {c}" for (a, b), c in sorted(pairs.items(), key=lambda x: x[1])[:60]))
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phoneme coverage report for a transcript CSV.")
    parser.add_argument("--csv", required=True, help="Transcript CSV with a file_name column")
    parser.add_argument("--column", help="Transcript column (detected when there is only one)")
    parser.add_argument("--min-count", type=int, default=20, help="Below this a phoneme counts as rare")
    parser.add_argument("--examples", type=int, default=3, help="Example rows listed per rare phoneme")
    parser.add_argument("--out", help="Write the report here instead of printing it")
    args = parser.parse_args()

    report = build_report(args.csv, args.column, args.min_count, args.examples)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
