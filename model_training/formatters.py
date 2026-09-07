"""
Coqui-TTS-compatible formatter functions for VoiceLK's 3 datasets.

Each formatter reads one of the pipe-delimited metadata.txt files produced by
data_preparation/*.py and returns a list of sample dicts in the format Coqui's
`load_tts_samples()` expects: {text, audio_file, speaker_name, root_path}.

`text` here is already the final IPA phoneme sequence (space-separated words,
character-level phonemes with no internal separator) — NOT raw Sinhala text.
The VITS config must be set with use_phonemes=False so this string is consumed
directly by the character tokenizer instead of being re-phonemized.
"""

import os


def custom_formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """VoiceLK custom (YouTube-scraped) dataset — single speaker.

    Line format: wavs/{file_name}|{ipa_sequence}

    Uses split("|", 1) rather than a plain split("|") because the ipa_sequence
    itself may legitimately contain "|" characters (e.g. from an older G2P
    version's phoneme separator) — a plain split would silently truncate the
    text to whatever precedes the first embedded "|".
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "voicelk_custom"
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("|", 1)
            if len(cols) < 2:
                continue
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def openslr_formatter(root_path, meta_file, ignored_speakers=None, **kwargs):  # pylint: disable=unused-argument
    """OpenSLR Sinhala corpus — multi-speaker.

    Line format: wavs/{file_id}|{ipa_sequence}|{user_id}

    The file path and speaker id are split off from the outside in (first "|"
    for the path, last "|" for the speaker id), so any "|" characters embedded
    in the ipa_sequence itself (see custom_formatter's note) stay intact
    instead of being mistaken for column separators.
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wav_col, _, rest = line.partition("|")
            text, sep, speaker_name = rest.rpartition("|")
            if not sep:
                continue
            wav_file = os.path.join(root_path, wav_col)
            if ignored_speakers and speaker_name in ignored_speakers:
                continue
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def pathnirwana_formatter(root_path, meta_file, ignored_speakers=None, **kwargs):  # pylint: disable=unused-argument
    """Pathnirwana dataset — multi-speaker.

    Line format: wavs/{file_name}|{ipa_sequence}|{speaker_id}

    See openslr_formatter's note: split from the outside in so embedded "|"
    characters in the ipa_sequence don't get mistaken for column separators.
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wav_col, _, rest = line.partition("|")
            text, sep, speaker_name = rest.rpartition("|")
            if not sep:
                continue
            wav_file = os.path.join(root_path, wav_col)
            if ignored_speakers and speaker_name in ignored_speakers:
                continue
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items
