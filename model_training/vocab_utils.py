"""
Builds a Coqui `CharactersConfig` directly from a metadata.txt file's IPA column,
so each train_*.py script is self-contained and doesn't depend on a separately
generated ipa_vocab.json being in sync with the dataset it's about to train on.
"""

import os

from TTS.tts.configs.shared_configs import CharactersConfig

# Punctuation the normalizer/G2P pipeline is known to pass through unchanged
# (see model_engine/normalizer.py pad_punctuation / remove_noise).
KNOWN_PUNCTUATION = set(".,?!'\"")


def build_characters_config(metadata_path: str) -> CharactersConfig:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Could not find '{metadata_path}'. Run the matching data_preparation/*.py script first."
        )

    chars = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            chars.update(parts[1])

    chars.discard(" ")
    punctuations = "".join(sorted(c for c in chars if c in KNOWN_PUNCTUATION))
    phoneme_characters = "".join(sorted(c for c in chars if c not in KNOWN_PUNCTUATION))

    return CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=phoneme_characters,
        punctuations=punctuations + " ",
        phonemes=None,
    )
