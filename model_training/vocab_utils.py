"""
Builds a Coqui `CharactersConfig` from a metadata.txt file's IPA column together with
every symbol the text front-end can produce, so each train_*.py script is
self-contained and doesn't depend on a separately generated ipa_vocab.json being in
sync with the dataset it's about to train on.

Taking the front-end's full inventory, not just what this dataset happens to contain,
means an unseen word at inference never carries a symbol the model has no embedding
for — Coqui would otherwise drop that symbol silently and the word comes out garbled.
"""

import os
import sys

from TTS.tts.configs.shared_configs import CharactersConfig

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "model_engine"))

from g2p import CodeSwitchedG2P  # noqa: E402

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

    inventory = CodeSwitchedG2P().phoneme_inventory()
    unheard = sorted(inventory - chars)
    if unheard:
        print(
            "Vocabulary: the front-end can produce these symbols but the training data never "
            f"contains them, so their embeddings stay untrained: {' '.join(unheard)}"
        )
    unknown = sorted(c for c in chars - inventory if c not in KNOWN_PUNCTUATION)
    if unknown:
        print(
            "Vocabulary: the metadata contains symbols the current front-end no longer produces "
            f"(regenerate it with data_preparation/*.py): {' '.join(unknown)}"
        )
    # Punctuation too: the front-end passes it through whatever this dataset happens to
    # contain, and a mark missing from the vocabulary is silently dropped at inference
    chars |= inventory | KNOWN_PUNCTUATION

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
