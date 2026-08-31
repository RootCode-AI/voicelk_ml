import os
import json
import re

import eng_to_ipa as ipa
from sinling import SinhalaTokenizer

# ----------------------------------------------------------------------
# Coqui TTS-style phoneme formatting
# ----------------------------------------------------------------------
#
# Coqui TTS's ESpeak phonemizer (TTS.tts.utils.text.phonemizers.ESpeak)
# emits one phoneme string per word with individual phonemes joined by a
# separator (``|`` by default — see ``BasePhonemizer.phonemize``), while
# words themselves stay separated by plain whitespace and punctuation is
# preserved verbatim (``keep_puncs=True``). e.g. espeak-ng's underscore
# output ``"p_ɹ_ˈaɪ_ɚ t_ə n_oʊ_v_ˈɛ_m_b_ɚ"`` is re-joined by Coqui as
# ``"p|ɹ|ˈaɪ|ɚ t|ə n|oʊ|v|ˈɛ|m|b|ɚ"``.
#
# We mirror that convention here: every IPA string this module produces —
# from the lexicon, from ``eng_to_ipa``, or from the rule-based Sinhala
# decomposer — is segmented into individual phoneme units and re-joined
# with ``PHONEME_SEP``, while word boundaries stay plain spaces.

PHONEME_SEP = "|"

# Multi-character IPA units that must stay a single phoneme token,
# ordered longest-first so the regex below matches them greedily before
# falling back to single characters.
_MULTI_CHAR_PHONEMES = [
    # dental stops (Sinhala) with optional aspiration
    "t̪h", "d̪h", "t̪", "d̪",
    # post-alveolar affricates (Sinhala/English) with optional aspiration
    "tʃh", "dʒh", "tʃ", "dʒ",
    # aspirated stops (Sinhala)
    "kh", "gh", "ph", "bh", "ʈh", "ɖh",
    # English diphthongs
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "eə", "ɪə", "ʊə",
    # Sinhala diphthongs
    "ai", "au",
    # long vowels
    "iː", "aː", "uː", "eː", "oː", "ɜː", "ɔː", "æː", "ɑː",
]

# Stress marks (ˈ primary / ˌ secondary) and length/aspiration diacritics
# that were not absorbed into a cluster above are kept as their own
# phoneme token — reproducing exact espeak-internal stress/nucleus fusion
# would require full syllabification, which dictionary-derived IPA (no
# syllable boundaries) can't support reliably, so each mark stands alone.
_PHONEME_TOKEN_RE = re.compile(
    "(?P<multi>" + "|".join(_MULTI_CHAR_PHONEMES) + ")"
    "|(?P<single>.)",
)

# Splits a whitespace-delimited chunk into (leading punctuation, IPA core,
# trailing punctuation), e.g. "(hello!" -> ("(", "hello", "!").
_PUNCT_STRIP_RE = re.compile(r"^([^\wɐ-˿]*)(.*?)([^\wɐ-˿]*)$")


def _segment_ipa_word(word: str) -> list:
    """Splits a single (punctuation-free) IPA word into phoneme tokens."""
    tokens = []
    for match in _PHONEME_TOKEN_RE.finditer(word):
        tokens.append(match.group())
    return tokens


def format_coqui_ipa(raw_ipa: str) -> str:
    """
    Reformats a raw IPA transcription into Coqui TTS's phoneme convention:
    phonemes within a word joined by ``|``, words separated by a plain
    space, punctuation preserved as-is.

    Idempotent — a string that has already been formatted (or a lexicon
    entry authored directly in this format) is returned unchanged.
    """
    if not raw_ipa:
        return raw_ipa
    if PHONEME_SEP in raw_ipa:
        return raw_ipa

    chunks = re.split(r"(\s+)", raw_ipa)
    formatted = []
    for chunk in chunks:
        if not chunk or chunk.isspace():
            formatted.append(chunk)
            continue

        # Coqui's Punctuation.restore() re-inserts punctuation directly
        # around the phonemized word, with no separator of its own — so
        # peel off leading/trailing punctuation before segmenting and
        # reattach it unpiped.
        lead, core, trail = _PUNCT_STRIP_RE.match(chunk).groups()
        formatted.append(lead)
        # Sub-word separators such as the "." in "dʌbəl.juː" (double-you)
        # become plain word breaks, same as everywhere else.
        for i, sub in enumerate(core.split(".")):
            if i > 0:
                formatted.append(" ")
            formatted.append(PHONEME_SEP.join(_segment_ipa_word(sub)))
        formatted.append(trail)
    return "".join(formatted)


class CodeSwitchedG2P:
    """
    Stage 2 of the VoiceLK NLP Pipeline — Code-Switched Grapheme-to-Phoneme (G2P).

    Detects language boundaries in mixed Sinhala-English text (produced by
    Stage 1 normalisation) and routes each segment to the appropriate phoneme
    conversion module:

      - English tokens  → ``eng_to_ipa`` library, with lexicon overrides and
                          smart acronym spelling (e.g. RAM → ɑːr eɪ em).
      - Sinhala tokens  → rule-based consonant/vowel decomposition, with
                          ``lexicon.json`` override for ICT domain terms.

    The two streams are merged into a single, unified IPA string that is
    passed directly to the VITS acoustic model.
    """

    def __init__(self):
        self.tokenizer = SinhalaTokenizer()
        self.en_g2p = ipa

        lexicon_path = os.path.join(os.path.dirname(__file__), "lexicon.json")
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                lexicon_data = json.load(f)
                self.en_lexicon = lexicon_data.get("en_lexicon", {})
                self.si_lexicon = lexicon_data.get("si_lexicon", {})
        except Exception:
            self.en_lexicon = {}
            self.si_lexicon = {}

        # Acronyms marked read_as='word' in the lexicon are pronounced as words
        self.word_acronyms = {
            key
            for key, entry in self.en_lexicon.items()
            if entry.get("read_as") == "word" and key.isalpha() and len(key) <= 5
        }

        # IPA for each English letter name (used when spelling acronyms),
        # pre-formatted in Coqui TTS's phoneme convention.
        self.letter_ipa = {
            letter: format_coqui_ipa(raw)
            for letter, raw in {
                "a": "eɪ",  "b": "biː", "c": "siː",  "d": "diː", "e": "iː",
                "f": "ef",  "g": "dʒiː","h": "eɪtʃ", "i": "aɪ",  "j": "dʒeɪ",
                "k": "keɪ", "l": "el",  "m": "em",    "n": "en",  "o": "oʊ",
                "p": "piː", "q": "kjuː","r": "ɑːr",   "s": "es",  "t": "tiː",
                "u": "juː", "v": "viː", "w": "dʌbəl.juː", "x": "eks",
                "y": "waɪ", "z": "ziː",
            }.items()
        }

        # Sinhala consonant → IPA base phoneme
        self.si_consonants = {
            "ක": "k",  "ඛ": "kh", "ග": "g",  "ඝ": "gh", "ඞ": "ng",
            "ච": "tʃ", "ඡ": "tʃh","ජ": "dʒ", "ඣ": "dʒh","ඤ": "ɲ",
            "ට": "ʈ",  "ඨ": "ʈh", "ඩ": "ɖ",  "ඪ": "ɖh", "ණ": "ɳ",
            "ත": "t̪",  "ථ": "t̪h", "ද": "d̪",  "ධ": "d̪h", "න": "n",
            "ප": "p",  "ඵ": "ph", "බ": "b",  "භ": "bh", "ම": "m",
            "ය": "j",  "ර": "r",  "ල": "l",  "ව": "ʋ",
            "ශ": "ʃ",  "ෂ": "ʂ",  "ස": "s",  "හ": "h",  "ළ": "ɭ",  "ෆ": "f",
        }

        # Sinhala independent vowel → IPA
        self.si_vowels = {
            "අ": "a",  "ආ": "aː", "ඇ": "æ",  "ඈ": "æː",
            "ඉ": "i",  "ඊ": "iː", "උ": "u",  "ඌ": "uː",
            "එ": "e",  "ඒ": "eː", "ඔ": "o",  "ඕ": "oː",
        }

        # Sinhala vowel diacritics (attached to a consonant base)
        self.si_modifiers = {
            "්": "",   "ා": "aː", "ැ": "æ",  "ෑ": "æː",
            "ි": "i",  "ී": "iː", "ු": "u",  "ූ": "uː",
            "ෙ": "e",  "ේ": "eː", "ො": "o",  "ෝ": "oː",
            "ෞ": "au", "ෛ": "ai",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spell_as_letters(self, text: str) -> str:
        """Spells an uppercase acronym letter-by-letter (e.g. MB → em biː)."""
        return " ".join(
            self.letter_ipa[ch] for ch in text.lower() if ch in self.letter_ipa
        )

    def _process_english(self, text: str) -> str:
        """
        Converts an English token to IPA.

        Priority order:
          1. Domain lexicon override (lexicon.json → en_lexicon)
          2. Single letter → letter-name IPA
          3. ALL-CAPS word acronym → pronounce as word via eng_to_ipa
          4. ALL-CAPS other acronym → spell letter-by-letter
          5. General case → eng_to_ipa library
        """
        word_lower = text.lower()

        if word_lower in self.en_lexicon:
            return format_coqui_ipa(self.en_lexicon[word_lower].get("ipa", ""))

        if len(text) == 1 and word_lower in self.letter_ipa:
            return self.letter_ipa[word_lower]

        if text.isupper() and text.isalpha() and len(text) >= 2:
            if word_lower in self.word_acronyms:
                return format_coqui_ipa(self.en_g2p.convert(text).replace("*", ""))
            return self._spell_as_letters(text)

        return format_coqui_ipa(self.en_g2p.convert(text).replace("*", ""))

    def _process_sinhala_word(self, word: str) -> str:
        """
        Converts a Sinhala token to IPA.

        Uses the ``si_lexicon`` for ICT domain overrides; falls back to
        rule-based consonant-modifier decomposition.
        """
        if word in self.si_lexicon:
            return format_coqui_ipa(self.si_lexicon[word].get("ipa", ""))

        # Each entry is one phoneme token (Coqui/ESpeak convention), not a
        # concatenated syllable, so a consonant and its vowel diacritic are
        # appended as two separate tokens.
        phonemes = []
        i, length = 0, len(word)

        while i < length:
            char = word[i]
            if char in self.si_consonants:
                phonemes.append(self.si_consonants[char])
                if i + 1 < length and word[i + 1] in self.si_modifiers:
                    vowel = self.si_modifiers[word[i + 1]]
                    if vowel:
                        phonemes.append(vowel)
                    i += 1
                else:
                    phonemes.append("a")
            elif char in self.si_vowels:
                phonemes.append(self.si_vowels[char])
            i += 1

        return PHONEME_SEP.join(phonemes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_unified_ipa(self, text: str) -> str:
        """
        Tokenises *text* with the Sinling tokeniser and routes each token to
        the appropriate G2P module.

        Returns a single space-joined IPA string covering all tokens,
        preserving punctuation in its original position.
        """
        tokens = self.tokenizer.tokenize(text)
        unified_ipa = []

        for token in tokens:
            if re.match(r"^[a-zA-Z]+$", token):
                unified_ipa.append(self._process_english(token))
            elif re.match(r"^[\u0D80-\u0DFF\u200D]+$", token):
                unified_ipa.append(self._process_sinhala_word(token))
            else:
                # Retain punctuation and whitespace tokens unchanged
                unified_ipa.append(token)

        return " ".join(unified_ipa).strip()