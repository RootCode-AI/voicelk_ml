"""
Regression tests for the text front-end (model_engine/): each case is a reading that
was once wrong. A change that breaks one of them changes what the model is trained on
and what it hears at inference.

    python -m pytest tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_engine"))

from normalizer import SinhalaTextNormalizer  # noqa: E402
from pipeline import TextProcessingPipeline  # noqa: E402


@pytest.fixture(scope="module")
def normalizer():
    return SinhalaTextNormalizer()


@pytest.fixture(scope="module")
def pipeline():
    return TextProcessingPipeline()


@pytest.mark.parametrize("text, expected", [
    # "!" is NOT only as an operator; after a word it is an exclamation mark
    ("හරි!", "හරි !"),
    ("!!!", "! ! !"),
    ("if (!done)", "if නොවේ done"),
    ("a != b", "a අසමානයි b"),
    # Large numbers keep their zeros and scale words
    ("1000000000", "බිලියන"),
    ("1000000001", "එක් බිලියන එක"),
    ("1234567890123", "එක දෙක තුන හතර පහ හය හත අට නවය බිංදුව එක දෙක තුන"),
    ("1024", "එක්දහස් විසි හතර"),
    ("0", "බිංදුව"),
    # Digits against letters are read, not dropped
    ("4G", "හතර G"),
    ("MP3", "MP තුන"),
    ("5ක්", "පහක්"),
    ("10ට", "දහයට"),
    # Decimals, separators, dotted groups
    ("3.14", "තුන දශම එක හතර"),
    ("12.5%", "සියයට දොළහ දශම පහක්"),
    ("1,000", "දහස"),
    ("1,2,3", "එක , දෙක , තුන"),
    ("192.168.1.1", "එකසිය අනූ දෙක ඩොට් එකසිය හැට අට ඩොට් එක ඩොට් එක"),
    # Slashes by context
    ("2024/10/11", "දෙදහස් විසි හතර දහය එකොළහ"),
    ("km/h", "km h"),
    ("x/y", "x බෙදීම y"),
    ("10/2", "දහය බෙදීම දෙක"),
    # Hyphens: inside a word a space, otherwise minus
    ("compile-time", "compile time"),
    ("COVID-19", "COVID දහනවය"),
    ("i-1", "i සෘණ එක"),
    ("5-3", "පහ සෘණ තුන"),
    ("-5", "සෘණ පහ"),
    # URLs, accents, typographic apostrophe
    ("https://www.google.com", "ඩබ්ලිව් ඩබ්ලිව් ඩබ්ලිව් ඩොට් google ඩොට් කොම්"),
    ("Düsseldorf", "Dusseldorf"),
    ("don’t", "don't"),
])
def test_normalizer(normalizer, text, expected):
    assert normalizer.normalize(text) == expected


@pytest.mark.parametrize("text, expected", [
    # Schwa rules and the spoken (unaspirated) consonants
    ("කරනවා", "karənəʋaː"),
    ("පුළුවන්", "puɭuʋan"),
    ("කරන්න", "karannə"),
    ("ධර්මය", "d̪arməjə"),
    ("සිංහල", "siŋhələ"),
    ("කෘතිම", "krut̪imə"),
    # English: affricates as tʃ / dʒ, camelCase, contractions, possessives
    ("check", "tʃɛk"),
    ("setState", "sɛt steɪt"),
    ("don't", "doʊnt"),
    ("I'm", "aɪm"),
    ("flutter's", "ˈflətərz"),
    # A Sinhala suffix typed against an English word
    ("Microsoftලාගේ", "ˈmaɪˌkroʊˈsɔft laːgeː"),
])
def test_g2p(pipeline, text, expected):
    assert pipeline.process(text)["ipa_sequence"] == expected


def test_ipa_stays_inside_the_phoneme_inventory(pipeline):
    """Unknown words must not leak raw spelling (letters outside the IPA inventory)."""
    inventory = pipeline.g2p_engine.phoneme_inventory() | set(".,?!'\" ")
    for text in ["smartwatches xeno", "Microsoftලාගේ Gamesුත්", "openWeatherAPI fooBarBaz", "හරි! 4G 3.14"]:
        ipa = pipeline.process(text)["ipa_sequence"]
        assert set(ipa) <= inventory, f"{text!r} -> {ipa!r}"
