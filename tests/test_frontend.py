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
    # Digits against letters are read, not dropped — in English, the way the
    # speakers read a number written against Latin letters
    ("4G", "four G"),
    ("MP3", "MP three"),
    ("5ක්", "පහක්"),
    ("10ට", "දහයට"),
    # Language of a number: English next to English words, Sinhala with a
    # Sinhala ending or among Sinhala words (checked against the phonetic
    # transcripts of the recorded corpus)
    ("ඒකත් එක්කම 6 month", "ඒකත් එක්කම six month"),
    ("Microsoft 365 පිටිපස්සෙත්", "Microsoft three hundred sixty five පිටිපස්සෙත්"),
    ("Windows 7, ඊට", "Windows seven , ඊට"),
    ("go, ready, 3, 2, 1", "go , ready , three , two , one"),
    ("කීවා මේක 100% safe", "කීවා මේක hundred percent safe"),
    ("144Hz", "one hundred forty four hertz"),
    ("2.4 GHz", "two point four gigahertz"),
    ("හැබැයි 2026 බලනකොට", "හැබැයි දෙදහස් විසි හය බලනකොට"),
    ("So 2026 යේ", "So දෙදහස් විසි හය යේ"),
    ("inches 100ක්", "inches සියයක්"),
    ("10%ටත් වඩා", "සියයට දහයටත් වඩා"),
    # Units, ordinals, decades, resolutions, dimensions, times, ratios
    ("වැඩ කරන්නේ 25W TDP", "වැඩ කරන්නේ twenty five watt TDP"),
    ("4mm", "four millimeter"),
    ("2.4kg", "two point four KG"),
    ("60fps", "sixty FPS"),
    ("28th", "twenty eighth"),
    ("1950s", "nineteen fifties"),
    ("90s", "nineties"),
    ("1080p", "one zero eight zero p"),
    ("720p webcam", "seven two zero p webcam"),
    ("3 ms", "three millisecond"),
    ("RTX 4050", "RTX forty fifty"),
    ("Dimensity 6300", "Dimensity sixty three hundred"),
    ("Ryzen 5 8640HS", "Ryzen five eighty six forty HS"),
    ("Ryzen 8000 series", "Ryzen eight thousand series"),
    ("21x9.", "twenty one by nine ."),
    ("7:30 p.m.", "seven thirty p . m ."),
    ("16:9", "sixteen is to nine"),
    ("180° rotate", "one hundred eighty degrees rotate"),
    ("180° කැරකෙනවා", "අංශක එකසිය අසූව කැරකෙනවා"),
    ("$20 phone", "twenty dollars phone"),
    ("P1+", "P one plus"),
    ("2-in-1", "two in one"),
    ("64-bit", "sixty four bit"),
    ("24/7", "twenty four seven"),
    ("Android 4.2.1", "Android four point two point one"),
    # Sinhala thousands and lakhs, Western and lakh grouping
    ("30,000ක්", "තිස්දහසක්"),
    ("75,000කට", "හැත්තෑ පන්දහසකට"),
    ("4,79,000ක්", "ලක්ෂ හතර හැත්තෑ නවදහසක්"),
    ("500000ක්", "ලක්ෂ පහක්"),
    ("100,000", "ලක්ෂය"),
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
    ("COVID-19", "COVID nineteen"),
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
    # Stress mark in front of the whole onset cluster, not inside it
    ("future", "ˈfjutʃər"),
    ("quality", "ˈkwɑləti"),
    ("throttling", "ˈθrɑtəlɪŋ"),
    ("require", "ˌriˈkwaɪər"),
    ("network", "ˈnɛtˌwərk"),
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
