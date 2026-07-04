import os
import json
import re
import eng_to_ipa as ipa
from sinling import SinhalaTokenizer

class CodeSwitchedG2P:
    """
    Handles Code-Switched Grapheme-to-Phoneme (G2P) conversion.
    Routes English and Sinhala text to distinct modules and merges them 
    into a unified IPA sequence using a structured metadata lexicon.
    """

    def __init__(self):
        self.tokenizer = SinhalaTokenizer()
        self.en_g2p = ipa
        
        lexicon_path = os.path.join(os.path.dirname(__file__), 'lexicon.json')
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon_data = json.load(f)
                self.en_lexicon = lexicon_data.get('en_lexicon', {})
                self.si_lexicon = lexicon_data.get('si_lexicon', {})
        except Exception:
            self.en_lexicon = {}
            self.si_lexicon = {}

        self.word_acronyms = {
            key for key, entry in self.en_lexicon.items()
            if entry.get('read_as') == 'word' and key.isalpha() and len(key) <= 5
        }

        self.letter_ipa = {
            'a': 'eɪ', 'b': 'biː', 'c': 'siː', 'd': 'diː', 'e': 'iː', 'f': 'ef', 'g': 'dʒiː',
            'h': 'eɪtʃ', 'i': 'aɪ', 'j': 'dʒeɪ', 'k': 'keɪ', 'l': 'el', 'm': 'em', 'n': 'en',
            'o': 'oʊ', 'p': 'piː', 'q': 'kjuː', 'r': 'ɑːr', 's': 'es', 't': 'tiː',
            'u': 'juː', 'v': 'viː', 'w': 'dʌbəl.juː', 'x': 'eks', 'y': 'waɪ', 'z': 'ziː'
        }
        
        self.si_consonants = {
            'ක': 'k', 'ඛ': 'kh', 'ග': 'g', 'ඝ': 'gh', 'ඞ': 'ng',
            'ච': 'tʃ', 'ඡ': 'tʃh', 'ජ': 'dʒ', 'ඣ': 'dʒh', 'ඤ': 'ɲ',
            'ට': 'ʈ', 'ඨ': 'ʈh', 'ඩ': 'ɖ', 'ඪ': 'ɖh', 'ණ': 'ɳ',
            'ත': 't̪', 'ථ': 't̪h', 'ද': 'd̪', 'ධ': 'd̪h', 'න': 'n',
            'ප': 'p', 'ඵ': 'ph', 'බ': 'b', 'භ': 'bh', 'ම': 'm',
            'ය': 'j', 'ර': 'r', 'ල': 'l', 'ව': 'ʋ',
            'ශ': 'ʃ', 'ෂ': 'ʂ', 'ස': 's', 'හ': 'h', 'ළ': 'ɭ', 'ෆ': 'f'
        }
        
        self.si_vowels = {
            'අ': 'a', 'ආ': 'aː', 'ඇ': 'æ', 'ඈ': 'æː', 'ඉ': 'i', 'ඊ': 'iː',
            'උ': 'u', 'ඌ': 'uː', 'එ': 'e', 'ඒ': 'eː', 'ඔ': 'o', 'ඕ': 'oː'
        }
        
        self.si_modifiers = {
            '්': '', 'ා': 'aː', 'ැ': 'æ', 'ෑ': 'æː', 'ි': 'i', 'ී': 'iː',
            'ු': 'u', 'ූ': 'uː', 'ෙ': 'e', 'ේ': 'eː', 'ො': 'o', 'ෝ': 'oː',
            'ෞ': 'au', 'ෛ': 'ai'
        }

    def _spell_as_letters(self, text):
        """Spells an uppercase acronym letter-by-letter (e.g., MB -> em biː)."""
        parts = []
        for char in text.upper():
            if char in self.letter_ipa:
                parts.append(self.letter_ipa[char])
        return ' '.join(parts)

    def _process_english(self, text):
        """Converts English text to IPA, with smart handling for single letters and lexicon overrides."""
        word_lower = text.lower()
        
        if word_lower in self.en_lexicon:
            return self.en_lexicon[word_lower].get('ipa', '')

        if len(text) == 1 and word_lower in self.letter_ipa:
            return self.letter_ipa[word_lower]

        if text.isupper() and text.isalpha() and len(text) >= 2:
            if word_lower in self.word_acronyms:
                result = self.en_g2p.convert(text)
                return result.replace('*', '')
            return self._spell_as_letters(text)
            
        result = self.en_g2p.convert(text)
        return result.replace('*', '')

    def _process_sinhala_word(self, word):
        """Converts a Sinhala word using rules or extracts the precise IPA from the lexicon."""
        if word in self.si_lexicon:
            return self.si_lexicon[word].get('ipa', '')
            
        phonemes = []
        i = 0
        length = len(word)
        
        while i < length:
            char = word[i]
            if char in self.si_consonants:
                base_phoneme = self.si_consonants[char]
                if i + 1 < length and word[i+1] in self.si_modifiers:
                    phonemes.append(base_phoneme + self.si_modifiers[word[i+1]])
                    i += 1 
                else:
                    phonemes.append(base_phoneme + 'a')
            elif char in self.si_vowels:
                phonemes.append(self.si_vowels[char])
            i += 1
            
        return "".join(phonemes)

    def generate_unified_ipa(self, text):
        """Tokenizes text using Sinling and routes segments to appropriate G2P modules."""
        tokens = self.tokenizer.tokenize(text)
        unified_ipa = []

        for token in tokens:
            if re.match(r'^[a-zA-Z]+$', token):
                unified_ipa.append(self._process_english(token))
            elif re.match(r'^[\u0D80-\u0DFF\u200D]+$', token):
                unified_ipa.append(self._process_sinhala_word(token))
            else:
                # FIXED: Retains punctuation marks and spaces instead of dropping them
                unified_ipa.append(token)
                
        return " ".join(unified_ipa).strip()

if __name__ == "__main__":
    g2p_engine = CodeSwitchedG2P()
    sample_text = "දත්ත Database පද්ධතිය සහ ඩිජිටල් තාක්ෂණය."
    result = g2p_engine.generate_unified_ipa(sample_text)
    print(f"Input: {sample_text}\nUnified IPA: {result}")