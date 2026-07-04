import sys
from normalizer import SinhalaTextNormalizer
from g2p import CodeSwitchedG2P

class TextProcessingPipeline:
    """
    Main NLP Pipeline for the Adaptive Sinhala TTS System.
    Integrates Text Normalization (Stage 1) and Code-Switched G2P (Stage 2).
    Outputs the final IPA sequence required for the acoustic model.
    """
    
    def __init__(self):
        # Initialize Stage 1 (Text Cleaning & Expansion)
        self.normalizer = SinhalaTextNormalizer()
        
        # Initialize Stage 2 (Grapheme-to-Phoneme Conversion)
        self.g2p_engine = CodeSwitchedG2P()
        
    def process(self, raw_text):
        """
        Executes the end-to-end text processing pipeline.
        """
        # Stage 1: Clean text and expand symbols/numbers
        normalized_text = self.normalizer.normalize(raw_text)
        
        # Stage 2: Convert cleaned text to IPA phonemes
        ipa_sequence = self.g2p_engine.generate_unified_ipa(normalized_text)
        
        return {
            "raw_input": raw_text,
            "normalized_text": normalized_text,
            "ipa_sequence": ipa_sequence
        }

# --- Testing the Integration Pipeline ---
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pipeline = TextProcessingPipeline()
    
    # Complex O/L ICT test query testing both Stage 1 and Stage 2
    test_query = "RAM එක 1024 MB වේ. A == B නම් 100 % නිවැරදියි."
    
    print("\n--- End-to-End NLP Pipeline Test ---")
    try:
        result = pipeline.process(test_query)
        print(f"1. Raw Input      : {result['raw_input']}")
        print(f"2. Normalized Text: {result['normalized_text']}")
        print(f"3. Final IPA      : {result['ipa_sequence']}\n")
        print("Integration Successful! 🚀")
    except Exception as e:
        print(f"\n[ERROR] Integration Failed.\nDetails: {e}")