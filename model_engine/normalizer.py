import re

class SinhalaTextNormalizer:
    """
    Stage 1 Pipeline: Normalizes raw Sinhala and English mixed text for TTS processing.
    Fully optimized for Arithmetic, Relational, and Logical Operators in O/L ICT.
    """
    
    def __init__(self):
        # Full Operator & Symbol Mapping
        self.symbols_map = {
            # Logical Operators
            '&&': ' සහ ',
            '||': ' හෝ ',
            '!': ' නොවේ ',
            
            # Comparison / Relational Operators
            '<=': ' කුඩායි හෝ සමානයි ',
            '>=': ' විශාලයි හෝ සමානයි ',
            '<>': ' අසමානයි ',
            '!=': ' අසමානයි ',
            '==': ' සමානයි ',
            '<': ' කුඩායි ',
            '>': ' විශාලයි ',
            
            # Arithmetic Operators
            '+': ' ධන ',
            '-': ' සෘණ ',
            '*': ' ගුණ කිරීම ',
            '/': ' බෙදීම ',
            '^': ' බලය ',
            '%': ' ප්‍රතිශතය ',
            
            # Assignment & Other Symbols
            '=': ' සමානයි ',
            '&': ' සහ ',
            '@': ' ඇට් ',
            '$': ' ඩොලර් ',
            '#': ' හැෂ් '
        }

        self.url_map = {
            'www.': 'ඩබ්ලිව් ඩබ්ලිව් ඩබ්ලිව් ඩොට් ',
            '.com': ' ඩොට් කොම් ',
            '.lk': ' ඩොට් එල් කේ ',
            '.org': ' ඩොට් ඕර්ජී ',
            '.edu': ' ඩොට් එඩියු ',
            '.gov': ' ඩොට් ගව් '
        }

    def expand_urls_and_ips(self, text):
        """Converts web domains and correctly formats IP addresses/decimals."""
        for symbol, word in self.url_map.items():
            text = re.sub(re.escape(symbol), word, text, flags=re.IGNORECASE)
        
        # Identify dots between numbers (e.g., 192.168) and replace with text
        text = re.sub(r'(?<=\d)\.(?=\d)', ' ඩොට් ', text)
        return text

    def expand_symbols(self, text):
        """
        Converts mathematical and logical symbols into Sinhala words.
        Safely sorts by length descending to prevent partial matching.
        """
        # Sort keys by length (longest first: '==' comes before '=')
        sorted_symbols = sorted(self.symbols_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for symbol, word in sorted_symbols:
            text = text.replace(symbol, word)
        return text

    def get_sinhala_number(self, num_str):
        """Dynamically generates Sinhala phonetic words for numbers 0 - 999."""
        num = int(num_str)
        if num == 0: return "බිංදුව"

        ones = ["", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය"]
        tens_exact = ["", "දහය", "විස්ස", "තිහ", "හතළිහ", "පනහ", "හැට", "හැත්තෑව", "අසූව", "අනූව"]
        tens_prefix = ["", "දහ", "විසි", "තිස්", "හතළිස්", "පනස්", "හැට", "හැත්තෑ", "අසූ", "අනූ"]
        teens = ["දහය", "එකොළහ", "දොළහ", "දහතුන", "දහහතර", "පහළොව", "දහසය", "දහහත", "දහඅට", "දහනවය"]
        hundreds_exact = ["", "සියය", "දෙසියය", "තුන්සියය", "හාරසියය", "පන්සියය", "හයසියය", "හත්සියය", "අටසියය", "නවසියය"]
        hundreds_prefix = ["", "එකසිය", "දෙසිය", "තුන්සිය", "හාරසිය", "පන්සිය", "හයසිය", "හත්සිය", "අටසිය", "නවසිය"]

        words = []
        
        # Hardcoded fallbacks for specific large ICT numbers
        if num == 1000: return "දහස"
        if num == 1024: return "දහස් විසි හතර"
        if num > 999: 
            return " ".join([ones[int(d)] if int(d)>0 else "බිංදුව" for d in num_str])

        # Processing Hundreds
        if num >= 100:
            h = num // 100
            rem = num % 100
            if rem == 0:
                words.append(hundreds_exact[h])
            else:
                words.append(hundreds_prefix[h])
            num = rem

        # Processing Tens and Ones
        if 10 <= num <= 19:
            words.append(teens[num - 10])
        elif num >= 20 or (num > 0 and len(words) == 0):
            t = num // 10
            o = num % 10
            if t > 0:
                if o == 0:
                    words.append(tens_exact[t])
                else:
                    words.append(tens_prefix[t])
            if o > 0:
                words.append(ones[o])
        elif 0 < num <= 9 and len(words) > 0:
            words.append(ones[num])

        return " ".join(words).strip()

    def expand_numbers(self, text):
        """Finds all digits in the text and converts them using get_sinhala_number."""
        def replace_num(match):
            return self.get_sinhala_number(match.group())
        return re.sub(r'\b\d+\b', replace_num, text)

    def remove_noise(self, text):
        """Removes Emojis and undefined special characters."""
        text = re.sub(r'[^\u0D80-\u0DFFa-zA-Z\s.,?!\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize(self, text):
        """Executes the full preprocessing pipeline in logical order."""
        text = self.expand_urls_and_ips(text)
        text = self.expand_symbols(text)
        text = self.expand_numbers(text)
        text = self.remove_noise(text)
        return text

# --- Testing the Normalizer Pipeline ---
if __name__ == "__main__":
    normalizer = SinhalaTextNormalizer()
    
    # Complex Operator and Number Test Cases
    test_cases = [
        "A == B && C != 10",
        "Spreadsheet එකේ 2 ^ 3 යනු 8 වේ.",
        "Total >= 50 || Total <= 100",
        "IP Address එක 192.168.1.1 වේ."
    ]
    
    print("--- Stage 1: Text Normalization Output ---")
    for i, raw_text in enumerate(test_cases, 1):
        cleaned_text = normalizer.normalize(raw_text)
        print(f"\nTest {i}:")
        print(f"Raw Input  : {raw_text}")
        print(f"Normalized : {cleaned_text}")