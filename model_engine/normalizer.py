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

    def expand_percentages(self, text):
        """Converts '100 %' into natural Sinhala phrasing: 'සියයට සියයක්'."""
        def replace_pct(match):
            num_word = self.get_sinhala_number(match.group(1))
            # FIXED: Grammar bug fixed. Now formats as "සියයට {number}ක්"
            return f" සියයට {num_word}ක් " 
        text = re.sub(r'(\d+)\s*%', replace_pct, text)
        text = text.replace('%', ' ප්‍රතිශතය ')
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

    def _sinhala_below_100(self, n):
        """Natural Sinhala words for 1–99 (cardinal forms)."""
        ones = ["", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය"]
        teens = ["දහය", "එකොළහ", "දොළහ", "දහතුන", "දහහතර", "පහළොව", "දහසය", "දහහත", "දහඅට", "දහනවය"]
        tens_exact = ["", "දහය", "විස්ස", "තිහ", "හතළිහ", "පනහ", "හැට", "හැත්තෑව", "අසූව", "අනූව"]
        tens_prefix = ["", "දහ", "විසි", "තිස්", "හතළිස්", "පනස්", "හැට", "හැත්තෑ", "අසූ", "අනූ"]

        if n == 0:
            return ""
        if 10 <= n <= 19:
            return teens[n - 10]
        tens, unit = divmod(n, 10)
        parts = []
        if tens:
            parts.append(tens_exact[tens] if unit == 0 else tens_prefix[tens])
        if unit:
            parts.append(ones[unit])
        return " ".join(parts)

    def _sinhala_below_1000(self, n):
        """Natural Sinhala words for 1–999."""
        hundreds_exact = ["", "සියය", "දෙසියය", "තුන්සියය", "හාරසියය", "පන්සියය", "හයසියය", "හත්සියය", "අටසියය", "නවසියය"]
        hundreds_prefix = ["", "එකසිය", "දෙසිය", "තුන්සිය", "හාරසිය", "පන්සිය", "හයසිය", "හත්සිය", "අටසිය", "නවසිය"]

        if n < 100:
            return self._sinhala_below_100(n)
        hundreds, rem = divmod(n, 100)
        parts = [hundreds_exact[hundreds] if rem == 0 else hundreds_prefix[hundreds]]
        if rem:
            parts.append(self._sinhala_below_100(rem))
        return " ".join(p for p in parts if p)

    def _sinhala_thousands(self, thousands, remainder):
        """Forms like දහස, එක්දහස් විසි හතර, හතළිස් අට දහස්."""
        attr = ["", "එක්", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව"]
        if thousands == 1:
            prefix = "එක්දහස්" if remainder else "දහස"
        elif thousands < 10:
            prefix = f"{attr[thousands]}දහස්" if remainder else f"{attr[thousands]}දහස"
        else:
            prefix = f"{self.get_sinhala_number(str(thousands))} දහස්"
        parts = [prefix]
        if remainder:
            parts.append(self._sinhala_below_1000(remainder))
        return " ".join(parts)

    def _sinhala_millions(self, millions, remainder):
        """Forms like මිලියන, එක් මිලියන හතළිස් අට දහස්."""
        attr = ["", "එක්", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව"]
        if millions == 1:
            prefix = "එක් මිලියන" if remainder else "මිලියන"
        elif millions < 10:
            prefix = f"{attr[millions]} මිලියන"
        else:
            prefix = f"{self.get_sinhala_number(str(millions))} මිලියන"
        parts = [prefix]
        if remainder:
            parts.append(self.get_sinhala_number(str(remainder)))
        return " ".join(parts)

    def get_sinhala_number(self, num_str):
        """Generates natural Sinhala cardinal words (e.g. 1024 → එක්දහස් විසි හතර)."""
        num = int(num_str)
        if num == 0:
            return "බිංදුව"
        if num < 1000:
            return self._sinhala_below_1000(num)
        if num < 1_000_000:
            thousands, rem = divmod(num, 1000)
            return self._sinhala_thousands(thousands, rem)
        if num < 1_000_000_000:
            millions, rem = divmod(num, 1_000_000)
            return self._sinhala_millions(millions, rem)
        return " ".join(self.get_sinhala_number(d) for d in num_str if d != "0") or "බිංදුව"

    def expand_numbers(self, text):
        """Finds all digits in the text and converts them using get_sinhala_number."""
        def replace_num(match):
            return self.get_sinhala_number(match.group())
        return re.sub(r'\b\d+\b', replace_num, text)

    def remove_noise(self, text):
        """Removes Emojis and undefined special characters.
        Retains Sinhala block, ZWJ (\u200D) for conjuncts, English, and basic punctuation."""
        text = re.sub(r'[^\u0D80-\u0DFFa-zA-Z\s.,?!\'"\u200D]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize(self, text):
        """Executes the full preprocessing pipeline in logical order."""
        text = self.expand_urls_and_ips(text)
        text = self.expand_percentages(text)
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