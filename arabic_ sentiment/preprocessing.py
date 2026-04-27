import re
from typing import List

class ArabicPreprocessor:

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel / harakat). """
        return re.sub(r'[\u064B-\u065F]', '', text)

    def normalize_alef(self, text: str) -> str:
        """Normalize all Alef variants (أ إ آ ٱ) to plain Alef (ا)."""
        return re.sub(r'[أإآٱ]', 'ا', text)

    def normalize_teh_marbuta(self, text: str) -> str:
        """Normalize Teh Marbuta (ة) to Heh (ه)."""
        return text.replace('ة', 'ه')

    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs from text."""
        return re.sub(r'https?://\S+', '', text)

    def remove_mentions(self, text: str) -> str:
        """Remove Twitter @mentions."""
        return re.sub(r'@\S+', '', text)

    def remove_hashtags(self, text: str) -> str:
        """Remove # symbol but keep the word."""
        return re.sub(r'#(\S+)', r'\1', text)

    def remove_punctuation_and_emojis(self, text: str) -> str:
        """Remove punctuation and emojis, keep Arabic letters and numbers."""
        return re.sub(r'[^\u0600-\u06FF\s0-9]', '', text)

    def remove_repeated_characters(self, text: str) -> str:
        """ Normalize elongated words, e.g., 'جميييييل' → 'جميل'."""
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def tokenize(self, text: str) -> List[str]:
        """        
        Tokenize text by splitting on whitespace.
        
        After cleaning, a simple whitespace split is sufficient.
        Filter out any empty strings.
        
        Returns:
            A list of word tokens.
            """
        return [word for word in text.split() if word]

    def preprocess(self, text: str, tokenize: bool = True):
        """
        Run the full preprocessing pipeline.
        
        Apply all steps in a sensible order, then optionally tokenize.
        
        Args:
            text:      Raw input string.
            tokenize:  If True, return List[str]; otherwise return cleaned str.
        
        Returns:
            List of tokens or a cleaned string.
        """
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_diacritics(text)
        text = self.normalize_alef(text)
        text = self.normalize_teh_marbuta(text)
        text = self.remove_punctuation_and_emojis(text)
        text = self.remove_repeated_characters(text)
        text = text.strip()
        if tokenize:
            return self.tokenize(text)
        return text