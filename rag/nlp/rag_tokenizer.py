

import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import logging

from rag.nlp.vi_tokenizer import UndertheseaTokenizer

nltk.download("averaged_perceptron_tagger_eng")

class RagTokenizer:
    def __init__(self, debug=False):
        """Initialize the tokenizer with debug mode, stemmer, lemmatizer, and VNCoreNLP."""
        self.DEBUG = debug
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.vntokenizer = UndertheseaTokenizer()
        self.SPLIT_CHAR = r"(\s+|[^\w\sÀ-ỹ]+)"

    def _strQ2B(self, ustring):
        """Convert full-width characters to half-width characters."""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xFEE0
            if inside_code < 0x0020 or inside_code > 0x7E:
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        """Return input unchanged for Vietnamese."""
        return line

    def freq(self, tk):
        """Return placeholder frequency for the token."""
        return 0

    def tag(self, tk):
        """Return POS tag for the token using VNCoreNLP for Vietnamese or NLTK for English."""
        if tk.isascii():
            return nltk.pos_tag([tk])[0][1]
        else:
            return self.vntokenizer.pos_tag(tk)

    def english_normalize_(self, tks):
        return [
            (
                self.stemmer.stem(self.lemmatizer.lemmatize(t))
                if re.match(r"[a-zA-Z_-]+$", t)
                else t
            )
            for t in tks
        ]

    def _split_by_lang(self, line):
        """Split text into segments classified as Vietnamese or English based on a window of 5 elements."""
        arr = re.split(self.SPLIT_CHAR, line)
        txt_lang_pairs = []
        for i in range(len(arr)):
            if not arr[i]:
                continue
            # Combine the current element with up to two previous and two next neighbors
            combined = ""
            if i >= 2 and arr[i - 2]:
                combined += arr[i - 2]
            if i >= 1 and arr[i - 1]:
                combined += arr[i - 1]
            combined += arr[i]
            if i < len(arr) - 1 and arr[i + 1]:
                combined += arr[i + 1]
            if i < len(arr) - 2 and arr[i + 2]:
                combined += arr[i + 2]
            # Check if the combined string contains non-ASCII characters
            is_vietnamese = any(not c.isascii() for c in combined)
            txt_lang_pairs.append((arr[i], is_vietnamese))
        return txt_lang_pairs

    def tokenize(self, line):
        """Tokenize input text into Vietnamese and English tokens."""
        logging.debug(f"Input of tokenizer: {line}")
        line = self._strQ2B(line).lower()
        arr = self._split_by_lang(line)
        res = []
        for L, is_vn in arr:
            if not is_vn:
                if re.match(r"\s+$", L):
                    res.append(" ")
                elif re.match(r"[a-zA-Z0-9]+$", L):
                    tokens = nltk.word_tokenize(L)
                    res.extend(tokens)
                else:
                    res.append(L)
            else:
                tokens = self.vntokenizer.word_segment(L).split()
                res.extend(tokens)
        logging.debug(f"Output of tokenizer: {res}")
        return " ".join(res).replace("  ", " ")

    def fine_grained_tokenize(self, tks):
        """Return input as is for now, assuming VNCoreNLP provides sufficient segmentation."""
        return tks


def is_chinese(s):
    if s >= "\u4e00" and s <= "\u9fa5":
        return True
    else:
        return False


def is_number(s):
    if s >= "\u0030" and s <= "\u0039":
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= "\u0041" and s <= "\u005a") or (s >= "\u0061" and s <= "\u007a"):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split():
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

# Example usage
if __name__ == "__main__":
    tokenizer = RagTokenizer(debug=True)
    text = "Hello, I am John. Tôi thích machine learning."
    tokens = tokenizer.tokenize(text)
    print(tokens)  # Output: "Hello , tôi là John . Tôi thích machine learning ."
    print(tokenizer.tag("tôi"))  # POS tag for "tôi"
    print(tokenizer.tag("Hello"))  # POS tag for "Hello"
