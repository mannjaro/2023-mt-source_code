import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


class Preprocess:
    def __init__(self, lang: str = "english") -> None:
        self.stopwords = stopwords.words(lang)
        self.__download_()

    def __download_(self) -> None:
        nltk.download("omw-1.4", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("tagsets", quiet=True)
        nltk.download("stopwords", quiet=True)

    def lemmatize_sentence(self, sentence: str) -> str:
        ret_lines = []
        for line in sentence.splitlines():
            lemmatizer = WordNetLemmatizer()
            clean_tokens = []
            code_regex = re.compile(
                # "[!\"#$%&'\\\\()*+,.;<=>?@[\\]^`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
                "[!-/:-@[-`{-~]"
                # r"[!@[`{~*<=>?-]`]"
            )
            line = " ".join(line.split("-"))
            line = " ".join(line.split("_"))
            line = " ".join(line.split("/"))
            for token, tag in pos_tag(word_tokenize(line)):
                token = code_regex.sub("", token)
                if tag.startswith("NN"):
                    pos = "n"
                elif tag.startswith("VB"):
                    pos = "v"
                else:
                    pos = "a"

                token = lemmatizer.lemmatize(token, pos)

                if (
                    len(token) > 3
                    and len(token) < 20
                    and token not in string.punctuation
                    # and token.lower() not in self.stopwords
                    and token.isascii()
                    and not token.isdigit()
                ):
                    clean_tokens.append(token.lower())
            ret_lines.append(" ".join(clean_tokens))
        return "\n".join(ret_lines)
