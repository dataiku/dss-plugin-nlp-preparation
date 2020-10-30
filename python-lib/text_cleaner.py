# -*- coding: utf-8 -*-
"""Module with a class to clean text data in multiple languages"""

import logging
from typing import AnyStr, Dict, Iterable
from enum import Enum
from time import time
from unicodedata import normalize
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from spacy.tokens import Doc, Token

from spacy_tokenizer import MultilingualTokenizer
from plugin_io_utils import generate_unique


class TokenSimplification(Enum):
    """TODO"""

    NONE = "None"
    NORMALIZATION = "Fail"
    LEMMATIZATION_LOOKUPS = "Lemmatization (lookups)"


class UnicodeNormalization(Enum):
    """TODO"""

    NONE = "None"
    NFC = "Normalization Form: (Canonical) Composition"
    NFKC = "Normalization Form: Compatibility (K) Composition"
    NFD = "Normalization Form: (Canonical) Decomposition."
    NFKD = "Normalization Form: Compatibility (K) Decomposition"


class TextCleaner:
    """TODO

    Attributes:
        TODO

    """

    DEFAULT_NUM_THREADS = 4
    OUTPUT_COLUMN_DESCRIPTIONS = {
        **{"cleaned": "Cleaned version of the original text"},
        **MultilingualTokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES,
    }
    """TODO"""

    def __init__(
        self,
        token_filters: Iterable[AnyStr],
        token_simplification: TokenSimplification = TokenSimplification.LEMMATIZATION_LOOKUPS,
        unicode_normalization: UnicodeNormalization = UnicodeNormalization.NONE,
        lowercase: bool = True,
    ):
        """Initialization method for the TextCleaner class, with optional arguments

        Args:
            TODO

        """
        self.token_filters = token_filters
        self.token_simplification = token_simplification
        self.unicode_normalization = unicode_normalization
        self.lowercase = lowercase
        self.output_column_descriptions = (
            self.OUTPUT_COLUMN_DESCRIPTIONS.copy()
        )  # will be changed by `_prepare_df_for_cleaning`
        self._tokenizer = MultilingualTokenizer()

    def _prepare_df_for_cleaning(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr
    ) -> None:
        """TODO

        Args:
            TODO

        """
        self.output_column_descriptions = {}
        for k, v in self.OUTPUT_COLUMN_DESCRIPTIONS.items():
            if k == "cleaned":
                column_name = generate_unique(k, df.keys(), text_column)
                self.output_column_descriptions[column_name] = v
            elif k in self.token_filters:
                column_name = generate_unique(f"{v.lower()}_list", df.keys(), text_column)
                self.output_column_descriptions[column_name] = f"List of {v.lower()}s in the original text"
        self._tokenizer.tokenize_df(df, text_column, language_column, language)

    @lru_cache(maxsize=1024)  # Memory cache to avoid cleaning a token which has been cleaned before
    def clean_token(self, token: Token) -> AnyStr:
        cleaned_token = token.text
        if self.token_simplification == TokenSimplification.NORMALIZATION:
            cleaned_token = token.norm_
        elif self.token_simplification == TokenSimplification.LEMMATIZATION_LOOKUPS:
            cleaned_token = token.lemma_
        if self.lowercase:
            cleaned_token = cleaned_token.lower()
        if self.unicode_normalization != UnicodeNormalization.NONE:
            cleaned_token = normalize(unistr=cleaned_token, form=self.unicode_normalization.name)
        return cleaned_token.strip()

    def clean_document(self, document: Doc) -> Dict:
        """TODO

        Args:
            TODO

        Returns:
            TODO

        """
        output = {k: [] for k in self.OUTPUT_COLUMN_DESCRIPTIONS}
        (text_list, whitespace_list) = ([], [])
        for token in document:
            token_attributes = [t for t in self.token_filters if getattr(token, t, False) or getattr(token._, t, False)]
            if len(token_attributes) == 0:
                cleaned_token = self.clean_token(token)
                if cleaned_token:
                    text_list.append(cleaned_token)
                    try:
                        whitespace_list.append(len(token.whitespace_) != 0 or token.nbor().is_punct)
                    except IndexError:  # when reaching the end of the document, nbor() fails
                        whitespace_list.append(False)
            else:
                first_token_attribute = token_attributes[0]
                output[first_token_attribute].append(token.text)
        output["cleaned"] = "".join((t + " " if w else t for t, w in zip(text_list, whitespace_list)))
        return output

    def clean_df(self, df: pd.DataFrame, text_column, language_column, language) -> pd.DataFrame:
        """TODO

        Args:
            TODO

        Returns:
            TODO

        """
        self._prepare_df_for_cleaning(df, text_column, language_column, language)
        start = time()
        logging.info(f"Cleaning {len(df.index)} texts...")
        output = [{}] * len(df.index)
        doc_iterator = (doc for doc in df[self._tokenizer.tokenized_column])
        with ThreadPoolExecutor(max_workers=self.DEFAULT_NUM_THREADS) as executor:
            output = list(executor.map(lambda x: self.clean_document(x), doc_iterator))
        for k, v in self.OUTPUT_COLUMN_DESCRIPTIONS.items():
            if k == "cleaned":
                column_name = generate_unique(k, df.keys(), text_column)
                df[column_name] = pd.Series([d[k] for d in output])
            elif k in self.token_filters:
                column_name = generate_unique(f"{v.lower()}_list", df.keys(), text_column)
                df[column_name] = pd.Series([d[k] for d in output])
        logging.info(f"Cleaning {len(df.index)} texts: Done in {time() - start:.2f} seconds.")
        return df
