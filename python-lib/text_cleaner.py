# -*- coding: utf-8 -*-
"""Module with a class to clean text data in multiple languages"""

import logging
import re
from typing import AnyStr, Dict, Set
from enum import Enum
from time import time
from unicodedata import normalize
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from spacy.tokens import Doc, Token
from fastcore.utils import store_attr

from spacy_tokenizer import MultilingualTokenizer
from plugin_io_utils import generate_unique


WHITESPACE_REGEX = re.compile(r" +")


class UnicodeNormalization(Enum):
    """Enum class to identify each possible unicode normalization method"""

    NONE = "None"
    NFC = "Normalization Form: (Canonical) Composition"
    NFKC = "Normalization Form: Compatibility (K) Composition"
    NFD = "Normalization Form: (Canonical) Decomposition."
    NFKD = "Normalization Form: Compatibility (K) Decomposition"


class TextCleaner:
    """Clean dirty text data: tokenize, filter, lemmatize (and unicode-normalize as an option)

    Relies on spaCy for all steps, with custom-defined token attribute filters

    Attributes:
        tokenizer (MultilingualTokenizer): Tokenizer instance to handle the initial tokenization step
        token_filters (set): Set of spaCy token attributes to filter out
        lemmatization (bool): If True, lemmatize tokens using spaCy lookups data
        lowercase (bool): If True, convert everything to lowercase after filter and lemmatization steps
        unicode_normalization (UnicodeNormalization): Unicode normalization method (final post-processing)
        output_column_descriptions (dict): Column names (key) and descriptions (value) for the output dataset
            This attribute is computed automatically based on the input dataframe to clean

    """

    DEFAULT_NUM_THREADS = 4
    OUTPUT_COLUMN_DESCRIPTIONS = {
        **{"cleaned": "Cleaned version of the original text"},
        **MultilingualTokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES,
    }
    """dict: Default column names (key) and descriptions (value) for the output dataset"""

    def __init__(
        self,
        tokenizer: MultilingualTokenizer,
        token_filters: Set[AnyStr],
        lemmatization: bool = True,
        lowercase: bool = True,
        unicode_normalization: UnicodeNormalization = UnicodeNormalization.NONE,
    ):
        """Initialization method for the TextCleaner class, with optional arguments

        Args:
            tokenizer (MultilingualTokenizer): Tokenizer instance to handle the initial tokenization step
            token_filters (set): Set of spaCy token attributes to filter out
                Available token filters are defined in MultilingualTokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES
            lemmatization (bool, optional): If True, lemmatize tokens using spaCy lookups data
                Default is True, which simplifies all tokens to their lemma e.g. going -> go, mice -> mouse
            lowercase (bool, optional): If True, convert everything to lowercase after filter and lemmatization steps
                Default is True
            unicode_normalization (UnicodeNormalization, optional): Unicode normalization method (final post-processing)
                Default is not to apply normalization. Beware that it's a more complex topic than it looks.
                Read https://en.wikipedia.org/wiki/Unicode_equivalence if you want to understand more
                TL;DR: human languages are a mess => Unicode is a mess too

        """
        store_attr()
        self.output_column_descriptions = (
            self.OUTPUT_COLUMN_DESCRIPTIONS.copy()
        )  # will be changed by `_prepare_df_for_cleaning`

    def _prepare_df_for_cleaning(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr
    ) -> None:
        """Private method to prepare a Pandas dataframe in-place before feeding it to the `self.clean_df` method

        Tokenizes the content of the text column into a new column containing spaCy documents
        Adds new columns to hold the future outputs of the cleaner method

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format
            language: Language code in ISO 639-1 format
                If equal to "language_column" this parameter is ignored in favor of language_column

        """
        self.output_column_descriptions = {}
        for k, v in self.OUTPUT_COLUMN_DESCRIPTIONS.items():
            if k == "cleaned":
                column_name = generate_unique(k, df.keys(), text_column)
                self.output_column_descriptions[column_name] = v
            elif k in self.token_filters:
                column_name = generate_unique(f"{v.lower()}s", df.keys(), text_column)
                self.output_column_descriptions[column_name] = f"{v}s in the original text"
        self.tokenizer.tokenize_df(df, text_column, language_column, language)

    @lru_cache(maxsize=1024)  # Memory cache to save water: avoid cleaning a token which has been cleaned before
    def clean_token(self, token: Token) -> AnyStr:
        """Public method to clean a dirty spaCy token after it passed the filter step

        Applies `self.lemmatization`, `self.lowercase`, `self.unicode_normalization` soap on-demand

        Args:
            token: dirty spaCy token

        Returns:
            Cleaned string

        """
        cleaned_text = token.text
        if self.lemmatization:
            lemma = token.lemma_
            cleaned_text = lemma if lemma != "-PRON-" else token.text
        if self.lowercase:
            cleaned_text = cleaned_text.lower()
        if self.unicode_normalization != UnicodeNormalization.NONE:
            cleaned_text = normalize(self.unicode_normalization.name, cleaned_text)
        if token.is_space:
            cleaned_text = " "
        return cleaned_text

    def clean_document(self, document: Doc) -> Dict:
        """Public method to clean a dirty spaCy document after tokenization

        Iterate on the document, token-by-token:
            - Filter out unwanted tokens if they match selected `self.token_filters`
            - If the token passes the filter step, feed it to `self.clean_token`
            - Else, add it to a concatenated string for the matching token filter attribute
        All outputs are reconstituted as strings with whitespace separating each token

        Args:
            document: dirty SpaCy document

        Returns:
            Dictionary with the same keys as `self.OUTPUT_COLUMN_DESCRIPTIONS` and corresponding values:
                - cleaned text after filter, lemmatization, lowercase and unicode normalization steps
                - concatenation of filtered tokens based on selected `self.token_filters`

        """
        output = {k: "" for k in self.OUTPUT_COLUMN_DESCRIPTIONS}
        output["cleaned"] = ""
        for token in document:
            token_attributes = [t for t in self.token_filters if getattr(token, t, False) or getattr(token._, t, False)]
            if token_attributes:
                first_token_attribute = token_attributes[0]
                output[first_token_attribute] += token.lower_ if self.lowercase else token.text
                try:
                    if token.whitespace_ or token.nbor().is_punct or token.nbor().is_space:
                        output[first_token_attribute] += " "
                except IndexError:  # when reaching the end of the document, nbor() fails
                    pass
            else:
                cleaned_token = self.clean_token(token)
                if cleaned_token:
                    output["cleaned"] += cleaned_token
                try:
                    if (
                        token.whitespace_
                        or ("is_punct" in self.token_filters and token.nbor().is_punct)
                        or (self.lemmatization and token.nbor().idx == token.idx + len(token))
                    ):
                        output["cleaned"] += " "
                except IndexError:  # when reaching the end of the document, nbor() fails
                    pass
        for k in output:
            output[k] = re.sub(WHITESPACE_REGEX, " ", output[k]).strip()  # remove multiple spaces
        return output

    def clean_df(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr = "", language: AnyStr = "language_column",
    ) -> pd.DataFrame:
        """Public method to clean a text column in a pandas DataFrame, given language information

        Prepare the dataframe with `self._prepare_df_for_cleaning` to obtain a new column with spaCy documents
        Run `self.clean_document` on all documents with multithreading
        Format the output dataframe

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format
            language: Language code in ISO 639-1 format
                If equal to "language_column" this parameter is ignored in favor of language_column

        Returns:
            Input dataframe with new columns at the end:
                - Cleaned text after filter, lemmatization, lowercase and unicode normalization steps
                - One column for each selected `self.token_filters` with a concatenation of filtered tokens

        """
        self._prepare_df_for_cleaning(df, text_column, language_column, language)
        start = time()
        logging.info(f"Cleaning {len(df.index)} texts...")
        output = [{}] * len(df.index)
        doc_iterator = (doc for doc in df[self.tokenizer.tokenized_column])
        with ThreadPoolExecutor(max_workers=self.DEFAULT_NUM_THREADS) as executor:
            output = list(executor.map(lambda x: self.clean_document(x), doc_iterator))
        for k, v in self.OUTPUT_COLUMN_DESCRIPTIONS.items():
            if k == "cleaned":
                column_name = generate_unique(k, df.keys(), text_column)
                df[column_name] = [d.get(k, "") for d in output]
            elif k in self.token_filters:
                column_name = generate_unique(f"{v.lower()}s", df.keys(), text_column)
                df[column_name] = [d.get(k, "") for d in output]
        logging.info(f"Cleaning {len(df.index)} texts: Done in {time() - start:.2f} seconds.")
        del df[self.tokenizer.tokenized_column]
        return df
