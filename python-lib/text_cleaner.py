# -*- coding: utf-8 -*-
"""Module with a class to clean text data in multiple languages"""

import logging
from typing import AnyStr, Dict, Iterable
from enum import Enum
from time import time

import pandas as pd
from spacy.tokens import Doc

from spacy_tokenizer import MultilingualTokenizer
from plugin_io_utils import generate_unique


class TokenSimplification(Enum):
    NONE = "None"
    NORMALIZATION = "Fail"
    LEMMATIZATION_LOOKUPS = "Lemmatization (lookups)"


class UnicodeNormalization(Enum):
    NONE = "None"
    NFC = "Normalization Form: (Canonical) Composition"
    NFKC = "Normalization Form: Compatibility (K) Composition"
    NFD = "Normalization Form: (Canonical) Decomposition."
    NFKD = "Normalization Form: Compatibility (K) Decomposition"


class TextCleaner:
    """Wrapper class to handle tokenization with spaCy for multiple languages

    Attributes:
        default_language (str): Fallback language code in ISO 639-1 format
        use_models (bool): If True, load spaCy models for available languages.
            Slower but adds additional tagging capabilities to the pipeline.
        hashtags_as_token (bool): Treat hashtags as one token instead of two
        batch_size (int): Number of documents to process in spaCy pipelines
        spacy_nlp_dict (dict): Dictionary holding spaCy Language instances (value) by language code (key)
        tokenized_column (str): Name of the dataframe column storing tokenized documents
    """

    DEFAULT_NUM_THREADS = 4
    OUTPUT_COLUMN_DESCRIPTION_DICT = {}

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
        self._tokenizer = MultilingualTokenizer()

    def _prepare_df_for_cleaning(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr
    ) -> None:
        """Private method to prepare a Pandas dataframe in-place before feeding it to ``

        Tokenizes the content of the text column into a new column containing spaCy documents
        Adds new columns to hold the future outputs of the spellchecker

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format
            language: Language code in ISO 639-1 format
                If equal to "language_column" this parameter is ignored in favor of language_column
        """
        self.output_column_description_dict = {}
        for k, v in self.OUTPUT_COLUMN_DESCRIPTION_DICT.items():
            column_name = generate_unique(k, df.keys(), text_column)
            df[column_name] = pd.Series([""] * len(df.index))
            self.output_column_description_dict[column_name] = v
        self._tokenizer.tokenize_df(df, text_column, language_column, language)

    def clean_document(self, document: Doc) -> Dict:
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        pass

    def clean_df(self, df: pd.DataFrame, text_column, language_column, language) -> pd.DataFrame:
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        self._prepare_df_for_spellchecker(df, text_column, language_column, language)
        start = time()
        logging.info(f"Cleaning {len(df.index)} texts...")
        logging.info(f"Cleaning {len(df.index)} texts: Done in {time() - start:.2f}")
