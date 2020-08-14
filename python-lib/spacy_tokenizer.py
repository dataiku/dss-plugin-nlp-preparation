# -*- coding: utf-8 -*-
"""Use this module to tokenize text data in multiple languages"""

import re
import logging
from typing import List, AnyStr

import pandas as pd

import spacy
from spacymoji import Emoji

from language_dict import SUPPORTED_LANGUAGES_SPACY
from plugin_io_utils import generate_unique


class MultilingualTokenizer:
    """Wrapper class to handle tokenization with spaCy for multiple languages"""

    DEFAULT_BATCH_SIZE = 1000

    def __init__(
        self,
        default_language: AnyStr = "xx",  # Multilingual model from spaCy
        hashtags_as_token: bool = True,
        tag_emoji: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.spacy_nlp_dict = {}
        self.hashtags_as_token = hashtags_as_token
        self.tag_emoji = tag_emoji
        self.default_language = default_language
        if default_language is not None:
            self.spacy_nlp_dict[default_language] = self.create_spacy_tokenizer(
                default_language, hashtags_as_token, tag_emoji
            )
        self.batch_size = int(batch_size)
        self.tokenized_column = None

    @staticmethod
    def create_spacy_tokenizer(
        language: AnyStr, hashtags_as_token: bool = True, tag_emoji: bool = True
    ) -> spacy.language.Language:
        """Static utility method to create a custom spaCy tokenizer for a given language

        Args:
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages
            hashtags_as_token: If True, override spaCy default tokenizer to tokenize hashtags as a whole
            tag_emoji: If True, use spacymoji extension to add a tag on emojis

        Returns:
            Instanciated spaCy Language object with the tokenizer
        """
        logging.info("Loading spaCy tokenizer for language: {}".format(language))
        nlp = spacy.blank(language)
        if hashtags_as_token:
            re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
            re_token_match = r"""({re_token_match}|#\w+)"""
            nlp.tokenizer.token_match = re.compile(re_token_match).match
            _prefixes = list(nlp.Defaults.prefixes)
            if "#" in _prefixes:
                _prefixes.remove("#")
                nlp.tokenizer.prefix_search = spacy.util.compile_prefix_regex(_prefixes).search
        if tag_emoji:
            try:
                emoji = Emoji(nlp)
                nlp.add_pipe(emoji, first=True)
            except AttributeError as e:
                # As of spacy 2.3.2 we know this will not work for Chinese, Thai and Japanese
                logging.warning("Could not load spacymoji for language: {} because of error: {}".format(language, e))
        return nlp

    def _add_spacy_tokenizer(self, language: AnyStr) -> bool:
        """Private method to add a spaCy tokenizer for a given language to the `spacy_nlp_dict` attribute

        This method only adds the tokenizer if the language code is valid and recognized among
        the list of supported languages (`SUPPORTED_LANGUAGES_SPACY` constant),
        else it will raise a ValueError exception.

        Args:
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Returns:
            True if the tokenizer was added, else False
        """
        added_tokenizer = False
        if pd.isnull(language) or language == "":
            raise ValueError("Missing language code for tokenization")
        if language not in SUPPORTED_LANGUAGES_SPACY.keys():
            raise ValueError("Unsupported language code for tokenization: {}".format(language))
        if language not in self.spacy_nlp_dict.keys():
            self.spacy_nlp_dict[language] = self.create_spacy_tokenizer(
                language, self.hashtags_as_token, self.tag_emoji
            )
            added_tokenizer = True
        return added_tokenizer

    def tokenize_list(self, text_list: List[AnyStr], language: AnyStr) -> List[spacy.tokens.Doc]:
        """Public method to tokenize a list of strings for a given language

        This method calls `_add_spacy_tokenizer` in case the requested language has not already been added.
        In case of an error in `_add_spacy_tokenizer`, it falls back to the default tokenizer.

        Args:
            text_list: List of strings
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Returns:
            List of tokenized spaCy documents
        """
        text_list = [str(t) if pd.notnull(t) else "" for t in text_list]
        try:
            self._add_spacy_tokenizer(language)
            tokenized = self.spacy_nlp_dict[language].pipe(text_list, batch_size=self.batch_size)
        except ValueError as e:
            logging.warning("Tokenization error: {} for text list: {}".format(e, text_list))
            logging.info("Fallback to default spaCy tokenizer for language: {}".format(self.default_language))
            tokenized = self.spacy_nlp_dict[self.default_language].pipe(text_list, batch_size=self.batch_size)
        return list(tokenized)

    def tokenize_df(self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr) -> pd.DataFrame:
        """Public method to tokenize a text column in a pandas DataFrame, given a language column

        This methods adds a new column to the DataFrame, whose name is saved as the `tokenized_column` attribute

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format

        Returns:
            DataFrame with all columns from the input, plus a new column with tokenized spaCy documents
        """
        message = "Tokenizing column '{}' in dataframe of {:d} rows".format(text_column, len(df.index))
        logging.info(message + "...")
        self.tokenized_column = generate_unique("tokenized", df.keys(), text_column)
        # Initialize the tokenized column to empty documents
        df[self.tokenized_column] = pd.Series(
            [self.spacy_nlp_dict[self.default_language]("")] * len(df.index), dtype="object"
        )
        language_list = df[language_column].unique()
        for language in language_list:  # iterate over languages
            language_indices = df[language_column] == language
            language_df = df.loc[language_indices, text_column]  # slicing input df by language
            tokenized_list = self.tokenize_list(text_list=language_df.values, language=language)
            df.loc[language_indices, self.tokenized_column] = pd.Series(
                tokenized_list, dtype="object", index=language_df.index,  # keep index (important)
            )
        logging.info(message + ": Done!")
        return df

    @staticmethod
    def convert_spacy_doc_to_list(
        document: spacy.tokens.Doc,
        filter_token_attributes: List[AnyStr] = [
            "is_space",
            "is_punct",
            "is_digit",
            "is_stop",
            "like_url",
            "like_email",
            "like_num",
        ],
        to_lower: bool = False,
    ) -> List[AnyStr]:
        """Static utility method to convert a spaCy document into a list of strings

        Args:
            document: A spaCy document returned by `tokenize_list` or `tokenized_df`
            filter_token_attributes: List of spaCy token attributes to filter out https://spacy.io/api/token#attributes
            to_lower: If True, convert all strings to lowercase

        Returns:
            Filtered list of strings
        """
        output_text = []
        for token in document:
            match_token_attributes = [getattr(token, t, False) for t in filter_token_attributes]
            if not any(match_token_attributes):
                text = token.text.lower() if to_lower else token.text
                output_text.append(text.strip())
        return output_text
