# -*- coding: utf-8 -*-
"""Module with a class to tokenize text data in multiple languages"""


import regex as re
import os
import logging
from typing import List, AnyStr
from time import perf_counter
from tempfile import mkdtemp

import pandas as pd

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
from emoji import UNICODE_EMOJI
from fastcore.utils import store_attr

from language_support import SUPPORTED_LANGUAGES_SPACY, SPACY_LANGUAGE_MODELS
from plugin_io_utils import generate_unique, truncate_text_list


# Setting custom spaCy token extensions to allow for easier filtering in downstream tasks
Token.set_extension("is_hashtag", getter=lambda token: token.text[0] == "#", force=True)
Token.set_extension("is_username", getter=lambda token: token.text[0] == "@", force=True)
Token.set_extension("is_emoji", getter=lambda token: any(c in UNICODE_EMOJI for c in token.text), force=True)
SYMBOL_CHARS_REGEX = re.compile(r"(\p{M}|\p{S})+")  # matches unicode categories M (marks) and S (symbols)
Token.set_extension(
    "is_symbol",
    getter=lambda token: not token.is_punct
    and not token.is_currency
    and not getattr(token._, "is_emoji", False)
    and not re.sub(SYMBOL_CHARS_REGEX, "", token.text).strip(),
    force=True,
)
DATETIME_REGEX = re.compile(r"(:|-|\.|\/|am|pm|hrs|hr|h|minutes|mins|min|sec|s|ms|ns|y)+", flags=re.IGNORECASE)
Token.set_extension(
    "is_datetime",
    getter=lambda token: not token.like_num  # avoid conflict with existing token attribute
    and token.text[:1].isdigit()
    and re.sub(DATETIME_REGEX, "", token.text).isdigit(),
    force=True,
)
NUMERIC_SEPARATOR_REGEX = re.compile(r"[.,]")
ORDER_UNITS = {"eme", "th", "st", "nd", "rd", "k"}
WEIGHT_UNITS = {"mg", "g", "kg", "t", "lb", "oz"}
DISTANCE_SPEED_UNITS = {"mm", "cm", "m", "km", "in", "ft", "yd", "mi", "kmh", "mph"}
VOLUME_UNITS = {"ml", "dl", "l", "pt", "qt", "gal"}
MISC_UNITS = {"k", "a", "v", "mol", "cd", "w", "n", "c"}
ALL_UNITS = ORDER_UNITS | WEIGHT_UNITS | DISTANCE_SPEED_UNITS | VOLUME_UNITS | MISC_UNITS
Token.set_extension(
    "is_measure",
    getter=lambda token: not token.like_num
    and not getattr(token._, "is_datetime", False)
    and token.text[:1].isdigit()
    and any(re.sub(NUMERIC_SEPARATOR_REGEX, "", token.lower_).replace(unit, "").isdigit() for unit in ALL_UNITS),
    force=True,
)

INVISIBLE_CHARS_REGEX = re.compile(
    r"(\p{C}|\p{Z}|\p{M})+"
)  # matches unicode categories C (control chars), Z (separators) and M (marks)
Token.set_extension(
    "is_space",
    getter=lambda token: not getattr(token._, "is_symbol", False)  # avoid conflict with existing token attribute
    and (
        not "".join(c for c in token.text.strip() if c.isprintable())
        or not re.sub(INVISIBLE_CHARS_REGEX, "", token.text.strip())
    ),
    force=True,
)  # spaCy does not correctly detect all invisible characters so we need to add this custom attribute


class TokenizationError(RuntimeError):
    """Custom exception raised when one of the `MultilingualTokenizer` methods fails"""

    pass


class MultilingualTokenizer:
    """Wrapper class to handle tokenization with spaCy for multiple languages

    Attributes:
        stopwords_folder_path: Path to a folder with stopword text files (one line per stopword)
            Files should be named "{language_code}.txt" with the code in ISO 639-1 format
        use_models (bool): If True, load spaCy models for available languages.
            Slower but adds additional tagging capabilities to the pipeline.
        hashtags_as_token (bool): Treat hashtags as one token instead of two
        batch_size (int): Number of documents to process in spaCy pipelines
        max_num_characters (int): Maximum number of characters in a single text
        spacy_nlp_dict (dict): Dictionary holding spaCy Language instances (value) by language code (key)
        tokenized_column (str): Name of the dataframe column storing tokenized documents

    """

    DEFAULT_BATCH_SIZE = 1000
    MAX_NUM_CHARACTERS = 10 ** 7
    DEFAULT_NUM_PROCESS = 1
    DEFAULT_FILTER_TOKEN_ATTRIBUTES = {
        "is_space": "Whitespace",
        "is_punct": "Punctuation",
        "is_stop": "Stopword",
        "like_num": "Number",
        "is_symbol": "Symbol",
        "is_currency": "Currency sign",
        "is_measure": "Measure",
        "is_datetime": "Datetime",
        "like_url": "URL",
        "like_email": "Email",
        "is_username": "Username",
        "is_hashtag": "Hashtag",
        "is_emoji": "Emoji",
    }
    """dict: Available native and custom spaCy token attributes for filtering

    Key: name of the token attribute defined on spacy Token objects
    Value: label to designate the token attribute in the user interface
    """

    def __init__(
        self,
        stopwords_folder_path: AnyStr = None,
        use_models: bool = False,
        hashtags_as_token: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_num_characters: int = MAX_NUM_CHARACTERS,
    ):
        """Initialization method for the MultilingualTokenizer class, with optional arguments

        Args:
            stopwords_folder_path (str, optional): Path to a folder with stopword text files (one line per stopword)
                Files should be named "{language_code}.txt" with the code in ISO 639-1 format
            use_models (bool): If True (default), loads spaCy models, which is slower but allows to retrieve
                Part-of-Speech and Entities tags for downstream tasks
            hashtags_as_token (bool): Treat hashtags as one token instead of two
                Default is True, which overrides the spaCy default behavior
            batch_size (int): Number of documents to process in spaCy pipelines
                Default is set by the DEFAULT_BATCH_SIZE class constant
            max_num_characters (int): Maximum number of characters in a single text
                Default is 10 million, higher than spaCy more conservative default at 1 million

        """
        store_attr()
        self.spacy_nlp_dict = {}
        self.tokenized_column = None  # may be changed by tokenize_df

    def _create_spacy_tokenizer(self, language: AnyStr) -> Language:
        """Private method to create a custom spaCy tokenizer for a given language

        Args:
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Returns:
            spaCy Language instance with the tokenizer

        Raises:
            TokenizationError: If something went wrong with the tokenizer creation

        """
        start = perf_counter()
        logging.info(f"Loading tokenizer for language '{language}'...")
        try:
            if language == "th":  # PyThaiNLP requires a "data directory" even if nothing needs to be downloaded
                os.environ["PYTHAINLP_DATA_DIR"] = mkdtemp()  # dummy temp directory
            if language in SPACY_LANGUAGE_MODELS and self.use_models:
                nlp = spacy.load(SPACY_LANGUAGE_MODELS[language])
            else:
                nlp = spacy.blank(language)  # spaCy language without models (https://spacy.io/usage/models)
            nlp.max_length = self.max_num_characters
        except (ValueError, OSError) as e:
            raise TokenizationError(
                f"SpaCy tokenization not available for language '{language}' because of error: '{e}'"
            )
        if self.hashtags_as_token:
            re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
            re_token_match = r"""({re_token_match}|#\w+)"""
            nlp.tokenizer.token_match = re.compile(re_token_match).match
            _prefixes = list(nlp.Defaults.prefixes)
            if "#" in _prefixes:
                _prefixes.remove("#")
                nlp.tokenizer.prefix_search = spacy.util.compile_prefix_regex(_prefixes).search
        if self.stopwords_folder_path and language in SUPPORTED_LANGUAGES_SPACY:
            self._customize_stopwords(nlp, language)
        logging.info(f"Loading tokenizer for language '{language}': done in {perf_counter() - start:.2f} seconds")
        return nlp

    def _customize_stopwords(self, nlp: Language, language: AnyStr) -> None:
        """Private method to customize stopwords for a given spaCy language

        Args:
            nlp: Instanciated spaCy language
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Raises:
            TokenizationError: If something went wrong with the stopword customization

        """
        try:
            stopwords_file_path = os.path.join(self.stopwords_folder_path, f"{language}.txt")
            with open(stopwords_file_path) as f:
                custom_stopwords = set(f.read().splitlines())
            for word in custom_stopwords:
                nlp.vocab[word].is_stop = True
                nlp.vocab[word.capitalize()].is_stop = True
                nlp.vocab[word.upper()].is_stop = True
            for word in nlp.Defaults.stop_words:
                if word.lower() not in custom_stopwords:
                    nlp.vocab[word].is_stop = False
                    nlp.vocab[word.capitalize()].is_stop = False
                    nlp.vocab[word.upper()].is_stop = False
            nlp.Defaults.stop_words = custom_stopwords
        except (ValueError, OSError) as e:
            raise TokenizationError(f"Stopword file for language '{language}' not available because of error: '{e}'")

    def _add_spacy_tokenizer(self, language: AnyStr) -> bool:
        """Private method to add a spaCy tokenizer for a given language to the `spacy_nlp_dict` attribute

        This method only adds the tokenizer if the language code is valid and recognized among
        the list of supported languages (`SUPPORTED_LANGUAGES_SPACY` constant),
        else it will raise a TokenizationError exception.

        Args:
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Returns:
            True if the tokenizer was added, else False

        Raises:
            TokenizationError: If the language code is missing or not in SUPPORTED_LANGUAGES_SPACY

        """
        added_tokenizer = False
        if pd.isnull(language) or language == "":
            raise TokenizationError("Missing language code")
        if language not in SUPPORTED_LANGUAGES_SPACY:
            raise TokenizationError(f"Unsupported language code: '{language}'")
        if language not in self.spacy_nlp_dict:
            self.spacy_nlp_dict[language] = self._create_spacy_tokenizer(language)
            added_tokenizer = True
        return added_tokenizer

    def tokenize_list(self, text_list: List[AnyStr], language: AnyStr) -> List[Doc]:
        """Public method to tokenize a list of strings for a given language

        This method calls `_add_spacy_tokenizer` in case the requested language has not already been added.
        In case of an error in `_add_spacy_tokenizer`, it falls back to the default tokenizer.

        Args:
            text_list: List of strings
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Returns:
            List of tokenized spaCy documents

        """
        start = perf_counter()
        logging.info(f"Tokenizing {len(text_list)} document(s) in language '{language}'...")
        text_list = [str(t) if pd.notnull(t) else "" for t in text_list]
        try:
            self._add_spacy_tokenizer(language)
        except TokenizationError as e:
            raise TokenizationError(f"Tokenization error: {e} for document(s): '{truncate_text_list(text_list)}'")
        tokenized = list(
            self.spacy_nlp_dict[language].pipe(
                text_list, batch_size=self.batch_size, n_process=self.DEFAULT_NUM_PROCESS
            )
        )
        logging.info(
            f"Tokenizing {len(tokenized)} document(s) in language '{language}': "
            + f"done in {perf_counter() - start:.2f} seconds"
        )
        return tokenized

    def tokenize_df(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr = "", language: AnyStr = "language_column"
    ) -> pd.DataFrame:
        """Public method to tokenize a text column in a pandas DataFrame, given language information

        This methods adds a new column to the DataFrame, whose name is saved as the `tokenized_column` attribute

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages
                if equal to "language_column" this parameter is ignored in favor of language_column

        Returns:
            DataFrame with all columns from the input, plus a new column with tokenized spaCy documents

        """
        self.tokenized_column = generate_unique("tokenized", df.keys(), text_column)
        # Initialize the tokenized column to empty documents
        df[self.tokenized_column] = pd.Series([Doc(Vocab())] * len(df.index), dtype="object")
        if language == "language_column":
            languages = df[language_column].dropna().unique()
            unsupported_languages = set(languages) - set(SUPPORTED_LANGUAGES_SPACY.keys())
            if unsupported_languages:
                raise TokenizationError(
                    f"Found {len(unsupported_languages)} unsupported languages in dataset: {unsupported_languages}"
                )
            for lang in languages:  # iterate over languages
                language_indices = df[language_column] == lang
                text_slice = df.loc[language_indices, text_column]  # slicing input df by language
                if len(text_slice) != 0:
                    tokenized_list = self.tokenize_list(text_list=text_slice, language=lang)
                    df.loc[language_indices, self.tokenized_column] = pd.Series(
                        tokenized_list, dtype="object", index=text_slice.index,  # keep index (important)
                    )
        else:
            tokenized_list = self.tokenize_list(text_list=df[text_column], language=language)
            df[self.tokenized_column] = tokenized_list
        return df
