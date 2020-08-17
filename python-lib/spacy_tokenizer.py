# -*- coding: utf-8 -*-
"""Module with a class to tokenize text data in multiple languages"""

import re
import logging
from typing import List, AnyStr, Union
from time import time

import pandas as pd

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
from spacymoji import Emoji

from language_dict import SUPPORTED_LANGUAGES_SPACY, SPACY_LANGUAGE_MODELS
from plugin_io_utils import generate_unique, truncate_text_list


# The constants below should cover a majority of cases for tokens with symbols and unit measurements: "8h", "90kmh", ...
SYMBOL_REGEX = (
    r"""[º°'"%&()％＆*+-<=>?\\[\]\/^_`{|}~_！？｡。＂＇（）＊＋，－／：；＜＝＞［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏]+"""
)
ORDER_UNITS = ["eme", "th", "st", "nd", "rd", "k"]
WEIGHT_UNITS = ["mg", "g", "kg", "t", "lb", "oz"]
DISTANCE_SPEED_UNITS = ["mm", "cm", "m", "km", "in", "ft", "yd", "mi", "kmh", "mph"]
TIME_UNITS = ["ns", "ms", "s", "m", "min", "h", "d", "y"]
VOLUME_UNITS = ["ml", "dl", "l", "pt", "qt", "gal"]
MISC_UNITS = ["k", "a", "v", "mol", "cd", "w", "n", "c"]
UNITS = ORDER_UNITS + WEIGHT_UNITS + DISTANCE_SPEED_UNITS + TIME_UNITS + VOLUME_UNITS + MISC_UNITS
TIME_REGEX = r"""(:|-|\.|\/|am|pm|h)+"""

# Setting custom spaCy token extensions to allow for easier filtering in downstream tasks
Token.set_extension("is_hashtag", getter=lambda token: token.text[0] == "#", force=True)
Token.set_extension("is_username", getter=lambda token: token.text[0] == "@", force=True)
Token.set_extension(
    "is_symbol", getter=lambda token: re.sub(SYMBOL_REGEX, "", token.text) == "", force=True,
)
Token.set_extension(
    "is_unit", getter=lambda token: any([token.lower_.replace(s, "").isdigit() for s in UNITS]), force=True,
)
Token.set_extension(
    "is_time", getter=lambda token: re.sub(TIME_REGEX, "", token.lower_).isdigit(), force=True,
)


class MultilingualTokenizer:
    """Wrapper class to handle tokenization with spaCy for multiple languages

    Attributes:
        default_language (str): Fallback language code in ISO 639-1 format
        use_models (bool): If True, load spaCy models for available languages.
            Slower but adds additional tagging capabilities to the pipeline.
        hashtags_as_token (bool): Treat hashtags as one token instead of two
        tag_emoji (bool): Use the spacymoji library to tag emojis
        batch_size (int): Number of documents to process in spaCy pipelines
        spacy_nlp_dict (dict): Dictionary holding spaCy Language instances (value) by language code (key)
        tokenized_column (str): Name of the dataframe column storing tokenized documents
    """

    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_NUM_PROCESS = 2
    DEFAULT_FILTER_TOKEN_ATTRIBUTES = [
        "is_space",
        "is_punct",
        "is_digit",
        "is_currency",
        "is_stop",
        "like_url",
        "like_email",
        "like_num",
        "is_emoji",
        "is_hashtag",
        "is_username",
        "is_symbol",
        "is_unit",
        "is_time",
    ]
    """list: List of available native and custom spaCy token attributes"""

    def __init__(
        self,
        default_language: AnyStr = "xx",  # Multilingual model from spaCy
        use_models: bool = False,
        hashtags_as_token: bool = True,
        tag_emoji: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialization method for the MultilingualTokenizer class, with optional arguments

        Args:
            default_language (str, optional): Fallback language code in ISO 639-1 format.
                Default is the "multilingual language code": https://spacy.io/models/xx
            use_models: If True, loads spaCy models, which is slower but allows to retrieve
                Part-of-Speech and Entities tags for downstream tasks
            hashtags_as_token (bool, optional): Treat hashtags as one token instead of two
                Default is True, which overrides the spaCy default behavior
            tag_emoji (bool, optional): Use the spacymoji library to tag emojis
                Default is True, which allows to filter or extract emojis in downstream tasks
            batch_size (int, optional): Number of documents to process in spaCy pipelines
                Default is set by the DEFAULT_BATCH_SIZE class constant
        """
        self.default_language = default_language
        self.use_models = use_models
        self.hashtags_as_token = hashtags_as_token
        self.tag_emoji = tag_emoji
        self.batch_size = int(batch_size)
        self.spacy_nlp_dict = {}
        if default_language is not None:
            self.spacy_nlp_dict[default_language] = self._create_spacy_tokenizer(default_language)
        self.tokenized_column = None  # may be changed by tokenize_df

    def _create_spacy_tokenizer(self, language: AnyStr) -> Language:
        """Private method to create a custom spaCy tokenizer for a given language

        Args:
            language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages

        Returns:
            spaCy Language instance with the tokenizer
        """
        start = time()
        logging.info("Loading tokenizer for language '{}'...".format(language))
        if language in SPACY_LANGUAGE_MODELS.keys() and self.use_models:
            try:
                nlp = spacy.load(SPACY_LANGUAGE_MODELS[language])
            except OSError as e:
                logging.warning(
                    "Spacy model not available for language: '{}' because of error: '{}'".format(language, e)
                )
                nlp = spacy.blank(language)
        else:
            nlp = spacy.blank(language)  # spaCy language without models (https://spacy.io/usage/models)
        if self.hashtags_as_token:
            re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
            re_token_match = r"""({re_token_match}|#\w+)"""
            nlp.tokenizer.token_match = re.compile(re_token_match).match
            _prefixes = list(nlp.Defaults.prefixes)
            if "#" in _prefixes:
                _prefixes.remove("#")
                nlp.tokenizer.prefix_search = spacy.util.compile_prefix_regex(_prefixes).search
        if self.tag_emoji:
            try:
                emoji = Emoji(nlp)
                nlp.add_pipe(emoji, first=True)
            except AttributeError as e:
                # As of spacy 2.3.2 we know this will not work for Chinese, Thai and Japanese
                logging.info("Emoji tokenization not available for language: {} because: {}".format(language, e))
        logging.info("Loading tokenizer for language '{}': Done in {:.2f} seconds.".format(language, time() - start))
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

        Raises:
            ValueError: If the language code is missing or not in SUPPORTED_LANGUAGES_SPACY
        """
        added_tokenizer = False
        if pd.isnull(language) or language == "":
            raise ValueError("Missing language code")
        if language not in SUPPORTED_LANGUAGES_SPACY.keys():
            raise ValueError("Unsupported language code: '{}'".format(language))
        if language not in self.spacy_nlp_dict.keys():
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
        start = time()
        logging.info("Tokenizing {:d} texts in language '{}'...".format(len(text_list), language))
        text_list = [str(t) if pd.notnull(t) else "" for t in text_list]
        try:
            self._add_spacy_tokenizer(language)
            tokenized = list(
                self.spacy_nlp_dict[language].pipe(
                    text_list, batch_size=self.batch_size, n_process=self.DEFAULT_NUM_PROCESS
                )
            )
            logging.info(
                "Tokenizing {:d} texts in language '{}': Done in {:.2f} seconds.".format(
                    len(tokenized), language, time() - start
                )
            )
        except ValueError as e:
            logging.warning(
                "Tokenization error: '{}' for text list: '{}', defaulting to fallback tokenizer".format(
                    e, truncate_text_list(text_list)
                )
            )
            tokenized = list(self.spacy_nlp_dict[self.default_language].pipe(text_list, batch_size=self.batch_size))
            logging.info(
                "Tokenizing {:d} texts using fallback tokenizer: Done in {:.2f} seconds.".format(
                    len(tokenized), time() - start
                )
            )
        return tokenized

    def tokenize_df(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr = "language_column"
    ) -> pd.DataFrame:
        """Public method to tokenize a text column in a pandas DataFrame, given a language column

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

    @staticmethod
    def convert_spacy_doc(
        document: Doc,
        output_format: AnyStr = "list",
        filter_or_keep: AnyStr = "filter",
        token_attributes: List[AnyStr] = DEFAULT_FILTER_TOKEN_ATTRIBUTES,
        lowercase: bool = True,
    ) -> Union[AnyStr, List[AnyStr]]:
        """Static method to convert a spaCy document into a list of strings or a string

        Args:
            document: A spaCy document returned by `tokenize_list` or `tokenized_df`
            output_format: Choose "list" (default) to output a list of strings
                Else, choose
            filter_or_keep: Choose "filter" (default) to remove all tokens which match the list of `token_attributes`
                Else, choose "keep" to keep only the tokens which match the list of `token_attributes`
            token_attributes: List of spaCy token attributes, cf. https://spacy.io/api/token#attributes
                User-defined token attributes are also accepted, for instance token._.yourattribute
            to_lower: If True, convert all strings to lowercase

        Returns:
            List of strings if `output_format` == "list", or text string if `output_format` == "str",
        """
        (output_text_list, whitespace_list) = ([], [])
        assert output_format in {"list", "str"}, "Choose either 'list' or 'str' option"
        assert filter_or_keep in {"filter", "keep"}, "Choose either 'filter' or 'keep' option"
        for token in document:
            match_token_attributes = [getattr(token, t, False) or getattr(token._, t, False) for t in token_attributes]
            filter_conditions = filter_or_keep == "filter" and not any(match_token_attributes)
            keep_conditions = filter_or_keep == "keep" and sum(match_token_attributes) >= 1
            if filter_conditions or keep_conditions:
                token_text = token.text.strip()
                if token_text != "":
                    output_text_list.append(token_text)
                    try:
                        whitespace_list.append(len(token.whitespace_) != 0 or token.nbor().is_punct)
                    except IndexError:  # when reaching the end of the document, nbor() fails
                        whitespace_list.append(False)
        if lowercase:
            output_text_list = [t.lower() for t in output_text_list]
        if output_format == "list":
            return output_text_list
        else:
            output_document = Doc(vocab=document.vocab, words=output_text_list, spaces=whitespace_list)
            return output_document.text
