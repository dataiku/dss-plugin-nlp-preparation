# -*- coding: utf-8 -*-
import re
import logging
from typing import List, AnyStr

import pandas as pd

import spacy.lang
from spacy.tokens import Doc
from spacy.tokenizer import _get_regex_pattern
from spacymoji import Emoji

from language_dict import SUPPORTED_LANGUAGES
from dku_io_utils import generate_unique


class MultilingualTokenizer:

    DEFAULT_BATCH_SIZE = 1000

    def __init__(
        self,
        default_language="xx",
        hashtags_as_token: bool = True,
        tag_emoji: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.spacy_nlp_dict = {}
        self.hashtags_as_token = hashtags_as_token
        self.tag_emoji = tag_emoji
        self.default_language = default_language
        if default_language is not None:
            self._create_spacy_tokenizer(default_language)
        self.batch_size = int(batch_size)
        self.tokenized_column = None

    def _create_spacy_tokenizer(self, language: AnyStr):
        logging.info("Loading spaCy tokenizer for language: {}".format(language))
        nlp = spacy.blank(language)
        if self.hashtags_as_token:
            re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
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
                logging.warning("Could not load spacymoji for language: {} because of error: {}".format(language, e))
        self.spacy_nlp_dict[language] = nlp

    def _add_spacy_tokenizer(self, language):
        """
        Adds new languages to the spaCy tokenizers dictionary
        The tokenizers from languages given in chunks are added
        only if they were not already present in previous chunks.
        """
        if pd.isnull(language) or language == "":  # check for NaNs
            raise ValueError("Missing language code")
        if language not in SUPPORTED_LANGUAGES.keys():
            raise ValueError("Unsupported language code: {}".format(language))
        if language not in self.spacy_nlp_dict.keys():
            # new tokenizer is added only if not already present
            self._create_spacy_tokenizer(language)

    @staticmethod
    def convert_spacy_doc_to_list(
        document: Doc,
        to_lower: bool = False,
        filter_token_attributes: List[AnyStr] = [
            "is_space",
            "is_punct",
            "is_digit",
            "is_stop",
            "like_url",
            "like_email",
            "like_num",
        ],
    ) -> List[AnyStr]:
        """
        Converts a spacy Document into a list of strings
        Can filter out tokens which match the list of attributes computed by spaCy
        (https://spacy.io/api/token#attributes)
        """
        output_text = []
        for token in document:
            match_token_attributes = [getattr(token, t, False) for t in filter_token_attributes]
            if not any(match_token_attributes):
                text = token.text.lower() if to_lower else token.text
                output_text.append(text.strip())
        return output_text

    def tokenize_list(self, text_list: List[AnyStr], language: AnyStr) -> List[Doc]:
        text_list = list(map(str, text_list))
        try:
            self._add_spacy_tokenizer(language)
            tokenized = self.spacy_nlp_dict[language].pipe(text_list, batch_size=self.batch_size)
        except ValueError as e:
            logging.warning("Tokenization error: {} for text list: {}".format(e, text_list))
            logging.info("Fallback to default spaCy tokenizer {}".format(self.default_language))
            tokenized = self.spacy_nlp_dict[self.default_language].pipe(text_list, batch_size=self.batch_size)
        return list(tokenized)

    def tokenize_df(self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr) -> pd.DataFrame:
        """
        Returns the df with a new column containing spacy.tokens.doc.Doc
        spacy.tokens.doc.Doc can be further processed as follow (https://spacy.io/api/token):
            for token in doc:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                      token.shape_, token.is_alpha, token.is_stop)
        """
        self.tokenized_column = generate_unique("tokenized", df.keys(), text_column)
        df[self.tokenized_column] = pd.Series(
            [self.spacy_nlp_dict[self.default_language]("")] * len(df.index), dtype="object"
        )
        language_list = df[language_column].unique()
        for language in language_list:
            language_indices = df[language_column] == language
            language_df = df.loc[language_indices, text_column]
            tokenized_list = self.tokenize_list(text_list=language_df.values, language=language)
            df.loc[language_indices, self.tokenized_column] = pd.Series(
                tokenized_list, index=language_df.index, dtype="object"
            )
        return df
