# -*- coding: utf-8 -*-
from typing import List, AnyStr
import re
from spacy.tokens.token import Token
import spacy.lang
import pandas as pd
import logging
from plugin_io_utils import generate_unique
from spacy.tokenizer import _get_regex_pattern
from spacymoji import Emoji

from language_dict import SUPPORTED_LANGUAGES
from punctuation import PUNCTUATION


def is_url(token: Token) -> bool:
    return token.like_url


def is_email(token: Token) -> bool:
    return token.like_email


def is_mention(token: Token) -> bool:
    return str(token)[0] == "@"


def is_hashtag(token: Token) -> bool:
    return str(token)[0] == "#"


class TextPreprocessor:

    # All punctutation except '.' and '/' that are intentionally left for emails and url.
    # '.' and '/' will need to be futher removed in token if needed.
    # Otherwise, "hello." and "hello" will be two differen tokens.
    # Use the function remove_url_email_punct.
   

    def __init__(self):
        self.tokenizers = {}
        self.nlps = {}
        self.SUPPORTED_LANG_CODE = SUPPORTED_LANGUAGES.keys()

    def _custom_tokenizer(self, nlp):
        # Tokenizer that preserves hashtags and mentions
        return Tokenizer(nlp.vocab, prefix_search=self.PREFIX_TOKEN.search)

    def _add_tokenizers(self, lang_code_new_list: List[AnyStr]):
        """
        Adds tokenizers.
        The tokenizers from languages given in chunks are added
        only if they were not already present in previous chunks.
        """

        for lang_code in lang_code_new_list:

            if lang_code != lang_code:  # check for NaNs
                logging.warning("Missing language code")
                continue

            if lang_code not in self.SUPPORTED_LANG_CODE:
                logging.warning("Unsupported language code {}".format(lang_code))
                continue

            if lang_code in self.tokenizers.keys():
                # new tokenizer is added only if not already present
                continue

            else:
                # module import
                logging.info("Loading tokenizer object for language {}".format(lang_code))

                # tokenizer creation
                self.nlps[lang_code] = spacy.blank(lang_code)
                # get default pattern for tokens that don't get split
                re_token_match = _get_regex_pattern(self.nlps[lang_code].Defaults.token_match)
                # add hashtags and in-word hyphens
                re_token_match = r"""({re_token_match}|#\w+|\w+-\w+)"""
                # overwrite token_match function of the tokenizer
                self.nlps[lang_code].tokenizer.token_match = re.compile(re_token_match).match
                try: # ugly try except for zh, th, ja
                    # add emoji
                    emoji = Emoji(self.nlps[lang_code])
                    self.nlps[lang_code].add_pipe(emoji, first=True)
                except AttributeError:
                    pass

    def _normalize_text(self, doc: AnyStr, lang: AnyStr, lowercase: bool, remove_punctuation: bool) -> AnyStr:
        """
        - remove edge case: language not supported and empty string
        - lowercase
        - remove punctuation of self.PUNCTUATION. Note that further process is needed to remove '.' and '/'.
        Use remove_url_email_punct for that.
        """

        # remove edge cases
        if lang not in self.SUPPORTED_LANG_CODE:
            return ""
        if doc != doc:  # check for NaNs
            return ""
        if len(str(doc)) == 0:
            return ""

        # lowercase
        if lowercase:
            doc = str(doc).lower()
        else:
            doc = str(doc)

        # Remove leading spaces and multiple spaces
        # often created by removing punctuation and causing bad tokenized doc
        doc = " ".join(str(doc).split())

        if len(str(doc)) == 0:
            return ""
        else:
            return doc

    def _tokenize_sliced_series(
        self, sliced_series: pd.DataFrame, index: pd.core.indexes.range.RangeIndex, lang: AnyStr
    ) -> List:

        # tokenize with nlp objets
        token_list = list(self.nlps[lang].pipe(sliced_series.tolist(), disable=["tagger", "parser"]))
        # append token_list and keep same index
        token_series_sliced = pd.Series(token_list, index=index)

        return token_series_sliced

    def compute(
        self,
        df: pd.DataFrame,
        txt_col: AnyStr,
        preprocess_col: AnyStr,
        lang_col: AnyStr,
        tokenize: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
    ) -> pd.DataFrame:
        """
        Returns the df with a new column containing spacy.tokens.doc.Doc
        spacy.tokens.doc.Doc can be further processed as follow (https://spacy.io/api/token):
            for token in doc:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                      token.shape_, token.is_alpha, token.is_stop)
        """

        # Add tokenizers
        # As we process data by chunk of 10K rows,
        # the class TextPreprocessor is instantiated before the chunk processing.
        # Hence, the tokenizers from languages given in chunks are added
        # only if they were not already present in previous chunks.
        lang_list = list(df[lang_col].unique())
        self._add_tokenizers(lang_list)

        # remove edge cases, lowercase, remove punctuation
        existing_column_names = list(df.columns)
        normalized_text_column = generate_unique(txt_col, existing_column_names, "normalized")

        df[normalized_text_column] = df.apply(
            lambda x: self._normalize_text(x[txt_col], x[lang_col], lowercase, remove_punctuation), axis=1
        )

        # tokenize
        token_series = pd.Series()
        for lang in self.nlps.keys():
            # slice df with language
            df_sliced = df[df[lang_col] == lang]
            token_series = token_series.append(
                self._tokenize_sliced_series(df_sliced[normalized_text_column], df_sliced.index, lang)
            )

        df[preprocess_col] = token_series

        del df[normalized_text_column]

        return df
