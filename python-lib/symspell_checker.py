# -*- coding: utf-8 -*-
"""Use this module to check and correct misspellings"""

import re
import os
import logging
from typing import List, AnyStr
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import pandas as pd
import spacy
from symspellpy.symspellpy import SymSpell, Verbosity

from plugin_io_utils import generate_unique
from tokenizer import MultilingualTokenizer
from language_dict import SUPPORTED_LANGUAGES_SYMSPELL


class SpellChecker:
    """
    Spell checker wrapper class on top of symspellpy
    See https://symspellpy.readthedocs.io/en/latest/api/symspellpy.html#symspellpy
    """

    DEFAULT_EDIT_DISTANCE = 2
    SUGGESTION_VERBOSITY = Verbosity.TOP  # returns only the closest word
    NUM_THREADS = 4
    COLUMN_DESCRIPTION_DICT = OrderedDict(
        [
            ("corrected_text", "Text with misspellings corrected"),
            ("spelling_mistakes", "Spelling mistakes"),
            ("misspelling_count", "Number of spelling mistakes"),
        ]
    )

    def __init__(
        self,
        tokenizer: MultilingualTokenizer,
        dictionary_folder_path: AnyStr,
        custom_vocabulary_set: List[AnyStr] = [],
        edit_distance: int = DEFAULT_EDIT_DISTANCE,
        ignore_token: AnyStr = None,
        transfer_casing: bool = False,
    ):
        self.tokenizer = tokenizer
        self.custom_vocabulary_set = custom_vocabulary_set
        self.edit_distance = edit_distance
        self.ignore_token = ignore_token
        self.symspell_checker_dict = {}

    def create_symspell_checker(self, language: AnyStr, edit_distance: int) -> SymSpell:
        logging.info("Loading SymSpell checker for language: {}".format(language))
        symspell_checker = SymSpell(max_dictionary_edit_distance=edit_distance)
        freq_dict_path = self.dictionary_folder_path + "/" + language + ".txt"
        symspell_checker.load_dictionary(freq_dict_path, 0, 1)
        return symspell_checker

    def _add_symspell_checker(self, language: AnyStr) -> bool:
        added_checker = False
        if pd.isnull(language) or language == "":
            raise ValueError("Missing language code for tokenization")
        if language not in SUPPORTED_LANGUAGES_SYMSPELL.keys():
            raise ValueError("Unsupported language code for spell checker: {}".format(language))
        if language not in self.symspell_checker_dict.keys():
            self.symspell_checker_dict[language] = self.create_symspell_checker(
                language=language, edit_distance=self.edit_distance
            )
            added_checker = True
        return added_checker

    def _spellcheck_token(self, token: spacy.tokens.Token, language: AnyStr) -> (AnyStr, List, int):
        """
        Returns the corrected word if it has a correction, and the word not corrected otherwise
        See details here:
            https://symspellpy.readthedocs.io/en/latest/examples/lookup.html
        """
        raise NotImplementedError
        # return (corrected_word, misspell, misspell_count)

    def spellcheck_document(self, document: spacy.tokens.Doc, language: AnyStr) -> (AnyStr, List, int):
        spellchecked_document = document
        misspellings = []
        misspelling_count = 0
        try:
            self._add_symspell_checker(language)
            # TODO
        except ValueError as e:
            logging.warning("Spellcheking error: {} for document: {}".format(e, document.text))
            logging.info("Returning document as-is without checking spelling"
        return (document, misspellings, misspelling_count)

    def spell(self, token_lang: List) -> (AnyStr, List, int):

        """
        we did not consider word_segmentation as it is much slower.
            https://symspellpy.readthedocs.io/en/latest/examples/word_segmentation.html
        """

        token_list, lang = [token_lang[0], token_lang[1]]
        if lang not in SUPPORTED_LANGUAGES_SYMSPELL.keys():
            logging.warning("Unsupported language code: {}".format(lang))
            return ("", "", "")

        if token_list != token_list:
            return ("", "", "")

        if len(token_list) == 0:
            return ("", "", "")

        corrected_sentence = []
        misspellings = []
        misspelling_count = 0

        for token in token_list:

            # token to text
            word = token.text

            # check for url and emails
            if token.like_url or token.like_email:
                corrected_sentence.append(word)

            # check if emoji
            elif lang not in ["zh", "ja", "th"] and token._.is_emoji:
                corrected_sentence.append(word)

            # check if it is punctuation
            # if removing all punctuation leads to empty string, the token was only punct, and we keep it
            elif re.sub(r"[%s]+" % self.tokenizer.PUNCTUATION, "", word).replace("-", "") == "":
                corrected_sentence.append(word)

            # check for mentions and hashtags
            elif self.tokenizer.is_hashtag(token) or self.tokenizer.is_mention(token):
                corrected_sentence.append(word)

            elif word.lower() not in self.custom_vocabulary_set and not word.isdigit():
                (corrected_word, misspell, misspell_count) = self._fix_typos_in_word(word, lang)
                corrected_sentence.append(corrected_word)
                if len(misspell) > 0:
                    misspellings.append(misspell)
                    misspelling_count += 1
            else:
                corrected_sentence.append(word)

        corrected_sentence = " ".join(corrected_sentence)
        if len(misspellings) == 0:
            misspellings = ""
        return (corrected_sentence, misspellings, misspelling_count)

    def fix_typos_in_df(self, df: pd.DataFrame) -> pd.DataFrame:

        existing_column_names = list(df.columns)

        if self.language != "language_column":  # if language is selected from list
            # create a new column containing language
            lang_col = generate_unique("", existing_column_names, "language")
            existing_column_names.append(lang_col)
            df[lang_col] = [self.language] * df.shape[0]

        else:
            lang_col = self.language_column

        # Name creation of new columns

        # column of preprocessed text
        tokenized_col = generate_unique(self.text_column, existing_column_names, "tokenized")
        existing_column_names.append(tokenized_col)

        # the three output column name creation is based on self.COLUMN_DESCRIPTION_DICT
        self.column_description_dict = OrderedDict()

        for k, v in self.COLUMN_DESCRIPTION_DICT.items():
            new_col_name = generate_unique(k, existing_column_names, self.text_column)
            self.column_description_dict[new_col_name] = v
            existing_column_names.append(new_col_name)

        # Preprocessing of the text

        logging.info("Tokenizing text...")
        output_df = self.tokenizer.compute(df, self.text_column, tokenized_col, lang_col)
        logging.info("Tokenizing text: Done!")

        # Add new sym_spell objects

        # As we process data by chunk of 10K rows,
        # the class SymSpell is instantiated before the chunk processing.
        # Hence, the dictionaries from languages given in chunks are added
        # only if they were not already present in previous chunks.
        new_lang_code_list = list(df[lang_col].unique())
        self._add_sym_spell_objects(new_lang_code_list)

        # Parallel processing for the method _fix_typos_in_documen
        logging.info("Correcting misspellings...")
        # iterator creation over the tuple (tokenized document, laguage)
        doc_iterator = ((token, lang) for (token, lang) in zip(df[tokenized_col], df[lang_col].astype(str)))
        # Parallel computing
        # lang_output_tuple_list contains the columns as list for
        # ("corrected_text", "spelling_mistakes", "misspelling_count")
        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            lang_output_tuple_list = list(executor.map(self._fix_typos_in_document, doc_iterator))
        # copy of output column in the output dataframe
        for i, col in enumerate(self.column_description_dict.keys()):
            output_df[col] = [t[i] for t in lang_output_tuple_list]
        logging.info("Correcting misspellings: Done!")

        # remove unecessary columns
        output_df[tokenized_col]
        if self.language != "language_column":
            del output_df[lang_col]

        return output_df
