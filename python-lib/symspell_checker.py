# -*- coding: utf-8 -*-
"""Use this module to check and correct misspellings"""

import logging
from typing import List, AnyStr, Set
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import pandas as pd
from spacy.tokens import Token, Doc
from spacy.vocab import Vocab
from symspellpy.symspellpy import SymSpell, Verbosity

from plugin_io_utils import generate_unique, move_columns_after
from spacy_tokenizer import MultilingualTokenizer
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
        custom_vocabulary_set: Set[AnyStr] = set(),
        edit_distance: int = DEFAULT_EDIT_DISTANCE,
        ignore_token: AnyStr = None,
        transfer_casing: bool = True,
    ):
        self.tokenizer = tokenizer
        self.dictionary_folder_path = dictionary_folder_path
        self.custom_vocabulary_set = custom_vocabulary_set
        self.edit_distance = edit_distance
        self.ignore_token = ignore_token
        self.transfer_casing = transfer_casing
        self.symspell_checker_dict = {}
        self.column_description_dict = self.COLUMN_DESCRIPTION_DICT  # may be changed by check_df
        Token.set_extension("is_misspelled", default=False, force=True)
        Token.set_extension("correction", default="", force=True)

    def _create_symspell_checker(self, language: AnyStr, edit_distance: int) -> SymSpell:
        logging.info("Loading SymSpell checker for language: {}".format(language))
        symspell_checker = SymSpell(max_dictionary_edit_distance=edit_distance)
        frequency_dict_path = self.dictionary_folder_path + "/" + language + ".txt"
        (term_index, count_index) = (0, 1)
        if language in {"ar", "fa", "he"}:  # These dictionaries have reverse index
            (term_index, count_index) = (1, 0)
        symspell_checker.load_dictionary(
            frequency_dict_path, term_index=term_index, count_index=count_index, encoding="utf-8"
        )
        return symspell_checker

    def _add_symspell_checker(self, language: AnyStr) -> bool:
        added_checker = False
        if pd.isnull(language) or language == "":
            raise ValueError("Missing language code")
        if language not in SUPPORTED_LANGUAGES_SYMSPELL.keys():
            raise ValueError("Unsupported language code: {}".format(language))
        if language not in self.symspell_checker_dict.keys():
            self.symspell_checker_dict[language] = self._create_symspell_checker(
                language=language, edit_distance=self.edit_distance
            )
            added_checker = True
        return added_checker

    def check_token(self, token: Token, language: AnyStr):
        match_token_attributes = [getattr(token, t, False) for t in self.tokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES]
        match_token_attributes.append(token.text[0] in {"#", "@"})
        if language not in {"zh", "ja", "th"}:
            # As of spacy 2.3.2 spacymoji does not work for Chinese, Thai and Japanese
            match_token_attributes.append(token._.is_emoji)
        if not any(match_token_attributes) and token.text.lower() not in self.custom_vocabulary_set:
            correction_suggestions = self.symspell_checker_dict[language].lookup(
                token.text,
                verbosity=self.SUGGESTION_VERBOSITY,
                max_edit_distance=self.edit_distance,
                ignore_token=self.ignore_token,
                transfer_casing=self.transfer_casing,
            )
            if len(correction_suggestions) != 0:
                correction = correction_suggestions[0].term
                if correction.lower() != token.text.lower():
                    token._.is_misspelled = True
                    token._.correction = correction

    def check_document(self, document: Doc, language: AnyStr) -> (AnyStr, List, int):
        spelling_mistakes = []
        corrected_word_list = []
        whitespace_list = []
        corrected_document = Doc(Vocab())
        try:
            self._add_symspell_checker(language)
            for token in document:
                self.check_token(token, language)
                whitespace_list.append(len(token.whitespace_) != 0)
                if token._.is_misspelled:
                    spelling_mistakes.append(token.text)
                    corrected_word_list.append(token._.correction)
                else:
                    corrected_word_list.append(token.text)
            corrected_document = Doc(vocab=document.vocab, words=corrected_word_list, spaces=whitespace_list)
        except ValueError as e:
            logging.warning("Spell checking error: {} for document: {}".format(e, document.text))
        return (corrected_document.text, spelling_mistakes, len(spelling_mistakes))

    def check_df(
        self,
        df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
        language: AnyStr = "language_column",
    ) -> pd.DataFrame:
        # Validate language inputs
        if language != "language_column":
            language_column = generate_unique(text_column, df.keys(), "language")
            df[language_column] = [language] * df.shape[0]
        # Tokenize DataFrame to obtain a new column with spaCy documents
        self.tokenizer.tokenize_df(df, text_column, language_column)
        # Apply check_document on all documents within the column
        message = "Spellchecking column '{}' in dataframe of {:d} rows".format(text_column, len(df.index))
        logging.info(message + "...")
        doc_lang_iterator = (
            (getattr(row, self.tokenizer.tokenized_column), getattr(row, language_column)) for row in df.itertuples()
        )
        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            output_tuple_list = list(executor.map(lambda x: self.check_document(*x), doc_lang_iterator))
        logging.info(message + ": Done!")
        # Format DataFrame: assign and reorder columns
        self.column_description_dict = OrderedDict()
        for k, v in self.COLUMN_DESCRIPTION_DICT.items():
            self.column_description_dict[generate_unique(k, df.keys(), text_column)] = v
        for i, column in enumerate(self.column_description_dict.keys()):
            df[column] = [t[i] for t in output_tuple_list]
            if i == 1:  # post-processing for spelling mistakes
                df[column] = df[column].apply(lambda x: "" if len(x) == 0 else x)
        del df[self.tokenizer.tokenized_column]
        move_columns_after(df, columns_to_move=list(self.column_description_dict.keys()), after_column=language_column)
        return df
