# -*- coding: utf-8 -*-
"""Module with a class to check and correct misspellings in multiple languages"""

import logging
import itertools
from string import punctuation
from typing import List, AnyStr, Set, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from time import time

import pandas as pd
from spacy.tokens import Token, Doc
from spacy.vocab import Vocab
from symspellpy.symspellpy import SymSpell, Verbosity

from plugin_io_utils import unique_list, generate_unique, move_columns_after, truncate_text_list
from spacy_tokenizer import MultilingualTokenizer
from language_dict import SUPPORTED_LANGUAGES_SYMSPELL

# Setting custom spaCy token extensions to store spellchecking information
Token.set_extension("is_misspelled", default=False, force=True)
Token.set_extension("correction", default="", force=True)
Token.set_extension("original_word", default="", force=True)


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
            ("spelling_mistakes", "List of spelling mistakes"),
            ("misspelling_count", "Number of spelling mistakes"),
        ]
    )

    def __init__(
        self,
        tokenizer: MultilingualTokenizer,
        dictionary_folder_path: AnyStr,
        custom_vocabulary_set: Set[AnyStr] = set(),
        custom_corrections: Dict = {},
        edit_distance: int = DEFAULT_EDIT_DISTANCE,
        ignore_token: AnyStr = None,
        transfer_casing: bool = True,
    ):
        self.tokenizer = tokenizer
        self.dictionary_folder_path = dictionary_folder_path
        self.custom_vocabulary_set = custom_vocabulary_set
        self.custom_corrections = custom_corrections
        self.edit_distance = edit_distance
        self.ignore_token = ignore_token
        self.transfer_casing = transfer_casing
        self.symspell_checker_dict = {}
        self.column_description_dict = self.COLUMN_DESCRIPTION_DICT  # may be changed by check_df

    def _create_symspell_checker(self, language: AnyStr, edit_distance: int) -> SymSpell:
        start = time()
        logging.info("Loading spellchecker for language '{}'...".format(language))
        symspell_checker = SymSpell(max_dictionary_edit_distance=edit_distance)
        frequency_dict_path = self.dictionary_folder_path + "/" + language + ".txt"
        symspell_checker.load_dictionary(frequency_dict_path, term_index=0, count_index=1, encoding="utf-8")
        if len(self.custom_vocabulary_set) != 0:
            for word in self.custom_vocabulary_set:
                symspell_checker.create_dictionary_entry(key=word, count=1)
        logging.info("Loading spellchecker for language '{}': Done in {:.2f} seconds.".format(language, time() - start))
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
        text = token.text
        if text != "" and text[:1] in punctuation:  # special case with punctation prefixes
            text = text[1:]
        if text in self.custom_corrections.keys():  # special case of custom corrections
            token._.is_misspelled = True
            token._.correction = str(self.custom_corrections[text])
        else:
            match_token_attributes = [
                getattr(token, t, False) or getattr(token._, t, False)
                for t in self.tokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES
            ]
            if not any(match_token_attributes) and text not in self.custom_vocabulary_set:
                correction_suggestions = self.symspell_checker_dict[language].lookup(
                    text,
                    verbosity=self.SUGGESTION_VERBOSITY,
                    max_edit_distance=self.edit_distance,
                    ignore_token=self.ignore_token,
                    transfer_casing=self.transfer_casing,
                )
                if len(correction_suggestions) != 0:
                    correction = correction_suggestions[0].term
                    if correction.lower() != text.lower():
                        token._.is_misspelled = True
                        token._.correction = correction
                else:
                    token._.is_misspelled = True
                    token._.correction = token.text

    def check_document(self, document: Doc, language: AnyStr) -> Tuple[AnyStr, List, int, List]:
        (spelling_mistakes, corrected_word_list, whitespace_list, uncorrected_list) = ([], [], [], [])
        corrected_document = Doc(Vocab())
        try:
            self._add_symspell_checker(language)
            for token in document:
                self.check_token(token, language)
                whitespace_list.append(len(token.whitespace_) != 0)
                if token._.is_misspelled:
                    spelling_mistakes.append(token.text)
                    corrected_word_list.append(token._.correction)
                    if token._.correction == token.text:
                        uncorrected_list.append(token.text)
                else:
                    corrected_word_list.append(token.text)
            corrected_document = Doc(vocab=document.vocab, words=corrected_word_list, spaces=whitespace_list)
        except ValueError as e:
            logging.warning(
                "Spellchecking error: {} for document: {}, output columns will be empty".format(
                    e, truncate_text_list([document.text])
                )
            )
        spelling_mistakes = unique_list(spelling_mistakes)
        return (corrected_document.text, spelling_mistakes, len(spelling_mistakes), uncorrected_list)

    def check_document_list(self, document_list: List[Doc], language: AnyStr) -> List[Tuple[AnyStr, List, int, List]]:
        start = time()
        logging.info("Spellchecking {:d} documents in language '{}'...".format(len(document_list), language))
        tuple_list = [("", [], 0, [])] * len(document_list)
        try:
            self._add_symspell_checker(language)
            doc_lang_iterator = ((doc, language) for doc in document_list)
            with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
                tuple_list = list(executor.map(lambda x: self.check_document(*x), doc_lang_iterator))
            logging.info(
                "Spellchecking {:d} documents in language '{}': Done in {:.2f} seconds.".format(
                    len(tuple_list), language, time() - start
                )
            )
            uncorrected_list = unique_list(list(itertools.chain.from_iterable([t[3] for t in tuple_list])))
            if len(uncorrected_list) != 0:
                logging.warning(
                    "No corrections found for '{}' in language '{}', keeping as-is".format(uncorrected_list, language)
                )
        except ValueError as e:
            logging.warning(
                "Spellchecking error: {} for documents: {}, output columns will be empty".format(
                    e, truncate_text_list([d.text for d in document_list])
                )
            )
        return tuple_list

    def _prepare_df_for_spellchecking(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr
    ):
        self.column_description_dict = OrderedDict()
        for k, v in self.COLUMN_DESCRIPTION_DICT.items():
            column_name = generate_unique(k, df.keys(), text_column)
            df[column_name] = pd.Series([""] * len(df.index))
            self.column_description_dict[column_name] = v
        self.tokenizer.tokenize_df(df, text_column, language_column, language)

    def _format_output_df(self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr):
        del df[self.tokenizer.tokenized_column]
        corrected_text_column = list(self.column_description_dict.keys())[0]
        spelling_mistakes_column = list(self.column_description_dict.keys())[1]
        misspelling_count_column = list(self.column_description_dict.keys())[2]
        df[spelling_mistakes_column] = df[spelling_mistakes_column].apply(lambda x: "" if len(x) == 0 else x)
        df.loc[df[corrected_text_column] == "", misspelling_count_column] = ""
        move_columns_after(df, columns_to_move=list(self.column_description_dict.keys()), after_column=text_column)

    def check_df(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr = "language_column",
    ) -> pd.DataFrame:
        self._prepare_df_for_spellchecking(df, text_column, language_column, language)
        if language == "language_column":
            languages = df[language_column].dropna().unique()
            for lang in languages:  # iterate over languages
                language_indices = df[language_column] == lang
                document_slice = df.loc[language_indices, self.tokenizer.tokenized_column]  # slicing df by language
                if len(document_slice) != 0:
                    tuple_list = self.check_document_list(document_list=document_slice, language=lang)
                    for i, column in enumerate(self.column_description_dict.keys()):
                        df.loc[language_indices, column] = pd.Series(
                            [t[i] for t in tuple_list], index=document_slice.index
                        )
        else:
            tuple_list = self.check_document_list(document_list=df[self.tokenizer.tokenized_column], language=language)
            for i, column in enumerate(self.column_description_dict.keys()):
                df[column] = [t[i] for t in tuple_list]
        # Format output DataFrame
        self._format_output_df(df, text_column, language_column, language)
        return df
