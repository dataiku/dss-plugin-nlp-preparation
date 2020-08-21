# -*- coding: utf-8 -*-
"""Module with a class to check and correct misspellings in multiple languages"""

import logging
from typing import List, AnyStr, Set, Tuple, Dict, Pattern
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, Counter
from time import time
from functools import lru_cache
from threading import Lock

import pandas as pd
from spacy.tokens import Token, Doc
from spacy.vocab import Vocab
from symspellpy.symspellpy import SymSpell, Verbosity

from plugin_io_utils import unique_list, generate_unique, truncate_text_list, clean_empty_list
from spacy_tokenizer import MultilingualTokenizer
from language_dict import SUPPORTED_LANGUAGES_SYMSPELL

# Setting custom spaCy token extensions to store spellchecking information
Token.set_extension("is_misspelled", default=False, force=True)
Token.set_extension("correction", default="", force=True)


class SpellChecker:
    """Wrapper class to check spelling with SymSpellPy

    Relies on spaCy for tokenization of text data before calling the spellchecker

    Attributes:
        dictionary_folder_path (str): Local path to a folder containing SymSpell dictionary files
        custom_vocabulary_set (set): Set of words that should not be corrected
        custom_corrections (dict): Dictionary of words (key) and their custom correction (value)
        edit_distance (int): Maximum edit distance between a word and its correction
        ignore_token (Pattern): Regular expression for words not to be corrected
        transfer_casing (bool): Transfer input word case to the corrected word
        compute_diagnosis (bool): Compute spellchecker diagnosis of each word
    """

    DEFAULT_EDIT_DISTANCE = 2
    SUGGESTION_VERBOSITY = Verbosity.TOP  # returns only the closest word
    DEFAULT_NUM_THREADS = 4
    OUTPUT_COLUMN_DESCRIPTION_DICT = OrderedDict(
        [
            ("corrected", "Corrected text"),
            ("misspelling_list", "List of misspellings"),
            ("misspelling_count", "Number of misspellings"),
        ]
    )
    DIAGNOSIS_COLUMN_DESCRIPTION_DICT = OrderedDict(
        [
            ("language", "Language code in ISO 639-1 format"),
            ("original_word", "Original word in the input dataset"),
            ("is_misspelled", "Word detected as misspelling"),
            ("corrected_word", "Correction in case of misspelling"),
            ("spellcheck_diagnosis", "Diagnosis of the spellchecker"),
            ("word_count", "Word count in the input dataset"),
        ]
    )

    def __init__(
        self,
        dictionary_folder_path: AnyStr,
        custom_vocabulary_set: Set[AnyStr] = set(),
        custom_corrections: Dict = {},
        edit_distance: int = DEFAULT_EDIT_DISTANCE,
        ignore_token: Pattern = None,
        transfer_casing: bool = True,
        compute_diagnosis: bool = True,
    ):
        """Initialization method for the SpellChecker class, with optional arguments

        Args:
            dictionary_folder_path: Local path to a folder containing SymSpell dictionary files
                Each dictionary file in the folder should be named "xx.txt"
                where xx is the language code in ISO 639-1 format
            custom_vocabulary_set: Optional - Set of words that should not be corrected
            custom_corrections: Optional - Dictionary of words (key) and their custom correction (value)
            edit_distance: Maximum edit distance between a word and its correction.
                Default is 2, which is SymSpell recommendation for reasonable speed and quality
            ignore_token: Regular expression for words not to be corrected
                Should be a compiled regex object, use re.compile beforehand
            transfer_casing (bool): If True, transfer input word case to the corrected word
                Default is True, which works well for European languages
            compute_diagnosis (bool): If True, compute spellchecker diagnosis of each word
                Adds ~20% processing time but allows to understand what the spellchecker did
        """
        self._tokenizer = MultilingualTokenizer()
        self.dictionary_folder_path = dictionary_folder_path
        self.custom_vocabulary_set = custom_vocabulary_set
        self.custom_corrections = custom_corrections
        self.edit_distance = int(edit_distance)
        self.ignore_token = ignore_token
        self.transfer_casing = transfer_casing
        self._symspell_checker_dict = {}
        self._output_column_description_dict = self.OUTPUT_COLUMN_DESCRIPTION_DICT  # may be changed by check_df
        self.compute_diagnosis = compute_diagnosis
        if self.compute_diagnosis:
            self._diagnosis_lock = Lock()
            self._token_dict = {k: Counter() for k in SUPPORTED_LANGUAGES_SYMSPELL}  # may be changed by check_token
            self._diagnosis_list = []  # may be changed by check_token

    def _create_symspell_checker(self, language: AnyStr) -> SymSpell:
        """Private method to create a SymSpell instance for a given language

        Args:
            language: Language code in ISO 639-1 format

        Returns:
            SymSpell checker instance loaded with the language dictionary
        """
        start = time()
        logging.info("Loading spellchecker for language '{}'...".format(language))
        symspell_checker = SymSpell(max_dictionary_edit_distance=self.edit_distance)
        frequency_dict_path = self.dictionary_folder_path + "/" + language + ".txt"
        symspell_checker.load_dictionary(frequency_dict_path, term_index=0, count_index=1, encoding="utf-8")
        if len(self.custom_vocabulary_set) != 0:
            for word in self.custom_vocabulary_set:
                symspell_checker.create_dictionary_entry(key=word, count=1)
        logging.info("Loading spellchecker for language '{}': Done in {:.2f} seconds.".format(language, time() - start))
        return symspell_checker

    def _add_symspell_checker(self, language: AnyStr) -> bool:
        """Private method to add a SymSpell checker for a given language

        The SymSpell checker is added to the `_symspell_checker_dict` private dictionary attribute,
        if the language code is valid and recognized among the list of supported languages
        (`SUPPORTED_LANGUAGES_SYMSPELL` constant), else it will raise a ValueError exception.

        Args:
            language: Language code in ISO 639-1 format

        Returns:
            True if the SymSpell spellchecker was added, else False

        Raises:
            ValueError: If the language code is missing or not in SUPPORTED_LANGUAGES_SYMSPELL
        """
        added_checker = False
        if pd.isnull(language) or language == "":
            raise ValueError("Missing language code")
        if language not in SUPPORTED_LANGUAGES_SYMSPELL:
            raise ValueError("Unsupported language code: {}".format(language))
        if language not in self._symspell_checker_dict:
            self._symspell_checker_dict[language] = self._create_symspell_checker(language=language)
            added_checker = True
        return added_checker

    @lru_cache(maxsize=1024)  # Memory cache to avoid checking a word which has been checked before
    def symspell_check_word(self, word: AnyStr, language: AnyStr) -> Tuple[bool, AnyStr, AnyStr]:
        """Public method to check the spelling of a word for a given language using SymSpell

        Args:
            word: String to feed to the spellchecker
            language: Language code in ISO 639-1 format

        Returns:
            Tuple of 3 elements:
                1. Boolean if the word is misspelled
                2. Corrected word if the word is misspelled and a correction if found,
                    else keep the original word
                3. Spellchecker diagnosis string explaining the spellchecker action
        """
        (is_misspelled, correction, diagnosis) = (False, word, "")
        correction_suggestions = self._symspell_checker_dict[language].lookup(
            word,
            verbosity=self.SUGGESTION_VERBOSITY,
            max_edit_distance=self.edit_distance,
            ignore_token=self.ignore_token,
            transfer_casing=self.transfer_casing,
        )
        if len(correction_suggestions) != 0:
            correction_suggestion = correction_suggestions[0].term
            if correction_suggestion.lower() != word.lower():
                diagnosis = "NOK - Corrected by spellchecker"
                (is_misspelled, correction) = (True, correction_suggestion)
            else:
                diagnosis = "OK - Approved by spellchecker"
        else:
            diagnosis = "WARN - No correction found, keeping as-is"
            (is_misspelled, correction) = (True, word)
        return (is_misspelled, correction, diagnosis)

    def check_token(self, token: Token, language: AnyStr) -> Tuple[bool, AnyStr, AnyStr]:
        """Public method to check the spelling of a spaCy token for a given language

        Applies pre-processing checks before checking with SymSpell:
            Checks if the token is in custom_correction or custom_vocabulary_set
            Checks if the token has any attributes indicating that it shouldn't be corrected
            (see spacy_tokenizer.MultilingualTokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES)
        If the checks are passed, feed the token to `symspell_check_word`

        Args:
            token: SpaCy token to feed to the spellchecker
            language: Language code in ISO 639-1 format

        Returns:
            Tuple of 3 elements:
                1. Boolean if the word is misspelled
                2. Corrected word if the word is misspelled and a correction if found,
                    else keep the original word
                3. Spellchecker diagnosis string explaining the spellchecker action
        """
        (is_misspelled, correction, diagnosis) = (False, token.text, "")
        if token.text in self.custom_corrections.keys():  # special case of custom corrections
            diagnosis = "NOK - Corrected by custom correction"
            (is_misspelled, correction) = (True, str(self.custom_corrections[token.text]))
        else:
            if token.text in self.custom_vocabulary_set:
                diagnosis = "OK - In custom vocabulary"
            else:
                token_attributes = [
                    t
                    for t in self._tokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES
                    if getattr(token, t, False) or getattr(token._, t, False)
                ]
                if len(token_attributes) == 0:
                    symspell_check = self.symspell_check_word(token.text, language)
                    (is_misspelled, correction, diagnosis) = (
                        symspell_check[0],
                        symspell_check[1],
                        symspell_check[2],
                    )
                else:
                    diagnosis = "OK - Detected as '{}', keeping as-is".format(token_attributes[0].upper())
        if self.compute_diagnosis:
            diagnosis_tuple = (language, token.text, is_misspelled, correction, diagnosis)
            self._add_to_diagnosis(token, language, diagnosis_tuple)
        return (is_misspelled, correction, diagnosis)

    def check_document(self, document: Doc, language: AnyStr) -> Tuple[AnyStr, List, int]:
        """Public method to check the spelling of a spaCy document

        Feed document to `check_token`, token-by-token (yum!)
        This method calls `_add_symspell_checker` in case the requested language has not already been added.
        In case of an error, the output will be empty.

        Args:
            document: SpaCy document to feed to the spellchecker
            language: Language code in ISO 639-1 format

        Returns:
            Tuple of 3 elements:
                1. Corrected spaCy document
                2. List of misspellings as strings
                3. Number of misspellings
        """
        (spelling_mistakes, corrected_word_list, whitespace_list) = ([], [], [])
        corrected_document = Doc(Vocab())
        try:
            self._add_symspell_checker(language)
            for token in document:
                check_token = self.check_token(token, language)
                token._.is_misspelled = check_token[0]
                token._.correction = check_token[1]
                if token._.is_misspelled:
                    spelling_mistakes.append(token.text)
                if token._.correction != "":
                    whitespace_list.append(len(token.whitespace_) != 0)
                    corrected_word_list.append(token._.correction)
            corrected_document = Doc(vocab=document.vocab, words=corrected_word_list, spaces=whitespace_list)
        except ValueError as e:
            logging.warning(
                "Spellchecking error: '{}' for document: '{}', output columns will be empty".format(
                    e, truncate_text_list([document.text])
                )
            )
        spelling_mistakes = unique_list(spelling_mistakes)
        return (corrected_document.text, spelling_mistakes, len(spelling_mistakes))

    def check_document_list(self, document_list: List[Doc], language: AnyStr) -> List[Tuple[AnyStr, List, int]]:
        """Public method to check the spelling of a list of documents for a given language

        Feed document to `check_document`, document-by-document (yum!)
        This method calls `_add_symspell_checker` in case the requested language has not already been added.
        In case of an error, the output will be empty.

        Args:
            document_list: List of spaCy documents
            language: Language code in ISO 639-1 format

        Returns:
            List of tuples with 3 elements:
                1. Corrected spaCy document
                2. List of misspellings as strings
                3. Number of misspellings
        """
        start = time()
        logging.info("Spellchecking {:d} documents in language '{}'...".format(len(document_list), language))
        tuple_list = [("", [], 0)] * len(document_list)
        try:
            self._add_symspell_checker(language)
            doc_lang_iterator = ((doc, language) for doc in document_list)
            with ThreadPoolExecutor(max_workers=self.DEFAULT_NUM_THREADS) as executor:
                tuple_list = list(executor.map(lambda x: self.check_document(*x), doc_lang_iterator))
            logging.info(
                "Spellchecking {:d} documents in language '{}': Done in {:.2f} seconds.".format(
                    len(tuple_list), language, time() - start
                )
            )
        except ValueError as e:
            logging.warning(
                "Spellchecking error: '{}' for documents: '{}', output columns will be empty".format(
                    e, truncate_text_list([d.text for d in document_list])
                )
            )
        return tuple_list

    def _prepare_df_for_spellchecker(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr, language: AnyStr
    ) -> None:
        """Private method to prepare a Pandas dataframe in-place before feeding it to the spellchecker

        Tokenizes the content of the text column into a new column containing spaCy documents
        Adds new columns to hold the future outputs of the spellchecker

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format
            language: Language code in ISO 639-1 format
                If equal to "language_column" this parameter is ignored in favor of language_column
        """
        self._output_column_description_dict = OrderedDict()
        for k, v in self.OUTPUT_COLUMN_DESCRIPTION_DICT.items():
            column_name = generate_unique(k, df.keys(), text_column)
            df[column_name] = pd.Series([""] * len(df.index))
            self._output_column_description_dict[column_name] = v
        self._tokenizer.tokenize_df(df, text_column, language_column, language)

    def _format_output_df(self, df: pd.DataFrame) -> None:
        """Private method to format the output dataframe after spellchecking

        Removes the tokenized column with spaCy documents
        Replaces empty lists of misspellings by an empty string

        Args:
            df: Input pandas DataFrame
        """
        del df[self._tokenizer.tokenized_column]
        corrected_text_column = list(self._output_column_description_dict.keys())[0]
        spelling_mistakes_column = list(self._output_column_description_dict.keys())[1]
        misspelling_count_column = list(self._output_column_description_dict.keys())[2]
        df[spelling_mistakes_column] = df[spelling_mistakes_column].apply(clean_empty_list)
        df.loc[df[corrected_text_column] == "", misspelling_count_column] = ""

    def check_df(
        self, df: pd.DataFrame, text_column: AnyStr, language_column: AnyStr = "", language: AnyStr = "language_column",
    ) -> pd.DataFrame:
        """Public method to check the spelling of a text column in a pandas DataFrame, given a language column

        Prepares the dataframe with `_prepare_df_for_spellchecker`
        Runs `check_document_list` for each language
        Formats the output dataframe

        Args:
            df: Input pandas DataFrame
            text_column: Name of the column containing text data
            language_column: Name of the column with language codes in ISO 639-1 format
            language: Language code in ISO 639-1 format
                If equal to "language_column" this parameter is ignored in favor of language_column

        Returns:
            Input dataframe with 3 new columns at the end:
                1. Corrected text
                2. List of misspellings
                3. Number of misspellings
        """
        self._prepare_df_for_spellchecker(df, text_column, language_column, language)
        if language == "language_column":
            languages = df[language_column].dropna().unique()
            for lang in languages:  # iterate over languages
                language_indices = df[language_column] == lang
                document_slice = df.loc[language_indices, self._tokenizer.tokenized_column]  # slicing df by language
                if len(document_slice) != 0:
                    tuple_list = self.check_document_list(document_list=document_slice, language=lang)
                    for i, column in enumerate(self._output_column_description_dict.keys()):
                        df.loc[language_indices, column] = pd.Series(
                            [t[i] for t in tuple_list], index=document_slice.index
                        )
        else:
            tuple_list = self.check_document_list(document_list=df[self._tokenizer.tokenized_column], language=language)
            for i, column in enumerate(self._output_column_description_dict.keys()):
                df[column] = [t[i] for t in tuple_list]
        self._format_output_df(df)
        return df

    def _add_to_diagnosis(self, token: Token, language: AnyStr, diagnosis_tuple) -> None:
        """Private method to add diagnosis information on a token, in a thread-safe way

        If the compute_diagnosis attribute is True, this function is ran whenever the `check_token` method is called
        Writes to the private _token_dict and _diagnosis_list attributes

        Args:
            token: spaCy token
            language: Language code in ISO 639-1 format
            diagnosis_tuple: Tuple of diagnosis information
                Should be ordered as DIAGNOSIS_COLUMN_DESCRIPTION_DICT except the word_count column
        """
        with self._diagnosis_lock:
            if token.text not in self._token_dict.get(language, {}):
                self._diagnosis_list.append(diagnosis_tuple)
            if language in SUPPORTED_LANGUAGES_SYMSPELL:
                self._token_dict[language][token.text] += 1

    def create_diagnosis_df(self) -> pd.DataFrame:
        """Public method to diagnose the spellchecker actions after running `check_df`

        Formats the private _token_dict and _diagnosis_list attributes into a human-readable dataframe

        Returns:
            Diagnosis dataframe with columns described in DIAGNOSIS_COLUMN_DESCRIPTION_DICT
        """
        df = pd.DataFrame()
        logging.info("Computing spellchecker diagnosis...")
        for i, column in enumerate(self.DIAGNOSIS_COLUMN_DESCRIPTION_DICT.keys()):
            if column != "word_count":
                df[column] = [t[i] for t in self._diagnosis_list]
        # Retrieve word_count information
        df["word_count"] = [""] * len(df.index)
        languages = df["language"].dropna().unique()
        for lang in languages:  # iterate over languages
            language_indices = df["language"] == lang
            df_slice = df.loc[language_indices, "original_word"]
            df.loc[language_indices, "word_count"] = df_slice.apply(lambda x: self._token_dict[lang].get(x, 0))
        # Cleaning and sorting output dataframe
        df.loc[~df["is_misspelled"], "corrected_word"] = ""
        df = df.sort_values(by=["is_misspelled", "word_count"], ascending=False)
        logging.info("Computing spellchecker diagnosis: Done!")
        return df
