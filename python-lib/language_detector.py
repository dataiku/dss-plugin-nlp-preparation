# -*- coding: utf-8 -*-
import logging
from typing import List, AnyStr
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import pandas as pd
import cld3
from langid.langid import LanguageIdentifier, model

from language_dict import (
    SUPPORTED_LANGUAGES_PYCLD3,
    SUPPORTED_LANGUAGES_PYCLD3_NOT_LANGID,
    LANGUAGE_REMAPPING_PYCLD3_LANGID,
)

from plugin_io_utils import generate_unique, truncate_text_list


class LanguageDetector:
    """
    Language detection wrapper class on top of cld3 and langid, with additional features:
    - Route to cld3 for documents with more than 140 characters, else langid
    - Harmonize small differences between cld3 and langid language scopes
    - Add filter on language scope and minimum confidence score, else replace detection by fallback
    """

    LANGID_CLD3_NUM_CHAR_THRESHOLD = 140
    NUM_THREADS = 4
    COLUMN_DESCRIPTION_DICT = OrderedDict(
        [
            ("language_code", "Language code in ISO 639-1 format"),
            ("language_name", "Language name in ISO 639-1 format"),
            ("language_score", "Confidence score from 0 to 1"),
        ]
    )

    def __init__(
        self,
        language_scope: List = SUPPORTED_LANGUAGES_PYCLD3.keys(),
        minimum_score: float = 0.0,
        fallback_language: AnyStr = "",
    ):
        self.language_scope = language_scope
        self.minimum_score = float(minimum_score)
        self.fallback_language = fallback_language
        self.column_description_dict = self.COLUMN_DESCRIPTION_DICT  # may be changed by detect_languages_df
        self._langid_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self._langid_identifier.set_languages(
            [l for l in self.language_scope if l not in SUPPORTED_LANGUAGES_PYCLD3_NOT_LANGID]
        )

    def _langid_detection(self, doc: AnyStr) -> (AnyStr, float):
        language_detection_object = self._langid_identifier.classify(doc)
        lang_id = language_detection_object[0][:2]
        lang_probability = float(language_detection_object[1])
        return (lang_id, lang_probability)

    def _cld3_detection(self, doc: AnyStr) -> (AnyStr, float):
        language_detection_object = cld3.get_language(doc)
        lang_id = language_detection_object.language[:2]
        for original_code, new_code in LANGUAGE_REMAPPING_PYCLD3_LANGID.items():  # make cld3 compatible with langid
            lang_id = lang_id.replace(original_code, new_code)
        lang_probability = float(language_detection_object.probability)
        return (lang_id, lang_probability)

    def _detection_filter(self, doc: AnyStr, lang_id: AnyStr, lang_probability: float) -> (AnyStr, float):
        if lang_probability < self.minimum_score or lang_id not in self.language_scope:
            warning_msg = "Problem encountered for document: '{}'.\n".format(truncate_text_list([doc])[0])
            if lang_id not in self.language_scope:
                warning_msg += "Detected language: '{}' not within language scope: {}.\n".format(
                    lang_id, self.language_scope
                )
            else:
                warning_msg += "Confidence score: {:.2f} below minimum: {:.2f}.\n".format(
                    lang_probability, self.minimum_score
                )
            warning_msg += "Replacing detected language: '{}' by fallback: '{}'.".format(
                lang_id, self.fallback_language
            )
            logging.warning(warning_msg)
            lang_id, lang_probability = self.fallback_language, None
        return (lang_id, lang_probability)

    def detect_language_doc(self, doc: AnyStr) -> (AnyStr, AnyStr, float):
        # Route to langid or cld3 depending on number of characters
        if not doc:
            return ("", "", None)
        if len(doc) <= self.LANGID_CLD3_NUM_CHAR_THRESHOLD:
            lang_id, lang_probability = self._langid_detection(doc)
        else:
            lang_id, lang_probability = self._cld3_detection(doc)
        # Filters for language scope and minimum scores
        lang_id, lang_probability = self._detection_filter(doc, lang_id, lang_probability)
        # Enrich with language human name
        lang_name = SUPPORTED_LANGUAGES_PYCLD3.get(lang_id, "")
        # Round probability to 3 decimals
        lang_probability = round(lang_probability, 3) if lang_probability else None
        return (lang_id, lang_name, lang_probability)

    def detect_languages_df(self, df: pd.DataFrame, text_column: AnyStr) -> pd.DataFrame:
        self.column_description_dict = OrderedDict()
        for k, v in self.COLUMN_DESCRIPTION_DICT.items():
            self.column_description_dict[generate_unique(k, df.keys(), text_column)] = v
        doc_iterator = (doc for _, doc in df[text_column].astype(str).iteritems())
        output_df = df.copy()
        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            lang_output_tuple_list = list(executor.map(self.detect_language_doc, doc_iterator))
        for i, col in enumerate(self.column_description_dict):
            output_df[col] = [t[i] for t in lang_output_tuple_list]
        return output_df
