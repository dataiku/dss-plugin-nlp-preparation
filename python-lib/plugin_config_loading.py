# -*- coding: utf-8 -*-
"""Module with utility functions for loading, resolving and validating plugin parameters"""

import logging
import re
import os
from typing import Dict, Set

import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_resource,
)

from plugin_io_utils import clean_text_df
from language_dict import SUPPORTED_LANGUAGES_PYCLD3, SUPPORTED_LANGUAGES_SYMSPELL, SUPPORTED_LANGUAGES_SPACY
from spacy_tokenizer import MultilingualTokenizer
from text_cleaner import UnicodeNormalization


class PluginParamValidationError(ValueError):
    """Custom exception raised when the plugin parameters chosen by the user are invalid"""

    pass


def load_plugin_config_langdetect() -> Dict:
    """Utility function to validate and load language detection parameters into a clean dictionary

    Returns:
        Dictionary of parameter names (key) and values

    """
    params = {}
    # input dataset
    input_dataset_names = get_input_names_for_role("input_dataset")
    if len(input_dataset_names) == 0:
        raise PluginParamValidationError("Please specify input dataset")
    params["input_dataset"] = dataiku.Dataset(input_dataset_names[0])
    input_dataset_columns = [p["name"] for p in params["input_dataset"].read_schema()]

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise PluginParamValidationError("Please specify output dataset")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])

    # Recipe parameters
    recipe_config = get_recipe_config()
    # Text column
    params["text_column"] = recipe_config.get("text_column")
    if params["text_column"] not in input_dataset_columns:
        raise PluginParamValidationError(f"Invalid text column selection: {params['text_column']}")
    logging.info(f"Text column: {params['text_column']}")
    # Language scope
    params["language_scope"] = recipe_config.get("language_scope", [])
    if len(params["language_scope"]) == 0:
        params["language_scope"] = SUPPORTED_LANGUAGES_PYCLD3
    if len(params["language_scope"]) == 1:
        raise PluginParamValidationError(
            "Please add more than one item to the language scope or leave it empty to use all 114 languages"
        )
    logging.info(f"Scope of {len(params['language_scope'])} languages: {params['language_scope']}")
    # Minimum score
    params["minimum_score"] = float(recipe_config.get("minimum_score", 0))
    if params["minimum_score"] < 0 or params["minimum_score"] > 1:
        raise PluginParamValidationError("Minimum score must be between 0 and 1")
    logging.info(f"Minimum score for detection: {params['minimum_score']:.2f}")
    # Fallback language
    params["fallback_language"] = recipe_config.get("fallback_language")
    if not params["fallback_language"] or params["fallback_language"] == "None":
        logging.info("No fallback language")
        params["fallback_language"] = ""
    else:
        logging.info(f"Fallback language: {params['fallback_language']}")
    return params


def custom_vocabulary_checker(custom_vocabulary_dataset: dataiku.Dataset) -> Set:
    """Utility function to check the content of the optional custom vocabulary dataset

    Args:
        custom_vocabulary_dataset: Dataset with a single column for words that should not be corrected

    Returns:
        Set of words in the custom vocabulary

    """
    dataset_schema = custom_vocabulary_dataset.get_config()["schema"]
    columns = dataset_schema["columns"]
    if len(columns) != 1:
        raise PluginParamValidationError("Custom vocabulary dataset must have only one column")

    col_name = columns[0]["name"]
    col_type = columns[0]["type"]
    if col_type != "string":
        raise PluginParamValidationError("Column of custom vocabulary dataset must be of string type")

    df = clean_text_df(custom_vocabulary_dataset.get_dataframe(infer_with_pandas=False))
    custom_vocabulary = set(df[col_name].astype(str).tolist())
    return custom_vocabulary


def custom_corrections_checker(custom_corrections_dataset: dataiku.Dataset) -> Dict:
    """Utility function to check the content of the optional custom corrections dataset

    Args:
        custom_corrections_dataset: Dataset instance with the first column for words
            and the second one for their correction

    Returns:
        Dictionary of words (key) and their custom correction (value)

    """
    dataset_schema = custom_corrections_dataset.get_config()["schema"]
    columns = dataset_schema["columns"]
    if len(columns) != 2:
        raise PluginParamValidationError("Custom corrections dataset must have only two columns")

    (word_column, correction_column) = (columns[0], columns[1])
    if word_column["type"] != "string" or correction_column["type"] != "string":
        raise PluginParamValidationError("Columns of custom corrections dataset must be of string type")

    df = custom_corrections_dataset.get_dataframe(infer_with_pandas=False)
    df = clean_text_df(df, dropna_columns=[word_column["name"]]).fillna("").astype(str)
    custom_corrections_dict = {row[0]: row[1] for row in df.itertuples(index=False)}
    return custom_corrections_dict


def load_plugin_config_spellchecker() -> Dict:
    """Utility function to validate and load spell checker parameters into a clean dictionary

    Returns:
        Dictionary of parameter names (key) and values

    """
    params = {}
    recipe_config = get_recipe_config()

    # input dataset
    input_dataset_names = get_input_names_for_role("input_dataset")
    if len(input_dataset_names) == 0:
        raise PluginParamValidationError("Please specify input dataset")
    params["input_dataset"] = dataiku.Dataset(input_dataset_names[0])
    input_dataset_columns = [p["name"] for p in params["input_dataset"].read_schema()]

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise PluginParamValidationError("Please specify output dataset")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])

    # custom_vocabulary (optional input dataset)
    params["custom_vocabulary_set"] = set()
    custom_vocabulary_input = get_input_names_for_role("custom_vocabulary")
    if len(custom_vocabulary_input) != 0:
        custom_vocabulary_dataset = dataiku.Dataset(custom_vocabulary_input[0])
        params["custom_vocabulary_set"] = custom_vocabulary_checker(custom_vocabulary_dataset)
    logging.info(f"Custom vocabulary set: {params['custom_vocabulary_set']}")

    # custom_corrections (optional input dataset)
    params["custom_corrections"] = {}
    custom_corrections_input = get_input_names_for_role("custom_corrections")
    if len(custom_corrections_input) != 0:
        custom_corrections_dataset = dataiku.Dataset(custom_corrections_input[0])
        params["custom_corrections"] = custom_corrections_checker(custom_corrections_dataset)
    logging.info(f"Custom corrections: {params['custom_corrections']}")

    # diagnosis dataset (optional output dataset)
    diagnosis_dataset_names = get_output_names_for_role("diagnosis_dataset")
    params["diagnosis_dataset"] = None
    params["compute_diagnosis"] = False
    if len(diagnosis_dataset_names) != 0:
        logging.info("Spellchecker diagnosis will be computed")
        params["compute_diagnosis"] = True
        params["diagnosis_dataset"] = dataiku.Dataset(diagnosis_dataset_names[0])
    else:
        logging.info("Spellchecker diagnosis will not be computed")

    # path to the folder of stopwords
    params["stopwords_folder_path"] = os.path.join(get_recipe_resource(), "stopwords")

    # path to the folder of dictionaries
    params["dictionary_folder_path"] = os.path.join(get_recipe_resource(), "dictionaries")

    # Text column selection
    params["text_column"] = recipe_config.get("text_column")
    logging.info(f"Text column: {params['text_column']}")
    if params["text_column"] not in input_dataset_columns:
        raise PluginParamValidationError(f"Invalid text column selection: {params['text_column']}")

    # Language selection
    params["language"] = recipe_config.get("language")
    if params["language"] == "language_column":
        params["language_column"] = recipe_config.get("language_column")
        if params["language_column"] not in input_dataset_columns:
            raise PluginParamValidationError(f"Invalid language column selection: : {params['language_column']}")
        logging.info(f"Language column: {params['language_column']}")
    else:
        if not params["language"]:
            raise PluginParamValidationError("Empty language selection")
        if params["language"] not in SUPPORTED_LANGUAGES_SYMSPELL:
            raise PluginParamValidationError(f"Unsupported language code: {params['language']}")
        params["language_column"] = ""
        logging.info(f"Language: {params['language']}")

    # Expert mode
    if recipe_config.get("expert"):
        logging.info("Expert mode is enabled")
    else:
        logging.info("Expert mode is disabled")

    # edit distance
    params["edit_distance"] = recipe_config.get("edit_distance")
    if params["edit_distance"] < 2 or params["edit_distance"] > 100:
        raise PluginParamValidationError("Edit distance must be between 2 and 100")
    logging.info(f"Maximum edit distance: {params['edit_distance']}")

    # ignore token
    if len(recipe_config.get("ignore_word_regex")) == 0:
        logging.info("No regular expression for words not to be corrected")
        params["ignore_word_regex"] = None  # symspellpy wants None
    else:
        params["ignore_word_regex"] = recipe_config.get("ignore_word_regex")
        # Check for valid regex
        try:
            ignore_token_compiled = re.compile(params["ignore_word_regex"])
        except re.error as e:
            raise PluginParamValidationError(f"Ignore pattern parameter is not a valid regex: {e}")
        params["ignore_word_regex"] = ignore_token_compiled.pattern
        logging.info(f"Regular expression for words not to be corrected: {params['ignore_word_regex']}")

    return params


def load_plugin_config_cleaning() -> Dict:
    """Utility function to validate and load text cleaning parameters into a clean dictionary

    Returns:
        Dictionary of parameter names (key) and values

    """
    params = {}
    recipe_config = get_recipe_config()

    # input dataset
    input_dataset_names = get_input_names_for_role("input_dataset")
    if len(input_dataset_names) == 0:
        raise PluginParamValidationError("Please specify input dataset")
    params["input_dataset"] = dataiku.Dataset(input_dataset_names[0])
    input_dataset_columns = [p["name"] for p in params["input_dataset"].read_schema()]

    # output dataset
    output_dataset_names = get_output_names_for_role("output_dataset")
    if len(output_dataset_names) == 0:
        raise PluginParamValidationError("Please specify output dataset")
    params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])

    # path to the folder of stopwords
    params["stopwords_folder_path"] = os.path.join(get_recipe_resource(), "stopwords")

    # Text column selection
    params["text_column"] = recipe_config.get("text_column")
    logging.info(f"Text column: {params['text_column']}")
    if params["text_column"] not in input_dataset_columns:
        raise PluginParamValidationError(f"Invalid text column selection: {params['text_column']}")

    # Language selection
    params["language"] = recipe_config.get("language")
    if params["language"] == "language_column":
        params["language_column"] = recipe_config.get("language_column")
        if params["language_column"] not in input_dataset_columns:
            raise PluginParamValidationError(f"Invalid language column selection: {params['language_column']}")
        logging.info(f"Language column: {params['language_column']}")
    else:
        if not params["language"]:
            raise PluginParamValidationError("Empty language selection")
        if params["language"] not in SUPPORTED_LANGUAGES_SPACY:
            raise PluginParamValidationError(f"Unsupported language code: {params['language']}")
        params["language_column"] = ""
        logging.info(f"Language: {params['language']}")

    # Cleaning parameters
    params["token_filters"] = set(recipe_config.get("token_filters", []))
    available_token_filters = set(MultilingualTokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES.keys())
    if not params["token_filters"] <= available_token_filters:
        raise PluginParamValidationError(f"Invalid token filters: {params['token_filters']-available_token_filters}")
    logging.info(f"Token filters: {params['token_filters']}")
    params["lemmatization"] = bool(recipe_config.get("lemmatization"))
    logging.info(f"Lemmatization: {params['lemmatization']}")
    params["lowercase"] = bool(recipe_config.get("lowercase"))
    logging.info(f"Lowercase: {params['lowercase']}")

    # Expert mode
    if recipe_config.get("expert"):
        logging.info("Expert mode is enabled")
    else:
        logging.info("Expert mode is disabled")
    params["unicode_normalization"] = UnicodeNormalization[recipe_config.get("unicode_normalization")]
    logging.info(f"Unicode normalization: {params['unicode_normalization']}")

    params["keep_filtered_tokens"] = bool(recipe_config.get("keep_filtered_tokens"))
    logging.info(f"Keep filtered tokens: {params['keep_filtered_tokens']}")

    return params
