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
from language_dict import SUPPORTED_LANGUAGES_SYMSPELL


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


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


def load_plugin_config() -> Dict:
    """Utility function to load, resolve and validate all plugin config into a clean `params` dictionary

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
    logging.info("Custom vocabulary set: {}".format(params["custom_vocabulary_set"]))

    # custom_corrections (optional input dataset)
    params["custom_corrections"] = {}
    custom_corrections_input = get_input_names_for_role("custom_corrections")
    if len(custom_corrections_input) != 0:
        custom_corrections_dataset = dataiku.Dataset(custom_corrections_input[0])
        params["custom_corrections"] = custom_corrections_checker(custom_corrections_dataset)
    logging.info("Custom corrections: {}".format(params["custom_corrections"]))

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

    # path to the folder of dictionaries
    params["dictionary_folder_path"] = os.path.join(get_recipe_resource(), "dictionaries")

    # List of text columns
    params["text_column"] = recipe_config.get("text_column")
    logging.info("Text column: {}".format(params["text_column"]))
    if not params["text_column"]:
        raise PluginParamValidationError("Empty text column selection")

    # Language selection
    params["language"] = recipe_config.get("language")
    if params["language"] == "language_column":
        params["language_column"] = recipe_config.get("language_column")
        if params["language_column"] is None or params["language_column"] == "":
            raise PluginParamValidationError("Empty language column selection")
        logging.info("Language column: {}".format(params["language_column"]))
    else:
        if not params["language"]:
            raise PluginParamValidationError("Empty language selection")
        if params["language"] not in SUPPORTED_LANGUAGES_SYMSPELL:
            raise PluginParamValidationError("Unsupported language code: {}".format(params["language"]))
        params["language_column"] = ""
        logging.info("Language: {}".format(params["language"]))

    # Expert mode
    if recipe_config.get("expert"):
        logging.info("Expert mode is enabled")
    else:
        logging.info("Expert mode is disabled")

    # edit distance
    params["edit_distance"] = recipe_config.get("edit_distance")
    if params["edit_distance"] < 2 or params["edit_distance"] > 100:
        raise PluginParamValidationError("Edit distance must be between 2 and 100")
    logging.info("Maximum edit distance: {}".format(params["edit_distance"]))

    # ignore token
    if len(recipe_config.get("ignore_word_regex")) == 0:
        logging.info("No regular expression for words  not to be corrected")
        params["ignore_word_regex"] = None  # symspellpy wants None
    else:
        params["ignore_word_regex"] = recipe_config.get("ignore_word_regex")
        # Check for valid regex
        try:
            ignore_token_compiled = re.compile(params["ignore_word_regex"])
        except re.error as e:
            raise PluginParamValidationError("Ignore pattern parameter is not a valid regex: {}".format(e))
        params["ignore_word_regex"] = ignore_token_compiled.pattern
        logging.info("Regular expression for words not to be corrected: {}".format(params["ignore_word_regex"]))

    return params
