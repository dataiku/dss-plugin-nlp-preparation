# -*- coding: utf-8 -*-
import logging
from typing import Dict, Set
import dataiku
import re


def custom_vocabulary_checker(custom_vocabulary_dataset: dataiku.Dataset) -> Set:

    dataset_schema = custom_vocabulary_dataset.get_config()["schema"]
    columns = dataset_schema["columns"]

    assert len(columns) == 1, "Custom vocabulary dataset must have only one column"

    col_name = columns[0]["name"]
    col_type = columns[0]["type"]

    assert col_type == "string", "Column of custom vocabulary dataset must be of type string"

    custom_vocabulary_set = set(custom_vocabulary_dataset.get_dataframe()[col_name].str.lower().tolist())

    return custom_vocabulary_set


def load_plugin_config(recipe_config: Dict) -> Dict:
    """
    Helper function to load plugin recipe config into a clean parameter dictionary.
    Applies assertion checks for correct input config.
    """
    params = {}

    # path to the folder of dictionaries
    params["folder_of_dictionaries"] = dataiku.customrecipe.get_recipe_resource()

    # List of text columns
    params["text_column"] = recipe_config.get("text_column")

    logging.info("Text column: {}".format(params["text_column"]))
    assert params["text_column"] != "", "Empty text column selection"

    # custom_vocabulary
    params["custom_vocabulary_set"] = set()
    custom_vocabulary_input = dataiku.customrecipe.get_input_names_for_role("custom_vocabulary")
    if len(custom_vocabulary_input) == 1:
        custom_vocabulary_dataset = dataiku.Dataset(custom_vocabulary_input[0])
        params["custom_vocabulary_set"] = custom_vocabulary_checker(custom_vocabulary_dataset)
    logging.info("Custom vocabulary set: {}".format(params["custom_vocabulary_set"]))

    # Language selection
    params["language_selection"] = recipe_config.get("language_selection")

    if params["language_selection"] == "from_list":
        params["language"] = recipe_config.get("language_from_list")
        assert params["language"] is not None and params["language"] != "", "Empty language selection"
        logging.info("Language: {}".format(params["language"]))
        params["language_column"] = ""

    if params["language_selection"] == "from_column":
        params["language_column"] = recipe_config.get("language_from_column")
        params["language"] = ""
        assert (
            params["language_column"] is not None and params["language_column"] != ""
        ), "Empty language column selection"
        logging.info("Language codes from column {}".format(params["language_column"]))

    # Expert mode
    if recipe_config.get("expert"):
        logging.info("Expert mode is enabled")
    else:
        logging.info("Expert mode is disabled")

    # distance
    params["distance"] = recipe_config.get("distance")
    logging.info("Maximum edit distance {}".format(params["distance"]))

    # ignore token
    if len(recipe_config.get("ignore_token")) == 0:
        logging.info("No token to be ignored")
        params["ignore_token"] = None  # symspellpy wants None
    else:
        params["ignore_token"] = recipe_config.get("ignore_token")
        # Check for valid regex
        try:
            ignore_token_compiled = re.compile(params["ignore_token"])
        except re.error:
            assert False, "Invalid regex"
        params["ignore_token"] = ignore_token_compiled.pattern
        logging.info("Token pattern to be ignored {}".format(params["ignore_token"]))

    return params
