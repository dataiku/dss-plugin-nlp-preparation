# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, AnyStr, Set
import dataiku
import re

from language_dict import SUPPORTED_LANGUAGES

def load_plugin_config(recipe_config: Dict, custom_vocabulary_set: List) -> Dict:
    """
    Helper function to load plugin recipe config into a clean parameter dictionary.
    Applies assertion checks for correct input config.
    """
    params = {}
        
    # path to the folder of dictionaries
    params['folder_of_dictionaries'] = dataiku.customrecipe.get_recipe_resource()
    
    # List of text columns
    params["text_column_list"] = recipe_config.get("text_column_list")

    assert params["text_column_list"] is not None and params["text_column_list"] != [], "Empty text column selection"
    logging.info("List of text column: {}".format(params["text_column_list"]))
    
    # custom_vocabulary_list
    params['custom_vocabulary_set'] = custom_vocabulary_set
    logging.info("Custom vocabulary set: {}".format(params["custom_vocabulary_set"]))
    
    # Language selection
    params["language_selection"] = recipe_config.get("language_selection")
    
    if params["language_selection"] == 'from_list':
        params["language"] = recipe_config.get("language_from_list")
        assert params["language"] is not None and params["language"] != '', "Empty language selection"
        logging.info("Language: {}".format(params["language"]))
        params["language_column"] = ""

    if params["language_selection"] == 'from_column':
        params["language_column"] = recipe_config.get("language_from_column")
        params["language"] = ""
        assert params["language_column"] is not None and params["language_column"] != '', "Empty language column selection"
        logging.info("Language codes from column {}".format(params["language_column"]))

    # Expert mode
    if recipe_config.get("expert"):
        logging.info("Expert mode is enabled")
        
        # ignore token
        if len(recipe_config.get("ignore_token")) == 0:
            logging.info("No token to be ignored")
            params["ignore_token"] = None # symspellpy wants None
        else:
            params["ignore_token"] = recipe_config.get("ignore_token")
            # Check for valid regex
            try:
                ignore_token_compiled = re.compile(params["ignore_token"])
            except re.error:
                assert False, "Invalid regex"
            params["ignore_token"] = ignore_token_compiled.pattern
            logging.info("Token pattern to be ignored {}".format(params["ignore_token"]))

    else:
        logging.info("Expert mode is disabled")

        params["ignore_token"] = None
                     
        logging.info("No token to be ignored")
                             
    return params

def custom_vocabulary_checker(custom_vocabulary: List) -> Set:
    
    if len(custom_vocabulary) == 1:
        
        custom_vocabulary_dataset = dataiku.Dataset(custom_vocabulary[0])
        dataset_schema = custom_vocabulary_dataset.get_config()['schema']
        columns = dataset_schema['columns']
        
        assert len(columns) == 1, "Custom vocabulary dataset must have only one column"

        col_name = columns[0]['name']
        col_type = columns[0]['type']
        
        assert col_type == 'string', "Column of custom vocabulary dataset must be of type string"
        
        custom_vocabulary_set =  set(custom_vocabulary_dataset.get_dataframe()[col_name].tolist())
            
    else:
        custom_vocabulary_set = set()
    
    return custom_vocabulary_set
        
        

