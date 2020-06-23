import dataiku
from dataiku.customrecipe import *
import re

def get_input_output():
    input_dataset = get_input_names_for_role('input_dataset')[0]
    not_to_be_corrected_dataset = get_input_names_for_role('not_to_be_corrected_dataset')
    
    not_to_be_corrected_dataset = NotToBeCorrectedDataset(not_to_be_corrected_dataset)
    
    if not_to_be_corrected_dataset.is_present:
        not_to_be_corrected_dataset.check()
    
    output_dataset = get_output_names_for_role('output_dataset')[0]
    
    return input_dataset, not_to_be_corrected_dataset, output_dataset


def get_spell_checker_params(recipe_config, not_to_be_corrected_dataset):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)
    
    params = {}

    # General parameters
    params["text_col_list"] = _p('text_col_list')
    params["suffix"] = _p('suffix')

    # Language parameters
    params["language"] = _p('language')
    params["detect_language"] = _p('detect_language')
    params["single_language_per_column"] = _p('single_language_per_column')
    params["constrain_languages"] = _p('constrain_languages')

    if params["detect_language"]:
        params["single_language_per_column"] = _p('single_language_per_column')
        params["constrain_languages"] = _p("constrain_languages") 
            
    # Get set of untouched words
    if not_to_be_corrected_dataset.is_present:
        params["set_untouched_words"] = list(not_to_be_corrected_dataset.get_df().iloc[:,0])
    else:
        params["set_untouched_words"] = []

    # Advanced parameters
    params["ignore_token"] = _p('ignore_token')
        
    # Create regex that matches the set of untouched word
    # This must be given to preserve untouched words while performing word_segmentation
    if len(params["set_untouched_words"]) > 0:
        params["ignore_token"] = create_regex_that_matches_set_untouched_words(params["ignore_token"], 
                                                                               params["set_untouched_words"])
        
    
    if len(params["ignore_token"]) == 0:
        params["ignore_token"] = None # None is the expected value in symspellpy when no ignore_token is given
    else:
        params["ignore_token"] = re.compile(params["ignore_token"])

    params["word_segmentation"] = _p('word_segmentation')

    return params


class NotToBeCorrectedDataset:
    """
    Sanity check for the dataframe not_to_be_corrected_df
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        
        #self.df = dataset.get_dataframe()
        if len(self.dataset) > 0:
            # flag indicating whether or not there is a dataset of word not to be corrected
            self.is_present = True 
            self._get_df()
        else: 
            self.is_present = False
            
    def _get_df(self):
        self.df = dataiku.Dataset(self.dataset[0]).get_dataframe()        
            
    def check(self):
        
        # The not_to_be_corrected_df dataset is supposed to have only one column containing a word per row
        if len(self.df.columns) > 1:
            raise InputDatasetError("Dataset {0} has more than one column.".format(self.dataset[0]))
            
    def get_df(self):
        return self.df
    
    
def create_regex_that_matches_set_untouched_words(ignore_token, set_untouched_words):
    """
    To use word_semgentation with untouched words, we must pass as a regex the set of untouched words
    """
    
    if len(ignore_token) >0:
        ignore_token = ignore_token[2:-1] # remove leading r" and ending "
        # If a regex for ignore_token was given in the UI, we simply add the vertical bar "|" meaning "or" for the regex.
        ignore_token += "|"
    for word in set_untouched_words:
        # Creation of the regex that matches all words in set_untouched_words
        ignore_token += r"\b" + str(word) + r"\b|"
    # remove last |
    ignore_token = ignore_token[:-1]
    
    return ignore_token