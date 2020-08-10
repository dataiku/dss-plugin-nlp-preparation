# -*- coding: utf-8 -*-
import dataiku
import re
import pandas as pd
import symspellpy
from symspellpy.symspellpy import SymSpell, Verbosity
from text_preprocessing import TextPreprocessor, is_email, is_url, is_mention, is_hashtag, remove_url_email_punct
from plugin_io_utils import generate_unique
from typing import List, AnyStr
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict


from language_dict import SUPPORTED_LANGUAGES

class SpellChecker:
    """
    Spell checker wrapper class on top of symspellpy
    """
    
    # See https://symspellpy.readthedocs.io/en/latest/api/symspellpy.html#symspellpy
    SUGGESTION_VERBOSITY = Verbosity.TOP # returns only the closest word
    TRANSFER_CASING = False # the original casing (lowercase and uppercase) is not carried over the text. Symspellpy returns errors for some words. Note that if a word is all uppercase, symspellpy returns the lowercased word. 
    NUM_THREADS = 4
    
    COLUMN_DESCRIPTION_DICT = OrderedDict(
        [
            ("corrected_text", "Text with misspellings corrected"),
            ("spelling_mistakes", "Spelling mistakes"),
            ("misspelling_count", "Number of spelling mistakes"),
        ]
    )
    
    def __init__(self, 
                text_column: AnyStr = "",
                language_selection: AnyStr = "from_column",
                language_column: AnyStr = "",
                language: AnyStr = "",
                ignore_token: AnyStr = None,
                distance: int = 2, 
                custom_vocabulary_set: List = [],
                folder_of_dictionaries: AnyStr = ""
                ):
        
        self.language_column = language_column
        self.language = language
            
        self.SUPPORTED_LANG_CODE = SUPPORTED_LANGUAGES.keys()
        
        self.text_column = text_column
                
        ## preprocessing ##
        self.text_preprocessor = TextPreprocessor()
        
        # custom vocabulary list
        self.custom_vocabulary_set = custom_vocabulary_set
        
        ## symspell ##
        self.distance = distance
        self.folder_of_dictionaries = folder_of_dictionaries
        self.ignore_token = ignore_token
        
        # sym spell objects as dictionary lang_id: sym_spell_object
        self.sym_spells = {}   
            
    def _add_sym_spell_objects(self, new_lang_code_list: List):
        """
        set_of_languages: set of language codes in ISO 639-1 format
        Adds symspell objects. 
        The symspell objects from languages given in chunks are added only if they were not already present in previous chunks.
        """

        for lang_code in new_lang_code_list:
       
            if lang_code not in self.SUPPORTED_LANG_CODE:
                logging.warning("Unsupported language code {}".format(lang_code))
                continue
                
            if lang_code in self.sym_spells.keys():
                # new sym spell object is added only if not already present
                continue
            logging.info("Loading sym spell object for language {}".format(lang_code))
            freq_dict_path = self.folder_of_dictionaries + '/' + lang_code + ".txt"
            self.sym_spells[lang_code] = SymSpell(max_dictionary_edit_distance=self.distance)
            self.sym_spells[lang_code].load_dictionary(freq_dict_path, 0 , 1)
    
    def _fix_typos_in_word(self, word: AnyStr, lang: AnyStr) -> (AnyStr, List, int):
        """
        Returns the corrected word if it has a correction, and the word not corrected otherwise
        See details here:
            https://symspellpy.readthedocs.io/en/latest/examples/lookup.html
        """
        correction = self.sym_spells[lang].lookup(word, 
                                                  self.SUGGESTION_VERBOSITY,  
                                                  transfer_casing=self.TRANSFER_CASING,
                                                  ignore_token=self.ignore_token
                                                  )
    
        if correction:
            corrected_word = correction[0].term

            if correction[0].term != word.lower() and word != ' ':
                misspell = word
                misspell_count = correction[0].count
            else:
                misspell = ""
                misspell_count = 0
            
        else:
            corrected_word = word # if no correction was found, original word is returned
            misspell = word # if no correction was found, we assume the word is misspelled
            misspell_count = 0
            
        return (corrected_word, misspell, misspell_count)
    
    def _fix_typos_in_document(self, token_lang: List) -> (AnyStr, List, int):

        """
        we did not consider word_segmentation as it is much slower. 
            https://symspellpy.readthedocs.io/en/latest/examples/word_segmentation.html
        """
        
        token_list, lang = [token_lang[0], token_lang[1]]
        if lang not in self.SUPPORTED_LANG_CODE:
            logging.warning("Unsupported language code {}".format(lang))
            return ('', '', '')
        
        if token_list != token_list:
                return ('', '', '')
            
        if len(token_list) == 0:
            return ('', '', '')

        corrected_sentence = []
        misspellings = []
        misspelling_count = 0

        for token in token_list:
            
            # word without . and / (intentionally left to check emails and urls)
            word = remove_url_email_punct(str(token))
            
            # check for url and emails
            if is_url(token) or is_email(token):
                corrected_sentence.append(str(token))
                
            # check for mentions and hashtags
            elif is_hashtag(token) or is_mention(token):
                corrected_sentence.append(str(token))

            elif word not in self.custom_vocabulary_set and not word.isdigit():
                (corrected_word, misspell, misspell_count) = self._fix_typos_in_word(word, lang)
                corrected_sentence.append(corrected_word)
                if len(misspell) > 0:
                    misspellings.append(misspell)
                    misspelling_count += 1
            else:
                corrected_sentence.append(word)

        corrected_sentence = ' '.join(corrected_sentence)
        
        return (corrected_sentence, misspellings, misspelling_count)
            
    def fix_typos_in_df(self, df: pd.DataFrame) -> pd.DataFrame:
        
        existing_column_names = list(df.columns)
        
        if self.language != "": # if language is selected from list
            # create a new column containing language
            lang_col = generate_unique("", existing_column_names, 'language')
            existing_column_names.append(lang_col)
            df[lang_col] = [self.language]*df.shape[0]
            
        else:
            lang_col = self.language_column
                    
        ### Name creation of new columns ###
        
        # column of preprocessed text
        preprocess_col = generate_unique(self.text_column, existing_column_names, 'preprocess')
        existing_column_names.append(preprocess_col)
                
        # the three output column name creation is based on self.COLUMN_DESCRIPTION_DICT
        self.column_description_dict = OrderedDict()

        for k, v in self.COLUMN_DESCRIPTION_DICT.items():
            new_col_name = generate_unique(k, existing_column_names, self.text_column)
            self.column_description_dict[new_col_name] = v
            existing_column_names.append(new_col_name)

        ### Preprocessing of the text ###
        
        logging.info("Text preprocessing ...")
        output_df = self.text_preprocessor.compute(df, 
                                                   self.text_column, 
                                                   preprocess_col, 
                                                   lang_col)

        ### Add new sym_spell objects ###
        
        # As we process data by chunk of 10K rows, 
        # the class SymSpell is instantiated before the chunk processing. 
        # Hence, the dictionaries from languages given in chunks are added only if they were not already present in previous chunks.
        new_lang_code_list = list(df[lang_col].unique())
        self._add_sym_spell_objects(new_lang_code_list)
        
        ### parallel processing for the method _fix_typos_in_document ###
        
        logging.info("Fixing typos ...")

        # iterator creation over the tuple (tokenized document, laguage) 
        doc_iterator = ((token, lang) for (token, lang) in zip(df[preprocess_col], df[lang_col].astype(str)))
        # Parallel computing 
        # lang_output_tuple_list contains the columns as list for ("corrected_text", "spelling_mistakes", "misspelling_count")
        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            lang_output_tuple_list = list(executor.map(self._fix_typos_in_document, doc_iterator))
            
        # copy of output column in the output dataframe
        for i, col in enumerate(self.column_description_dict.keys()):
            output_df[col] = [t[i] for t in lang_output_tuple_list]
            
        # remove unecessary columns
        del output_df[preprocess_col]
        existing_column_names = [k for k in existing_column_names if k != preprocess_col]
        
        if self.language != "":
            del output_df[lang_col]
            

        return output_df
