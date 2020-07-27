# -*- coding: utf-8 -*-
from typing import List, AnyStr
import re
import spacy.lang
import pandas as pd
import logging
import string
from plugin_io_utils import generate_unique


from language_dict import SUPPORTED_LANGUAGES

class TextPreprocessor:
    
    PUNCTUATION = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~_！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    
    def __init__(self):
        self.tokenizers = {}
        self.supported_lang_code = SUPPORTED_LANGUAGES.keys()
        
    def _add_tokenizers(self, lang_code_new_list: List[AnyStr]):
        
        language_modules = {}
        nlps = {}
        
        for lang_code in lang_code_new_list:
            
            if lang_code != lang_code: # check for NaNs
                logging.info("Missing language code")
                continue
                
            if lang_code not in self.supported_lang_code:
                logging.info("Unsupported language code {}".format(lang_code))
                continue
                
            if lang_code in self.tokenizers.keys():
                # new tokenizer is added only if not already present
                continue
            
            # Special treatment for Korean as Spacy korean tokenizer has dependecies
            if lang_code == 'ko':
                from konlpy.tag import Hannanum
                self.tokenizers[lang_code] = Hannanum()
                
            else:
                lang_name = SUPPORTED_LANGUAGES[lang_code]

                # module import
                logging.info("Loading tokenizer object for language {}".format(lang_code))
                __import__("spacy.lang." + lang_code)
                language_modules[lang_code] = getattr(spacy.lang, lang_code)

                # tokenizer creation
                nlps[lang_code] = getattr(language_modules[lang_code], lang_name)()
                self.tokenizers[lang_code] = nlps[lang_code].Defaults.create_tokenizer(nlps[lang_code])
                
    def _first_preprocessing(self, doc: AnyStr, lang: AnyStr, lowercase: bool) -> AnyStr:
        # remove edge cases
        if lang not in self.supported_lang_code:
            return []
        if doc != doc: # check for NaNs
            return []
        if len(str(doc)) == 0:
            return []
        
        # lowercase
        if lowercase:
            return str(doc).lower()
        else:
            return str(doc)
                
    def _remove_puncutation(self, doc: AnyStr, lang: AnyStr) -> AnyStr:
        if doc != []:
            doc = re.sub(r"[%s]+" %self.PUNCTUATION, " ", doc).replace('-', ' ') # hyphens are generally not escaped in regex
            # remove leading spaces and multiple spaces (often created by removing punctaution and causing bad spelling replacement)
            return ' '.join(str(doc).split())
        if len(str(doc)) == 0:
            return []
        else:
            return []

    def _tokenize(self, doc: AnyStr, lang: AnyStr) -> List:
        if doc != []:
            if lang == 'ko':
                tokens = [str(k) for k in self.tokenizers[lang].morphs(doc)]
            else:
                tokens = [str(k) for k in self.tokenizers[lang](doc)]
            return tokens
        else:
            return []
        
    def compute(self, 
                df: pd.DataFrame, 
                txt_col: AnyStr, 
                preprocess_col: AnyStr, 
                lang_col: AnyStr, 
                tokenize: bool = True,
                remove_puncutation: bool = True,
                lowercase: bool = True) -> pd.DataFrame:
        """
        Returns either a token list or an empty list.
        Empty list is returned if the document is NaN, empty or if the language is not supported.
        """
        
        # add tokenizers
        self._add_tokenizers(list(df[lang_col].unique()))
        
        # remove edge cases and lowercase the text
        existing_column_names = list(df.columns)
        first_preprocessing_col = generate_unique(txt_col, existing_column_names, 'edge_case')
        
        df[first_preprocessing_col] = df.apply(lambda x:self._first_preprocessing(x[txt_col],
                                                                              x[lang_col],
                                                                              lowercase),
                                      axis=1)
        
        # remove punctuation
        if remove_puncutation:
            df[preprocess_col] = df.apply(lambda x:self._remove_puncutation(x[first_preprocessing_col], 
                                                                            x[lang_col]),
                                         axis=1)
            
        else:
            df[preprocess_col] = df[first_preprocessing_col]

        # tokenize
        df[preprocess_col] = df.apply(lambda x:self._tokenize(x[preprocess_col],
                                                              x[lang_col]),
                                      axis=1)
        
        del df[first_preprocessing_col]

        return df