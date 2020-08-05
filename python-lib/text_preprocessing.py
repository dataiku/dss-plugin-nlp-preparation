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
        self.SUPPORTED_LANG_CODE = SUPPORTED_LANGUAGES.keys()
        
    def _add_tokenizers(self, lang_code_new_list: List[AnyStr]):
        """
        Adds tokenizers. 
        The tokenizers from languages given in chunks are added only if they were not already present in previous chunks.
        """
        
        language_modules = {}
        nlps = {}
        
        for lang_code in lang_code_new_list:
            
            if lang_code != lang_code: # check for NaNs
                logging.warning("Missing language code")
                continue
                
            if lang_code not in self.SUPPORTED_LANG_CODE:
                logging.warning("Unsupported language code {}".format(lang_code))
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
                
    def _normalize_text(self, doc: AnyStr,
                             lang: AnyStr,
                             lowercase: bool,
                             remove_punctuation: bool) -> AnyStr:
        """
        - remove edge case: language not supported and empty string
        - lowercase: if a word is all lowercase, symspell returns a mix of capital and lowercase letter in a word
        - remove punctuation: tokenizers often keep punctutation in seperates tokens. symspell corrects punctuation into a word
        """
        
        # remove edge cases
        if lang not in self.SUPPORTED_LANG_CODE:
            return []
        if doc != doc: # check for NaNs
            return []
        if len(str(doc)) == 0:
            return []
        
        # lowercase
        if lowercase:
            doc = str(doc).lower()
        else:
            doc = str(doc)
            
        # remove_punctuation
        if remove_punctuation:
            # Remove punctuation with regex. Remove hyphens with replace. Hyphens are generally not escaped in regex
            doc = re.sub(r"[%s]+" %self.PUNCTUATION, " ", doc).replace('-', ' ') 
            
        # Remove leading spaces and multiple spaces (often created by removing punctuation and causing bad tokenized doc)
        doc = ' '.join(str(doc).split())
        
        if len(str(doc)) == 0:
            return []
        else:
            return doc

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
        # As we process data by chunk of 10K rows, 
        # the class TextPreprocessor is instantiated before the chunk processing. 
        # Hence, the tokenizers from languages given in chunks are added only if they were not already present in previous chunks.
        self._add_tokenizers(list(df[lang_col].unique()))
        
        # remove edge cases, lowercase, remove punctuation
        existing_column_names = list(df.columns)
        normalized_text_column = generate_unique(txt_col, existing_column_names, 'normalized')
        
        df[normalized_text_column] = df.apply(lambda x:self._normalize_text(x[txt_col],
                                                                                  x[lang_col],
                                                                                  lowercase,
                                                                                  remove_puncutation),
                                               axis=1)

        # tokenize
        df[preprocess_col] = df.apply(lambda x:self._tokenize(x[normalized_text_column],
                                                              x[lang_col]),
                                      axis=1)
        
        del df[normalized_text_column]

        return df