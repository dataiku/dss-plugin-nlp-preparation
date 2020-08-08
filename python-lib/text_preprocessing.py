# -*- coding: utf-8 -*-
from typing import List, AnyStr
import re
from spacy.tokens.token import Token
import spacy.lang
import pandas as pd
import logging
import string
from plugin_io_utils import generate_unique
from spacy.tokenizer import Tokenizer

from language_dict import SUPPORTED_LANGUAGES

def is_url(token: Token) -> bool:
    return token.like_url

def is_email(token: Token) -> bool:
    return token.like_email
    
def is_mention(token: Token) -> bool:
    return str(token)[0] == '@'
    
def is_hashtag(token: Token) -> bool:
    return str(token)[0] == '#'

class TextPreprocessor:
    
    PUNCTUATION = "!\"$%&()*+,:;<=>?[\\]^_`{|}~_！？｡。＂＄％＆＇（）＊＋，－／：；＜＝＞［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    # special case for hashtags.
    # mentions are already taken into account by SpaCy
    PREFIX_TOKEN = re.compile(r'''#(\w+)''')
    
    def __init__(self):
        self.tokenizers = {}
        self.nlps = {}
        self.SUPPORTED_LANG_CODE = SUPPORTED_LANGUAGES.keys()
        
    def _custom_tokenizer(self, nlp):
        # Tokenizer that preserves hashtags and mentions
        return Tokenizer(nlp.vocab, prefix_search=self.PREFIX_TOKEN.search)
        
    def _add_tokenizers(self, lang_code_new_list: List[AnyStr]):
        """
        Adds tokenizers. 
        The tokenizers from languages given in chunks are added only if they were not already present in previous chunks.
        """
        
#        language_modules = {}
#        nlps = {}
        
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
                
            else:
                lang_name = SUPPORTED_LANGUAGES[lang_code]

                # module import
                logging.info("Loading tokenizer object for language {}".format(lang_code))

                # tokenizer creation
                self.nlps[lang_code] = spacy.blank(lang_code)
                self.nlps[lang_code].tokenizer = self._custom_tokenizer(self.nlps[lang_code])
                
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
            return ''
        if doc != doc: # check for NaNs
            return ''
        if len(str(doc)) == 0:
            return ''
        
        # lowercase
        if lowercase:
            doc = str(doc).lower()
        else:
            doc = str(doc)
            
        # remove_punctuation
        if remove_punctuation:
            # Remove punctuation with regex. Remove hyphens with replace. 
            # For some reasons, if hyphen is in self.PUNCTUATION, it removes also the dot "."
            doc = re.sub(r"[%s]+" %self.PUNCTUATION, " ", doc).replace('-', ' ') 
            
        # Remove leading spaces and multiple spaces (often created by removing punctuation and causing bad tokenized doc)
        doc = ' '.join(str(doc).split())
        
        if len(str(doc)) == 0:
            return ''
        else:
            return doc
        

    def _tokenize_sliced_series(self, sliced_series: pd.DataFrame, index: pd.core.indexes.range.RangeIndex, lang: AnyStr) -> List:
                    
        # tokenize with nlp objets
        token_list = list(self.nlps[lang].pipe(sliced_series.tolist()))
        # append token_list and keep same index
        token_series_sliced = pd.Series(token_list, index=index)
            
        return token_series_sliced
        
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
        
        # Add tokenizers        
        # As we process data by chunk of 10K rows, 
        # the class TextPreprocessor is instantiated before the chunk processing. 
        # Hence, the tokenizers from languages given in chunks are added only if they were not already present in previous chunks.
        lang_list = list(df[lang_col].unique())
        self._add_tokenizers(lang_list)
        
        # remove edge cases, lowercase, remove punctuation
        existing_column_names = list(df.columns)
        normalized_text_column = generate_unique(txt_col, existing_column_names, 'normalized')
        
        df[normalized_text_column] = df.apply(lambda x:self._normalize_text(x[txt_col],
                                                                            x[lang_col],
                                                                            lowercase,
                                                                            remove_puncutation),
                                              axis=1)

        # tokenize
        token_series = pd.Series()
        for lang in self.nlps.keys():
            # slice df with language
            df_sliced = df[df[lang_col]==lang]
            token_series = token_series.append(self._tokenize_sliced_series(df_sliced[normalized_text_column], df_sliced.index, lang))
            
        df[preprocess_col] = token_series
        
        #del df[normalized_text_column]

        return df