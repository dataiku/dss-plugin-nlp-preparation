import re
import spacy.lang

# All languages available in Spacy and for which we have a frequency dictionnary
lang_dict = {'sq': 'Albanian',
 'ar': 'Arabic',
 'bg': 'Bulgarian',
 'ca': 'Catalan',
 'zh': 'Chinese',
 'hr': 'Croatian',
 'cs': 'Czech',
 'da': 'Danish',
 'nl': 'Dutch',
 'en': 'English',
 'et': 'Estonian',
 'fi': 'Finnish',
 'fr': 'French',
 'de': 'German',
 'el': 'Greek',
 'he': 'Hebrew',
 'hu': 'Hungarian',
 'is': 'Icelandic',
 'id': 'Indonesian',
 'it': 'Italian',
 'ja': 'Japanese',
 'ko': 'Korean',
 'lv': 'Latvian',
 'lt': 'Lithuanian',
 'fa': 'Persian',
 'pl': 'Polish',
 'pt': 'Portuguese',
 'ro': 'Romanian',
 'ru': 'Russian',
 'sr': 'Serbian',
 'sk': 'Slovak',
 'sl': 'Slovenian',
 'es': 'Spanish',
 'sv': 'Swedish',
 'th': 'Thai',
 'tr': 'Turkish',
 'uk': 'Ukrainian',
 'vi': 'Vietnamese'}



class PreprocessText:
    def __init__(self, all_languages, params=None):
        self.all_languages = all_languages
        if params != None:
            self.params = params

    def _get_all_tokenizers(self):
        
        language_modules = {}
        nlps = {}
        tokenizers = {}
        
        for lang_code, lang_name in lang_dict.items():
            
            if lang_code not in self.all_languages:
                continue
            __import__("spacy.lang." + lang_code)
            language_modules[lang_code] = getattr(spacy.lang, lang_code)
            nlps[lang_code] = getattr(language_modules[lang_code], lang_name)()
            tokenizers[lang_code] = nlps[lang_code].Defaults.create_tokenizer(nlps[lang_code])
        
        return tokenizers

    def _tokenize(self, doc, tokenizer):
        return [str(k) for k in tokenizer(doc)]

    def _preprocess(self, doc, lang, word_segmentation, tokenizers):
        """
        doc is a string. The function returns string_or_token_list
        - string_or_token_list is the tokenized text if word_segmentation is true
        - string_or_token_list is str(doc) otherwise
        """
        if not word_segmentation:
            doc = str(doc) # pass str(doc) to avoid non-string (like numbers) issues
            # tokenize
            string_or_token_list = self._tokenize(doc, tokenizers[lang])
        else:
            string_or_token_list = str(doc)
        return string_or_token_list
    
    def compute(self, df, col_txt_preprocessed_dict, col_txt_lang_dict):
         # Get all tokenizers
        tokenizers = self._get_all_tokenizers()

        # Pre processing of the text
        print('Pre-processing text ...')
        for col_txt, col_preprocessed in col_txt_preprocessed_dict.items():
            col_lang = col_txt_lang_dict[col_txt]
            df[col_preprocessed] = df.apply(lambda x:self._preprocess(x[col_txt], 
                                                                      x[col_lang], 
                                                                      self.params['word_segmentation'],
                                                                      tokenizers), axis=1)
            
        return df
        
        

