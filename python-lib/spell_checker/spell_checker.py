import dataiku
from symspellpy.symspellpy import SymSpell, Verbosity
from spell_checker.language_detection import *
from spell_checker.dataframe_helpers import *
from spell_checker.text_preprocessing import *


class SpellChecker:
    def __init__(self, params):
        self.params = params

        # Root path of frequency dictionaries
        self.resource_path = dataiku.customrecipe.get_recipe_resource()

        # sym_spell params. See https://symspellpy.readthedocs.io/en/latest/api/symspellpy.html#symspellpy
        self.distance = 2
        self.suggestion_verbosity = Verbosity.TOP
        # The original casing (lowercase and uppercase) is carried over the text.
        # Note that if a word is all uppercase, symspellpy returns the lowercased word.
        self.transfer_casing = True

    def _get_all_languages(self, df):
        all_languages = set()
        for col_lang in self.col_txt_lang_dict.values():
            all_languages = all_languages.union(set(df[col_lang].unique()))

        return all_languages

    def _create_sym_spell_objects(self, df):
        """
        Creation of all sym_spell objects and load frequency dictionaries
        sym_spell objects are stored in a dictionary :
            - language: sym_spell['language']
        See https://symspellpy.readthedocs.io/en/latest/examples/dictionary.html
        """

        sym_spells = {}

        for language in self.all_languages:
            freq_dict_path = self.resource_path + "/" + language + "_50k.txt"
            sym_spells[language] = SymSpell(max_dictionary_edit_distance=2)
            sym_spells[language].load_dictionary(freq_dict_path, 0, 1)

        return sym_spells

    def _fix_typos_in_word(self, word, sym_spell):
        """
        Returns the corrected word if it has a correction, and the word not corrected otherwise
        See details here:
            https://symspellpy.readthedocs.io/en/latest/examples/lookup.html
        """
        correction = sym_spell.lookup(
            word,
            self.suggestion_verbosity,
            self.distance,
            transfer_casing=self.transfer_casing,
            ignore_token=self.params["ignore_token"],
        )
        return correction[0].term if correction else word

    def _fix_typos_in_document(self, string_or_token_list, sym_spell):
        """
        Returns the corrected document
        See details here for word segmentation: 
            https://symspellpy.readthedocs.io/en/latest/examples/word_segmentation.html
        """
        if not self.params["word_segmentation"]:
            sent_fixed = []
            for word in string_or_token_list:
                if word not in self.params["set_untouched_words"] and not word.isdigit():
                    sent_fixed.append(self._fix_typos_in_word(word, sym_spell))
                else:
                    sent_fixed.append(word)
            return " ".join(sent_fixed)
        else:
            try:
                # for some reason, word_segmentation got a stopIteration error. This is a symspellpy package issue.
                return sym_spell.word_segmentation(
                    string_or_token_list, ignore_token=self.params["ignore_token"]
                ).corrected_string
            except:
                print("Couldn't segment the string:", string_or_token_list)
                return string_or_token_list

    def compute(self, df):

        # new column names and matching dict
        (
            self.df_all_columns,
            self.col_txt_lang_dict,
            self.col_txt_preprocessed_dict,
            self.col_txt_spellchecked_dict,
        ) = create_all_new_column_names(df.columns, self.params["text_col_list"])

        # Language detection
        for col_txt, col_lang in self.col_txt_lang_dict.items():
            df[col_lang] = get_language_from_column(df[col_txt], df.shape[0], self.params)

        # Get all languages of the corpus
        self.all_languages = self._get_all_languages(df)

        # Preprocessint the text
        preprocessing = PreprocessText(self.all_languages, self.params)
        df = preprocessing.compute(df, self.col_txt_preprocessed_dict, self.col_txt_lang_dict)

        # Create sym_spell objects
        self.sym_spells = self._create_sym_spell_objects(df)

        # Fix typos
        print("Fixing typos ...")
        for col_txt, col_spellchecked in self.col_txt_spellchecked_dict.items():
            col_lang = self.col_txt_lang_dict[col_txt]
            col_preprocessed = self.col_txt_preprocessed_dict[col_txt]
            df[col_spellchecked] = df.apply(
                lambda x: self._fix_typos_in_document(x[col_preprocessed], self.sym_spells[x[col_lang]]), axis=1
            )

        # col of pre-processed text are removed
        for col_preprocessed in self.col_txt_preprocessed_dict.values():
            del df[col_preprocessed]

        # if language is given in the UI, column of the language is removed
        if not self.params["detect_language"]:
            for col_lang in self.col_txt_lang_dict.values():
                del df[col_lang]

        return df
