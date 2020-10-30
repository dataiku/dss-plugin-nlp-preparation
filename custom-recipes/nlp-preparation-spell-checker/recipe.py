# -*- coding: utf-8 -*-
"""Misspelling Correction recipe script"""

from plugin_config_loading import load_plugin_config_spellchecker
from spacy_tokenizer import MultilingualTokenizer
from symspell_checker import SpellChecker
from dku_io_utils import process_dataset_chunks, set_column_descriptions

# Setup
params = load_plugin_config_spellchecker()
spellchecker = SpellChecker(
    tokenizer=MultilingualTokenizer(stopwords_folder_path=params["stopwords_folder_path"]),
    dictionary_folder_path=params["dictionary_folder_path"],
    ignore_token=params["ignore_word_regex"],
    edit_distance=params["edit_distance"],
    custom_vocabulary_set=params["custom_vocabulary_set"],
    custom_corrections=params["custom_corrections"],
    compute_diagnosis=params["compute_diagnosis"],
)

# Run
process_dataset_chunks(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    func=spellchecker.check_df,
    text_column=params["text_column"],
    language=params["language"],
    language_column=params["language_column"],
)
set_column_descriptions(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    column_descriptions=spellchecker.output_column_descriptions,
)
if params["compute_diagnosis"]:
    diagnosis_df = spellchecker.create_diagnosis_df()
    params["diagnosis_dataset"].write_with_schema(diagnosis_df)
    set_column_descriptions(
        output_dataset=params["diagnosis_dataset"], column_descriptions=spellchecker.DIAGNOSIS_COLUMN_DESCRIPTIONS,
    )
