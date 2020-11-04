# -*- coding: utf-8 -*-
"""Text Cleaning recipe script"""

from plugin_config_loading import load_plugin_config_cleaning
from spacy_tokenizer import MultilingualTokenizer
from text_cleaner import TextCleaner
from dku_io_utils import process_dataset_chunks, set_column_descriptions

# Setup
params = load_plugin_config_cleaning()
text_cleaner = TextCleaner(
    tokenizer=MultilingualTokenizer(stopwords_folder_path=params["stopwords_folder_path"]),
    token_filters=params["token_filters"],
    lemmatization=params["lemmatization"],
    lowercase=params["lowercase"],
    unicode_normalization=params["unicode_normalization"],
    keep_filtered_tokens=params["keep_filtered_tokens"],
)

# Run
process_dataset_chunks(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    func=text_cleaner.clean_df,
    text_column=params["text_column"],
    language=params["language"],
    language_column=params["language_column"],
)
set_column_descriptions(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    column_descriptions=text_cleaner.output_column_descriptions,
)
