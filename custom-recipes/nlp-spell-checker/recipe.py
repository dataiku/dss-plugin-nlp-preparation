from plugin_config_loading import load_plugin_config
from spacy_tokenizer import MultilingualTokenizer
from symspell_checker import SpellChecker
from dku_io_utils import process_dataset_chunks, set_column_description

# Setup
params = load_plugin_config()

# Run
spellchecker = SpellChecker(
    tokenizer=MultilingualTokenizer(batch_size=params["batch_size"]),
    dictionary_folder_path=params["dictionary_folder_path"],
    ignore_token=params["ignore_word_regex"],
    edit_distance=params["edit_distance"],
    custom_vocabulary_set=params["custom_vocabulary_set"],
)

# Write output
process_dataset_chunks(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    chunksize=params["batch_size"],
    func=spellchecker.check_df,
    text_column=params["text_column"],
    language=params["language"],
    language_column=params["language_column"],
)
set_column_description(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    column_description_dict=spellchecker.column_description_dict,
)
