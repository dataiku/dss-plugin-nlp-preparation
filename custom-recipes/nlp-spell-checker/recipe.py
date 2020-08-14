import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from plugin_config_loading import load_plugin_config
from spacy_tokenizer import MultilingualTokenizer
from symspell_checker import SpellChecker
from dku_io_utils import process_dataset_chunks, set_column_description

# Setup
input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
params = load_plugin_config(get_recipe_config())

# Run
tokenizer = MultilingualTokenizer()
spell_checker = SpellChecker(
    tokenizer=tokenizer,
    dictionary_folder_path=params["dictionary_folder_path"],
    ignore_token=params["ignore_word_regex"],
    edit_distance=params["edit_distance"],
    custom_vocabulary_set=params["custom_vocabulary_set"],
)

# Write output
process_dataset_chunks(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    func=spell_checker.fix_typos_in_df,
    text_column=params["text_column"],
    language=params["language"],
    language_column=params["language_column"],
)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=spell_checker.column_description_dict,
)
