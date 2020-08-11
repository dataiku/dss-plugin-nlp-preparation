import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from plugin_config_loading import load_plugin_config
from spell_checker import SpellChecker
from dku_io_utils import process_dataset_chunks, set_column_description

# Setup
input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])
params = load_plugin_config(get_recipe_config())

# Run
spell_checker = SpellChecker(
    text_column=params["text_column"],
    language_selection=params["language_selection"],
    language_column=params["language_column"],
    language=params["language"],
    ignore_token=params["ignore_token"],
    distance=params["distance"],
    custom_vocabulary_set=params["custom_vocabulary_set"],
    folder_of_dictionaries=params["folder_of_dictionaries"],
)

# Write output
process_dataset_chunks(input_dataset=input_dataset, output_dataset=output_dataset, func=spell_checker.fix_typos_in_df)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=spell_checker.column_description_dict,
)
