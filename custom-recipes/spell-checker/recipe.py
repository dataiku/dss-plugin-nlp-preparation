import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from plugin_config_loading import load_plugin_config, custom_vocabulary_checker
from spell_checker import SpellChecker
from dku_io_utils import process_dataset_chunks

# --- Setup
input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
output_dataset = dataiku.Dataset(get_output_names_for_role("output_dataset")[0])

custom_vocabulary = get_input_names_for_role('custom_vocabulary')
custom_vocabulary_set = custom_vocabulary_checker(custom_vocabulary)

params = load_plugin_config(get_recipe_config(), custom_vocabulary_set)

# --- Run
df = input_dataset.get_dataframe()
spell_checker_object = SpellChecker(
                text_column_list = params['text_column_list'],
                language_selection = params['language_selection'],
                language_column = params['language_column'],
                language = params['language'],
                ignore_token = params['ignore_token'],
                custom_vocabulary_set = params['custom_vocabulary_set'],
                folder_of_dictionaries = params['folder_of_dictionaries']
                )

output_df = spell_checker_object.compute(df)

# --- Write output

process_dataset_chunks(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    func=spell_checker_object.compute
)