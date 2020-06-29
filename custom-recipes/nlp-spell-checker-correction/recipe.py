import dataiku
from dataiku.customrecipe import *
import logging
from spell_checker.input_output_params import *
from spell_checker.spell_checker import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="timeseries-preparation plugin %(levelname)s - %(message)s")

# --- Setup
input_dataset, not_to_be_corrected_dataset, output_dataset = get_input_output()
recipe_config = get_recipe_config()
params = get_spell_checker_params(recipe_config, not_to_be_corrected_dataset)

# --- Run
df = dataiku.Dataset(input_dataset).get_dataframe()
spell_chckr = SpellChecker(params)
output_df = spell_chckr.compute(df)

# --- Write output
output = dataiku.Dataset(output_dataset)
output.write_with_schema(df)
