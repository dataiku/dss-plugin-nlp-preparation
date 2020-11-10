# -*- coding: utf-8 -*-
"""Language Detection recipe script"""

from plugin_config_loading import load_plugin_config_langdetect
from language_detector import LanguageDetector
from dku_io_utils import process_dataset_chunks, set_column_descriptions

# Setup
params = load_plugin_config_langdetect()
detector = LanguageDetector(
    language_scope=params["language_scope"],
    minimum_score=params["minimum_score"],
    fallback_language=params["fallback_language"],
)

# Run
process_dataset_chunks(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    text_column=params["text_column"],
    func=detector.detect_languages_df,
)
set_column_descriptions(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    column_descriptions=detector.column_descriptions,
)
