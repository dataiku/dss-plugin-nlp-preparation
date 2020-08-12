# -*- coding: utf-8 -*-
from language_dict import SUPPORTED_LANGUAGES


def do(payload, config, plugin_config, inputs):
    language_choices = [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES.items()]
    language_choices.insert(0, {"label": "Detected language column", "value": "language_column"})
    return {"choices": language_choices}
