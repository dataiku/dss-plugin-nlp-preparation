# -*- coding: utf-8 -*-
from language_dict import SUPPORTED_LANGUAGES_PYCLD3, SUPPORTED_LANGUAGES_SYMSPELL


def do(payload, config, plugin_config, inputs):
    if config.get("edit_distance"):
        language_choices = sorted(
            [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES_SYMSPELL.items()], key=lambda x: x.get("label")
        )
        language_choices.insert(0, {"label": "Detected language column", "value": "language_column"})
    else:
        language_choices = sorted(
            [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES_PYCLD3.items()], key=lambda x: x.get("label")
        )
        language_choices.insert(0, {"label": "None", "value": "None"})
    return {"choices": language_choices}
