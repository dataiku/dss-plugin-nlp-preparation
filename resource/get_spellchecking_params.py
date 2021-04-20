# -*- coding: utf-8 -*-
from language_support import SUPPORTED_LANGUAGES_SYMSPELL, SUPPORTED_LANGUAGES_SPACY


def do(payload, config, plugin_config, inputs):
    language_choices = sorted(
        [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES_SYMSPELL.items() if k in SUPPORTED_LANGUAGES_SPACY],
        key=lambda x: x.get("label"),
    )
    language_choices.insert(0, {"label": "Multilingual", "value": "language_column"})
    return {"choices": language_choices}
