# -*- coding: utf-8 -*-
from spacy_tokenizer import MultilingualTokenizer
from language_dict import SUPPORTED_LANGUAGES_SPACY


def do(payload, config, plugin_config, inputs):
    choices = []
    if payload["parameterName"] == "language":
        choices = sorted(
            [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES_SPACY.items()], key=lambda x: x.get("label")
        )
        choices.insert(0, {"label": "Detected language column", "value": "language_column"})
    if payload["parameterName"] == "token_filters":
        choices = [{"value": k, "label": v} for k, v in MultilingualTokenizer.DEFAULT_FILTER_TOKEN_ATTRIBUTES.items()]
    return {"choices": choices}
