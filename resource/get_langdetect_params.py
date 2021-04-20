# -*- coding: utf-8 -*-
from language_support import SUPPORTED_LANGUAGES_PYCLD3


def do(payload, config, plugin_config, inputs):
    language_choices = sorted(
        [{"value": k, "label": v} for k, v in SUPPORTED_LANGUAGES_PYCLD3.items()], key=lambda x: x.get("label")
    )
    if payload["parameterName"] == "fallback_language":
        language_choices.insert(0, {"label": "None", "value": "None"})
    return {"choices": language_choices}
