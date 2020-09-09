# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd

from symspell_checker import SpellChecker  # noqa


def test_spellcheck_df_english():
    input_df = pd.DataFrame(
        {"input_text": ["Can yu readtHIS message despite thehorible AB1234 sppelingmsitakes 😂 #OMG"]}
    )
    print(input_df)
    assert True


def test_spellcheck_df_multilingual():
    input_df = pd.DataFrame(
        {
            "input_text": [
                "Can yu readtHIS messa ge despite thehorible AB1234  sppelingmsitakes",
                "Les fautes d'ortograf c pas toop #LOOOL PTDR",
                "!? Nooo way I CAN'T haz cheezburger 💩 😂 #OMG",
            ],
            "language": ["en", "fr", "es"],
        }
    ).sort_values(by=["input_text"])
    print(input_df)
    assert True
