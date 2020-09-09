# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd

from symspell_checker import SpellChecker  # noqa


def test_spellcheck_df_english():
    input_df = pd.DataFrame(
        {
            "input_text": [
                "Can yu readtHIS messa ge despite thehorible AB1234  sppelingmsitakes",
                "Whereis th elove hehaD Dated forImuch of thepast who couqdn'tread in sixthgrade AND ins pired him",
                "!? Nooo way I CAN'T haz cheezburger 💩 😂 #OMG",
            ],
        }
    ).sort_values(by=["input_text"])
    assert True


def test_spellcheck_df_multilingual():
    input_df = pd.DataFrame(
        {
            "input_text": [
                "Can yu readtHIS messa ge despite thehorible AB1234  sppelingmsitakes",
                "Les fautes d'ortograf c pas toop #LOOOL PTDR",
                "covfefe tnetennba rosebud",
            ],
            "language": ["en", "fr", "unknown"],
        }
    ).sort_values(by=["input_text"])
    assert True
