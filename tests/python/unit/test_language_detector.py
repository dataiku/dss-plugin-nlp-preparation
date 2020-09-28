# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd
import numpy as np

from language_detector import LanguageDetector  # noqa


INPUT_DF = pd.DataFrame(
    {
        "input_text": [
            "Comment est votre blanquette ?",
            "このオレはいずれ火影の名を受け継いで、先代のどの火影をも超えてやるんだ",
            "Every performance is an adventure with this group. They're called Fire Saga.",
            "",
            "1",
        ],
    }
).sort_values(by=["input_text"])


OUTPUT_DF = pd.DataFrame()
OUTPUT_DF["input_text"] = INPUT_DF["input_text"]
OUTPUT_DF["input_text_language_code"] = ["", "es", "fr", "en", "ja"]
OUTPUT_DF["input_text_language_name"] = ["", "Spanish", "French", "English", "Japanese"]
OUTPUT_DF["input_text_language_score"] = [np.NaN, np.NaN, 1.0, 1.0, 1.0]


def test_language_detector():
    detector = LanguageDetector(minimum_score=0.2, fallback_language="es")
    output_df = detector.detect_languages_df(INPUT_DF, "input_text").sort_values(by=["input_text"])
    for col in output_df.columns:
        np.testing.assert_array_equal(output_df[col].values, OUTPUT_DF[col].values)
