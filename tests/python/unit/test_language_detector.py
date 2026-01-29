# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import numpy as np
import pandas as pd
import pytest

from language_detector import LanguageDetector

INPUT_DF = pd.DataFrame(
    {
        "input_text": [
            "Comment est votre blanquette ?",
            "このオレはいずれ火影の名を受け継いで、先代のどの火影をも超えてやるんだ",
            "Every performance is an adventure with this group. They're called Fire Saga. This is a long string more than 140 chars to make sure it works too.",
            "",
            "1",
        ],
    }
).sort_values(by=["input_text"])

OUTPUT_DF = pd.DataFrame()
OUTPUT_DF["input_text"] = INPUT_DF["input_text"]
OUTPUT_DF["input_text_language_code"] = ["", "es", "fr", "en", "ja"]
OUTPUT_DF["input_text_language_name"] = ["", "Spanish", "French", "English", "Japanese"]
OUTPUT_DF["input_text_language_score"] = [np.nan, np.nan, 1.0, 1.0, 1.0]

def test_language_detector():
    detector = LanguageDetector(minimum_score=0.2, fallback_language="es")
    output_df = detector.detect_languages_df(INPUT_DF, "input_text").sort_values(by=["input_text"])
    for col in output_df.columns:
        if col == "input_text_language_score":
            # Allow small tolerance for score differences between pycld2 and pycld3
            np.testing.assert_array_almost_equal(output_df[col].values, OUTPUT_DF[col].values, decimal=1)
        else:
            np.testing.assert_array_equal(output_df[col].values, OUTPUT_DF[col].values)

# Langid path test cases: (expected_name, expected_code, text)
LANGID_TEST_CASES = [
    ("English", "en", "Hello world, how are you doing today?"),
    ("French", "fr", "Bonjour le monde, comment allez-vous?"),
    ("Japanese", "ja", "こんにちは世界"),
    ("Hebrew", "he", "שלום עולם מה שלומך היום"),
    ("Indonesian", "id", "Selamat pagi, apa kabar hari ini?"),
]

# CLD path test cases: (expected_name, expected_code, text)
CLD_TEST_CASES = [
    ("English", "en", "This is a very long English text that exceeds the one hundred and forty character threshold to ensure CLD is used instead of langid for language detection."),
    ("French", "fr", "Ceci est un texte très long en français qui dépasse le seuil de cent quarante caractères pour s'assurer que CLD est utilisé pour la détection."),
    ("Japanese", "ja", "このオレはいずれ火影の名を受け継いで、先代のどの火影をも超えてやるんだ。これは非常に長い日本語のテキストで、百四十文字を超えています。私は絶対に諦めない、それが私の忍道だ。夢を持ち続けることが大切です。一歩一歩前に進んでいけば、必ず目標に到達できると信じています。頑張りましょう。未来は明るい。"),
    # Hebrew: CLD returns 'iw' which is remapped to 'he'
    ("Hebrew", "he", "שלום עולם זהו טקסט ארוך בעברית שצריך להיות יותר ממאה וארבעים תווים כדי להפעיל את נתיב הזיהוי של CLD במקום langid. אני אוהב לכתוב בעברית כי זו שפה יפה מאוד."),
    # Indonesian: CLD returns 'in' which is remapped to 'id'
    ("Indonesian", "id", "Selamat pagi semuanya, ini adalah teks panjang dalam bahasa Indonesia yang harus lebih dari seratus empat puluh karakter untuk memicu jalur deteksi CLD"),
]

@pytest.mark.parametrize("expected_name,expected_code,text", LANGID_TEST_CASES)
def test_langid_path(expected_name, expected_code, text):
    """Test language detection for short text (<=140 chars) using langid"""
    assert len(text) <= 140, "Text must be <= 140 chars for langid path"
    detector = LanguageDetector()
    lang_code, lang_name, lang_score = detector.detect_language_doc(text)
    assert lang_code == expected_code
    assert lang_name == expected_name
    assert lang_score is not None and lang_score > 0.5

@pytest.mark.parametrize("expected_name,expected_code,text", CLD_TEST_CASES)
def test_cld_path(expected_name, expected_code, text):
    """Test language detection for long text (>140 chars) using CLD (pycld2 or cld3)"""
    assert len(text) > 140, "Text must be > 140 chars for CLD path"
    detector = LanguageDetector()
    lang_code, lang_name, lang_score = detector.detect_language_doc(text)
    assert lang_code == expected_code
    assert lang_name == expected_name
    assert lang_score is not None and lang_score > 0.5

def test_language_scope_filtering_cld():
    """Test that CLD detection outside language_scope uses fallback"""
    detector = LanguageDetector(language_scope=["en", "fr"], fallback_language="en")
    german_text = "Guten Tag, wie geht es Ihnen heute? Ich hoffe, dass Sie einen schönen Tag haben. Das Wetter ist sehr schön heute und ich freue mich auf den Sommer."
    assert len(german_text) > 140
    lang_code, lang_name, lang_score = detector.detect_language_doc(german_text)
    assert lang_code == "en"
    assert lang_score is None

def test_language_scope_filtering_langid():
    """Test that langid is restricted to language_scope"""
    detector = LanguageDetector(language_scope=["en", "fr"], fallback_language="xx")
    text = "Guten Tag"
    assert len(text) <= 140
    lang_code, lang_name, lang_score = detector.detect_language_doc(text)
    # Langid is restricted to scope, returns en or fr (not de, not fallback)
    assert lang_code in ["en", "fr"]
    assert lang_score is not None

def test_minimum_score_filtering():
    """Test that low confidence detection uses fallback"""
    detector = LanguageDetector(minimum_score=0.99, fallback_language="xx")
    lang_code, lang_name, lang_score = detector.detect_language_doc("a")
    assert lang_code == "xx"
    assert lang_score is None

def test_empty_input():
    """Test that empty input returns empty results"""
    detector = LanguageDetector()
    lang_code, lang_name, lang_score = detector.detect_language_doc("")
    assert lang_code == ""
    assert lang_name == ""
    assert lang_score is None
