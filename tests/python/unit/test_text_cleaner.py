# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd

from spacy_tokenizer import MultilingualTokenizer
from text_cleaner import UnicodeNormalization, TextCleaner
from test_utils import STOPWORDS_FOLDER_PATH


def test_clean_df_english():
    input_df = pd.DataFrame({"input_text": ["Hi, I have two apples costing 3$ 😂    \n and unicode has #snowpersons ☃"]})
    token_filters = {"is_punct", "is_stop", "like_num", "is_symbol", "is_currency", "is_emoji"}
    text_cleaner = TextCleaner(tokenizer=MultilingualTokenizer(), token_filters=token_filters, lemmatization=True)
    output_df = text_cleaner.clean_df(df=input_df, text_column="input_text", language="en")
    cleaned_text_column = list(text_cleaner.output_column_descriptions.keys())[0]
    cleaned_text = output_df[cleaned_text_column][0]
    expected_cleaned_text = "apple cost unicode #snowpersons"
    assert cleaned_text == expected_cleaned_text


def test_clean_df_multilingual():
    input_df = pd.DataFrame(
        {
            "input_text": [
                "I did a 10k run this morning at 6h34 follow me @superRunnerdu95 didn't I?",
                "Nous cherchâmes des informations sur https://www.google.com/ le 03/11/2046 l'aventures",
                "#Barcelona Fútbol es la vida me@me.com ℌ ①",
            ],
            "language": ["en", "fr", "es"],
        }
    )
    token_filters = {"is_stop", "is_measure", "is_datetime", "like_url", "like_email", "is_username", "is_hashtag"}
    text_cleaner = TextCleaner(
        tokenizer=MultilingualTokenizer(stopwords_folder_path=STOPWORDS_FOLDER_PATH),
        token_filters=token_filters,
        lemmatization=True,
        lowercase=False,
        unicode_normalization=UnicodeNormalization.NFKD,
    )
    output_df = text_cleaner.clean_df(df=input_df, text_column="input_text", language_column="language")
    cleaned_text_column = list(text_cleaner.output_column_descriptions.keys())[0]
    cleaned_texts = output_df[cleaned_text_column].values.tolist()
    expected_cleaned_texts = [
        "run morning follow not ?",
        "chercher information aventurer",
        "Fútbol vida H 1",
    ]
    assert cleaned_texts == expected_cleaned_texts
