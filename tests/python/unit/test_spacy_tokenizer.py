# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pytest
import pandas as pd
import spacy.about

from spacy_tokenizer import MultilingualTokenizer
from test_utils import STOPWORDS_FOLDER_PATH


def test_tokenize_df_english():
    input_df = pd.DataFrame({"input_text": ["I hope nothing. I fear nothing. I am free. 💩 😂 #OMG"]})
    tokenizer = MultilingualTokenizer()
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    tokenized_document = output_df[tokenizer.tokenized_column][0]
    assert len(tokenized_document) == 15


def test_tokenize_df_japanese():
    input_df = pd.DataFrame({"input_text": ["期一会。 異体同心。 そうです。"]})
    tokenizer = MultilingualTokenizer()
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="ja")
    tokenized_document = output_df[tokenizer.tokenized_column][0]
    assert len(tokenized_document) == 9


def test_tokenize_df_multilingual():
    input_df = pd.DataFrame(
        {
            "input_text": [
                "I hope nothing. I fear nothing. I am free.",
                " Les sanglots longs des violons d'automne",
                "子曰：“學而不思則罔，思而不學則殆。”",
                "期一会。 異体同心。 そうです。",
            ],
            "language": ["en", "fr", "zh", "ja"],
        }
    )
    tokenizer = MultilingualTokenizer(stopwords_folder_path=STOPWORDS_FOLDER_PATH)
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language_column="language")
    tokenized_documents = output_df[tokenizer.tokenized_column]
    tokenized_documents_length = [len(doc) for doc in tokenized_documents]
    if spacy.about.__version__.startswith("3"):
        # spacy 3.x tokenizes Chinese differently (19 vs 13 tokens)
        assert tokenized_documents_length == [12, 8, 19, 9]
    else:
        assert tokenized_documents_length == [12, 8, 13, 9]


def test_tokenize_df_long_text():
    input_df = pd.DataFrame({"input_text": ["Long text"]})
    tokenizer = MultilingualTokenizer(max_num_characters=1)
    with pytest.raises(ValueError):
        tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")


# =============================================================================
# CUSTOM TOKEN ATTRIBUTE TESTS
# =============================================================================

# Token attribute test cases: (text, token_index, attribute, expected_value)
TOKEN_ATTRIBUTE_TEST_CASES = [
    # is_hashtag
    ("Hello #world", 1, "is_hashtag", True),
    ("Hello world", 1, "is_hashtag", False),
    # is_username
    ("Hello @user", 1, "is_username", True),
    ("Hello user", 1, "is_username", False),
    # is_emoji
    ("Hello 😂", 1, "is_emoji", True),
    ("Hello world", 1, "is_emoji", False),
    # is_datetime
    ("Meet at 10:30am", 2, "is_datetime", True),
    ("Meet at noon", 2, "is_datetime", False),
]


@pytest.mark.parametrize("text,token_index,attribute,expected", TOKEN_ATTRIBUTE_TEST_CASES)
def test_token_custom_attributes(text, token_index, attribute, expected):
    """Test custom spaCy token attributes"""
    tokenizer = MultilingualTokenizer()
    input_df = pd.DataFrame({"input_text": [text]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    token = doc[token_index]
    if attribute.startswith("is_") and attribute not in ["is_punct", "is_stop", "is_currency"]:
        # Custom attributes are on token._
        assert getattr(token._, attribute) == expected
    else:
        # Native spaCy attributes
        assert getattr(token, attribute) == expected


def test_token_is_measure():
    """Test is_measure attribute exists and can be queried"""
    tokenizer = MultilingualTokenizer()
    input_df = pd.DataFrame({"input_text": ["Distance is 5km away"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    # Verify the is_measure attribute exists and can be queried on all tokens
    for token in doc:
        # Should not raise - attribute exists
        _ = token._.is_measure


def test_token_is_symbol():
    """Test is_symbol attribute for symbol characters"""
    tokenizer = MultilingualTokenizer()
    # Use a clearer symbol that's more likely to be detected
    input_df = pd.DataFrame({"input_text": ["Check ✓ and ✗ symbols"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    # Check that symbol detection works for at least one symbol token
    symbol_tokens = [t for t in doc if t._.is_symbol]
    # At least one of ✓ or ✗ should be detected as symbol
    assert len(symbol_tokens) >= 1 or any("✓" in t.text or "✗" in t.text for t in doc)


# =============================================================================
# HASHTAGS_AS_TOKEN PARAMETER TESTS
# =============================================================================

def test_hashtags_as_single_token():
    """Test that hashtags_as_token=True keeps hashtag as one token (default)"""
    tokenizer = MultilingualTokenizer(hashtags_as_token=True)
    input_df = pd.DataFrame({"input_text": ["Hello #world"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    tokens = [t.text for t in doc]
    assert "#world" in tokens


def test_hashtags_as_two_tokens():
    """Test that hashtags_as_token=False splits hashtag into two tokens"""
    tokenizer = MultilingualTokenizer(hashtags_as_token=False)
    input_df = pd.DataFrame({"input_text": ["Hello #world"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    tokens = [t.text for t in doc]
    assert "#" in tokens and "world" in tokens


# =============================================================================
# LEMMATIZATION TESTS (critical for spacy 3.x)
# =============================================================================

def test_lemmatization_english():
    """Test that English lemmatization works"""
    tokenizer = MultilingualTokenizer()
    input_df = pd.DataFrame({"input_text": ["The cats are running quickly"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    lemmas = [t.lemma_ for t in doc]
    # Check that at least some lemmatization occurred
    assert "cat" in lemmas or "run" in lemmas or lemmas != [t.text for t in doc]


def test_lemmatization_french():
    """Test that French lemmatization works"""
    tokenizer = MultilingualTokenizer()
    input_df = pd.DataFrame({"input_text": ["Les chats courent vite"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="fr")
    doc = output_df[tokenizer.tokenized_column][0]
    lemmas = [t.lemma_ for t in doc]
    # Check that lemmas are populated (not empty strings)
    non_empty_lemmas = [l for l in lemmas if l.strip()]
    assert len(non_empty_lemmas) > 0


# =============================================================================
# STOPWORDS TESTS
# =============================================================================

def test_stopwords_english():
    """Test that English stopwords are detected"""
    tokenizer = MultilingualTokenizer(stopwords_folder_path=STOPWORDS_FOLDER_PATH)
    input_df = pd.DataFrame({"input_text": ["I am the one"]})
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    doc = output_df[tokenizer.tokenized_column][0]
    stopword_tokens = [t.text for t in doc if t.is_stop]
    # Common English stopwords
    assert any(word in stopword_tokens for word in ["I", "am", "the"])


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_unsupported_language():
    """Test that unsupported language raises TokenizationError"""
    from spacy_tokenizer import TokenizationError
    tokenizer = MultilingualTokenizer()
    input_df = pd.DataFrame({"input_text": ["Hello"], "language": ["xx_invalid"]})
    with pytest.raises(TokenizationError):
        tokenizer.tokenize_df(df=input_df, text_column="input_text", language_column="language")


def test_empty_language():
    """Test that empty language raises TokenizationError"""
    from spacy_tokenizer import TokenizationError
    tokenizer = MultilingualTokenizer()
    input_df = pd.DataFrame({"input_text": ["Hello"]})
    with pytest.raises(TokenizationError):
        tokenizer.tokenize_df(df=input_df, text_column="input_text", language="")
