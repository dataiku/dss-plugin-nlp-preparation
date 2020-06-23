import re
from nltk.tokenize import TreebankWordTokenizer

t = TreebankWordTokenizer()
def tokenize(x):
    return t.tokenize(x)

def preprocess(x, word_segmentation):
    """
    x is a string. The function returns string_or_token_list
    - string_or_token_list is the tokenized text if word_segmentation is true
    - string_or_token_list is str(x) otherwise
    """
    if not word_segmentation:
        # remove punctuation but not the apostrophe
        x = re.sub(r"[^\w\d'\s]+", ' ', str(x)) # pass str(x) to avoid non-string (like numbers) issues
        # tokenize
        string_or_token_list = t.tokenize(x)
    else:
        string_or_token_list = str(x)
    return string_or_token_list