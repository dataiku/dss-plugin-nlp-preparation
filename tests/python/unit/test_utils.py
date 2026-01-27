# -*- coding: utf-8 -*-
"""Shared utilities for unit tests"""

import os

_RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "resource")

DICTIONARY_FOLDER_PATH = os.getenv("DICTIONARY_FOLDER_PATH", os.path.join(_RESOURCE_PATH, "dictionaries"))
STOPWORDS_FOLDER_PATH = os.getenv("STOPWORDS_FOLDER_PATH", os.path.join(_RESOURCE_PATH, "stopwords"))