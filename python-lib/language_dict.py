# -*- coding: utf-8 -*-
"""Module with constants defining the language support of underlying NLP libraries"""

SUPPORTED_LANGUAGES_PYCLD3 = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "an": "Aragonese",
    "hy": "Armenian",
    "as": "Assamese",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "br": "Breton",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "km": "Central Khmer",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "dz": "Dzongkha",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "fo": "Faroese",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ht": "Haitian",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "rw": "Kinyarwanda",
    "ky": "Kirghiz",
    "ko": "Korean",
    "ku": "Kurdish",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mn": "Mongolian",
    "ne": "Nepali",
    "se": "Northern Sami",
    "nb": "Norwegian Bokmål",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "ny": "Nyanja",
    "oc": "Occitan",
    "or": "Oriya",
    "pa": "Panjabi",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ps": "Pushto",
    "qu": "Quechua",
    "ro": "Romanian",
    "ru": "Russian",
    "sm": "Samoan",
    "gd": "Scottish Gaelic",
    "sr": "Serbian",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "st": "Southern Sotho",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "tg": "Tajik",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "cy": "Welsh",
    "fy": "Western Frisian",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu",
}
"""dict: Languages supported by pycld3

Dictionary with ISO 639-1 language code (key) and language name (value)
"""

SUPPORTED_LANGUAGES_PYCLD3_NOT_LANGID = [
    "fy",
    "gd",
    "ha",
    "ig",
    "mi",
    "my",
    "ny",
    "sd",
    "sm",
    "sn",
    "so",
    "st",
    "su",
    "tg",
    "uz",
    "yi",
    "yo",
]
"""list: Subset of languages codes supported by pycld3 but not langid"""

LANGUAGE_REMAPPING_PYCLD3_LANGID = {"iw": "he", "co": "it", "ji": "yi", "in": "id"}
"""dict: Rare cases of inconsistent language codes between pycld3 and langid"""

SUPPORTED_LANGUAGES_SYMSPELL = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese (simplified)",
}
"""dict: SymSpell dictionaries included in this plugin

Dictionary with ISO 639-1 language code (key) and language name (value)
"""

SUPPORTED_LANGUAGES_SPACY = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "kn": "Kannada",
    "lb": "Luxembourgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nb": "Norwegian Bokmål",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yo": "Yoruba",
    "zh": "Chinese (simplified)",
}
"""dict: Languages supported by spaCy: https://spacy.io/usage/models#languages

Dictionary with ISO 639-1 language code (key) and language name (value).
Japanese, Korean and Ukrainian not included because they require system-level operations
"""

SPACY_LANGUAGE_MODELS = {
    "en": "en_core_web_sm",  # OntoNotes
    "es": "es_core_news_sm",  # Wikipedia
    "zh": "zh_core_web_sm",  # OntoNotes
    "xx": "xx_ent_wiki_sm",  # Wikipedia
    "pl": "nb_core_news_sm",  # NorNE
    "fr": "fr_core_news_sm",  # Wikipedia
    "de": "de_core_news_sm",  # OntoNotes
}
"""dict: Mapping between ISO 639-1 language code and spaCy model identifiers

Models with Creative Commons licenses are not included because this plugin is licensed under Apache-2
"""
