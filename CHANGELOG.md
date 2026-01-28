# Changelog

## [Version 1.4.0](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.4.0) - Python support release - 2026-01
- ✨ Added Python 3.10, 3.11, 3.12 and 3.13 support
- 🐛 Fixed code env building on Python 3,6, 3.7, 3.8 and 3.9

## [Version 1.3.1](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.3.1) - Bugfix release - 2025-01
- 🐛 Fixed code env building on Python 3.7 and 3.8

## [Version 1.3.0](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.3.0) - New feature - 2023-04
- ✨ Added Python 3.8 and 3.9 support

## [Version 1.2.2](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.2.2) - Bugfix release - 2022-07
- Fix sudachipy version not being compatible with Python 3.6 anymore

## [Version 1.2.1](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.2.1) - Bugfix release - 2021-06
- ✨ Improved Japanese stopwords
- 🐛 Add explicit UI for languages which do not support lemmatization

## [Version 1.2.0](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.2.0) - New feature and bugfix release - 2021-04
- ✨ Added Python 3.7 and Japanese support
- 🐛 Fixed silent failure when tokenizing long text (> 1 million characters)
- 💄 Improved recipe interface loading time, enhanced logging and column descriptions

## [Version 1.1.1](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.1.1) - Bugfix release - 2020-12
- 🐛 Fixed macedonian support
- 💚 Added integration tests

## [Version 1.1.0](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.1.0) - New feature release - 2020-12
- ✨ Text cleaning recipe to tokenize, filter and lemmatize text data in 58 languages
- ✅ Stopwords peer-reviewed by native speakers
- 💄 Enhancements to the UX, the logging and the tokenization library

## [Version 1.0.1](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.0.1) - Bugfix release - 2020-09
- 👹 Removed Japanese support because of installation issues with the [User Isolation Framework](https://doc.dataiku.com/dss/latest/user-isolation/capabilities/index.html)

## [Version 1.0.0](https://github.com/dataiku/dss-plugin-nlp-preparation/releases/tag/v1.0.0) - Initial release - 2020-09
- 🌎 🌍 🌏  Recipe to detect dominant languages among 114 languages
- 🧐 Recipe to identify and correct misspellings in 36 languages

