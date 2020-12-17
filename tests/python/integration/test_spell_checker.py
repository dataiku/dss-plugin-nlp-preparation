# -*- coding: utf-8 -*-
import pytest
import logging

from dku_plugin_test_utils import dss_scenario

pytestmark = pytest.mark.usefixtures("plugin", "dss_target")

test_kwargs = {
    "user": "default",
    "project_key": "TEST_TEXTPREPARATIONPLUGIN",
    "logger": logging.getLogger("dss-plugin-test.nlp-preparation.test_spell_checker"),
}


def test_spellchecker_english_tweets(user_clients):
    dss_scenario.run(scenario_id="TEST_SPELLCHECKER_TWEETS", client=user_clients["default"], **test_kwargs)


def test_spellchecker_multilingual_news(user_clients):
    dss_scenario.run(scenario_id="TEST_SPELLCHECKER_NEWS", client=user_clients["default"], **test_kwargs)


def test_spellchecker_edge_cases(user_clients):
    dss_scenario.run(scenario_id="TEST_SPELLCHECKER_EDGECASES", client=user_clients["default"], **test_kwargs)
