# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario


def test_spellchecker_english_tweets(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TEST_SPELLCHECKER_TWEETS")


def test_spellchecker_multilingual_news(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TEST_SPELLCHECKER_NEWS")


def test_spellchecker_edge_cases(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TEST_SPELLCHECKER_EDGECASES")
