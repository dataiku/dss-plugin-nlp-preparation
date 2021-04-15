# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario


def test_textcleaning_english_tweets(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TEST_TEXTCLEANING_TWEETS")


def test_textcleaning_multilingual_news(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TEST_TEXTCLEANING_NEWS")


def test_textcleaning_edge_cases(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TEST_TEXTCLEANING_EDGECASES")
