# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario


def test_langdetect_wili_benchmark(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key="TEST_TEXTPREPARATIONPLUGIN", scenario_id="TESTLANGUAGEDETECTIONONWILI")
