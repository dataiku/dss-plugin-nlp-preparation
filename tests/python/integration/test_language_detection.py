# -*- coding: utf-8 -*-
import pytest
import logging

from dku_plugin_test_utils import dss_scenario

pytestmark = pytest.mark.usefixtures("plugin", "dss_target")

test_kwargs = {
    "user": "default",
    "project_key": "TEST_TEXTPREPARATIONPLUGIN",
    "logger": logging.getLogger("dss-plugin-test.nlp-preparation.test_language_detection"),
}


def test_langdetect_wili_benchmark(user_clients):
    dss_scenario.run(scenario_id="TESTLANGUAGEDETECTIONONWILI", client=user_clients["default"], **test_kwargs)
