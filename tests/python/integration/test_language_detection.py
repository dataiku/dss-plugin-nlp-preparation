# -*- coding: utf-8 -*-
import pytest
import logging

from dku_plugin_test_utils import dss_scenario

pytestmark = pytest.mark.usefixtures("plugin", "dss_target")

test_kwargs = {
    "user": "user1",
    "project_key": "TEST_TEXTPREPARATIONPLUGIN",
    "logger": logging.getLogger("dss-plugin-test.nlp-preparation.test_language_detection"),
}


def test_langdetect_wili_benchmark(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="TESTLANGUAGEDETECTIONONWILI", **test_kwargs)
