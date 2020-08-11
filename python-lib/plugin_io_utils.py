# -*- coding: utf-8 -*-
from typing import List, AnyStr


def generate_unique(name: AnyStr, existing_names: List[AnyStr], prefix: AnyStr = None) -> AnyStr:
    """
    Generate a unique name among existing ones by suffixing a number. Can also add a prefix.
    """
    new_name = "{}_{}".format(prefix, name)
    for i in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = "{}_{}_{}".format(prefix, name, i)
    raise Exception("Failed to generated a unique name")
