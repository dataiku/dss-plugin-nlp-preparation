# -*- coding: utf-8 -*-
from typing import List, AnyStr

import pandas as pd
import numpy as np


def unique_list(sequence: List) -> List:
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def clean_text_df(df: pd.DataFrame, dropna_columns: List[AnyStr] = None) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].str.strip().replace("", np.NaN)
    if dropna_columns is None:
        df = df.dropna()
    else:
        df = df.dropna(subset=dropna_columns)
    return df


def generate_unique(name: AnyStr, existing_names: List[AnyStr], prefix: AnyStr = None) -> AnyStr:
    """
    Generate a unique name among existing ones by suffixing a number. Can also add a prefix.
    """
    if prefix is not None:
        new_name = "{}_{}".format(prefix, name)
    else:
        new_name = name
    for i in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        if prefix is not None:
            new_name = "{}_{}_{}".format(prefix, name, i)
        else:
            new_name = "{}_{}".format(name, i)
    raise RuntimeError("Failed to generated a unique name")


def move_columns_after(df: pd.DataFrame, columns_to_move: List[AnyStr], after_column: AnyStr) -> pd.DataFrame:
    """
    Reorder columns by moving a list of columns after another
    """
    after_column_position = df.columns.get_loc(after_column) + 1
    reordered_columns = (
        df.columns[:after_column_position].tolist() + columns_to_move + df.columns[after_column_position:].tolist()
    )
    df.reindex(columns=reordered_columns)
    return df
