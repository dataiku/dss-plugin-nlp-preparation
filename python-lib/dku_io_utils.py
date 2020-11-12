# -*- coding: utf-8 -*-
"""Module with read/write utility functions based on the Dataiku API"""

import logging
import math
from time import time
from typing import Callable, Dict

from tqdm import tqdm
import dataiku


def count_records(dataset: dataiku.Dataset) -> int:
    """Count the number of records of a dataset using the Dataiku dataset metrics API

    Args:
        dataset: dataiku.Dataset instance

    Returns:
        Number of records

    """
    metric_id = "records:COUNT_RECORDS"
    partitions = dataset.read_partitions
    client = dataiku.api_client()
    project = client.get_project(dataset.project_key)
    record_count = 0
    logging.info(f"Counting records of dataset: {dataset.name}...")
    if partitions is None or len(partitions) == 0:
        project.get_dataset(dataset.short_name).compute_metrics(metric_ids=[metric_id])
        metric = dataset.get_last_metric_values()
        record_count = dataiku.ComputedMetrics.get_value_from_data(metric.get_global_data(metric_id=metric_id))
        logging.info(f"Dataset {dataset.name} contains {record_count:d} records and is not partitioned")
    else:
        for partition in partitions:
            project.get_dataset(dataset.short_name).compute_metrics(partition=partition, metric_ids=[metric_id])
            metric = dataset.get_last_metric_values()
            record_count += dataiku.ComputedMetrics.get_value_from_data(
                metric.get_partition_data(partition=partition, metric_id=metric_id)
            )
        logging.info(f"Dataset {dataset.name} contains {record_count:d} records in partition(s) {partitions}")
    return record_count


def process_dataset_chunks(
    input_dataset: dataiku.Dataset, output_dataset: dataiku.Dataset, func: Callable, chunksize: float = 1000, **kwargs
) -> None:
    """Read a dataset by chunks, process each dataframe chunk with a function and write back to another dataset.

    Pass keyword arguments to the function, adds a tqdm progress bar and generic logging.
    Directly write chunks to the output_dataset, so that only one chunk needs to be processed in-memory at a time.

    Args:
        input_dataset: Input dataiku.Dataset instance
        output_dataset: Output dataiku.Dataset instance
        func: The function to apply to the `input_dataset` by chunks of pandas.DataFrame
            This function must take a pandas.DataFrame as first input argument,
            and output another pandas.DataFrame
        chunksize: Number of rows of each chunk of pandas.DataFrame fed to `func`
        **kwargs: Optional keyword arguments fed to `func`

    Raises:
        ValueError: If the input dataset is empty or if pandas cannot read it without type inference

    """
    input_count_records = count_records(input_dataset)
    if input_count_records == 0:
        raise ValueError("Input dataset has no records")
    logging.info(f"Processing dataset {input_dataset.name} of {input_count_records} rows by chunks of {chunksize}...")
    start = time()
    # First, initialize output schema if not present. Required to show the real error if `iter_dataframes` fails.
    if not output_dataset.read_schema(raise_if_empty=False):
        df = input_dataset.get_dataframe(limit=5, infer_with_pandas=False)
        output_df = func(df=df, **kwargs)
        output_dataset.write_schema_from_dataframe(output_df)
    with output_dataset.get_writer() as writer:
        df_iterator = input_dataset.iter_dataframes(chunksize=chunksize, infer_with_pandas=False)
        len_iterator = math.ceil(input_count_records / chunksize)
        for i, df in tqdm(enumerate(df_iterator), total=len_iterator, unit="chunk", mininterval=1.0):
            output_df = func(df=df, **kwargs)
            if i == 0:
                output_dataset.write_schema_from_dataframe(
                    output_df, dropAndCreate=bool(not output_dataset.writePartition)
                )
            writer.write_dataframe(output_df)
    logging.info(
        f"Processing dataset {input_dataset.name} of {input_count_records} rows: Done in {time() - start:.2f} seconds."
    )


def set_column_descriptions(
    output_dataset: dataiku.Dataset, column_descriptions: Dict, input_dataset: dataiku.Dataset = None
) -> None:
    """Set column descriptions of the output dataset based on a dictionary of column descriptions

    Retain the column descriptions from the input dataset if the column name matches.

    Args:
        output_dataset: Output dataiku.Dataset instance
        column_descriptions: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions

    """
    output_dataset_schema = output_dataset.read_schema()
    input_dataset_schema = []
    input_columns_names = []
    if input_dataset is not None:
        input_dataset_schema = input_dataset.read_schema()
        input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_descriptions.get(output_col_name)
        if output_col_name in input_columns_names:
            matched_comment = [
                input_col_info.get("comment", "")
                for input_col_info in input_dataset_schema
                if input_col_info.get("name") == output_col_name
            ]
            if len(matched_comment) != 0:
                output_col_info["comment"] = matched_comment[0]
    output_dataset.write_schema(output_dataset_schema)
