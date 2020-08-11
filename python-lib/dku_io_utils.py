# -*- coding: utf-8 -*-
import logging
import math
from typing import Callable, Dict

from tqdm import tqdm
import dataiku


def count_records(dataset: dataiku.Dataset) -> int:
    """
    Count the number of records of a dataset using the Dataiku dataset metrics API
    """
    metric_id = "records:COUNT_RECORDS"
    dataset_name = dataset.name.split(".")[1]
    partitions = dataset.read_partitions
    client = dataiku.api_client()
    project = client.get_project(dataiku.default_project_key())
    logging.info("Counting records of dataset: {}".format(dataset_name))
    if partitions is None or len(partitions) == 0:
        project.get_dataset(dataset_name).compute_metrics(metric_ids=[metric_id])
        metric = dataset.get_last_metric_values()
        record_count = dataiku.ComputedMetrics.get_value_from_data(metric.get_global_data(metric_id=metric_id))
        logging.info("Dataset contains {:d} records and is not partitioned".format(record_count))
    else:
        record_count = 0
        for partition in partitions:
            project.get_dataset(dataset_name).compute_metrics(partition=partition, metric_ids=[metric_id])
            metric = dataset.get_last_metric_values()
            record_count += dataiku.ComputedMetrics.get_value_from_data(
                metric.get_partition_data(partition=partition, metric_id=metric_id)
            )
        logging.info("Dataset contains {:d} records in partition(s) {}".format(record_count, partitions))
    return record_count


def process_dataset_chunks(
    input_dataset: dataiku.Dataset, output_dataset: dataiku.Dataset, func: Callable, chunksize: float = 10000, **kwargs
) -> None:
    """
    Read a dataset by chunks, process each dataframe chunk with a function and write back to another dataset.
    Automatically adds a tqdm progress bar and generic logging.
    """
    logging.info("Processing dataframe chunks of size {:d})...".format(chunksize))
    with output_dataset.get_writer() as writer:
        df_iterator = input_dataset.iter_dataframes(chunksize=chunksize, infer_with_pandas=False)
        len_iterator = math.ceil(count_records(input_dataset) / chunksize)
        for i, df in tqdm(enumerate(df_iterator), total=len_iterator):
            output_df = func(df=df, **kwargs)
            if i == 0:
                if output_dataset.writePartition is None or output_dataset.writePartition == "":
                    output_dataset.write_schema_from_dataframe(output_df, dropAndCreate=True)
                else:
                    output_dataset.write_schema_from_dataframe(output_df)
            writer.write_dataframe(output_df)
    logging.info("Processing dataframe chunks: Done!")


def set_column_description(
    input_dataset: dataiku.Dataset, output_dataset: dataiku.Dataset, column_description_dict: Dict,
) -> None:
    """
    Set column descriptions of the output dataset based on a dictionary of column descriptions
    and retains the column descriptions from the input dataset if the column name matches
    """
    input_dataset_schema = input_dataset.read_schema()
    output_dataset_schema = output_dataset.read_schema()
    input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_description_dict.get(output_col_name)
        if output_col_name in input_columns_names:
            matched_comment = [
                input_col_info.get("comment", "")
                for input_col_info in input_dataset_schema
                if input_col_info.get("name") == output_col_name
            ]
            if len(matched_comment) != 0:
                output_col_info["comment"] = matched_comment[0]
    output_dataset.write_schema(output_dataset_schema)
