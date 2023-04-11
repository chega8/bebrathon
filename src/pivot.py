import random
import argparse
import yaml
from loguru import logger
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from pandas import Period

from os.path import join
import sys

sys.path.append(".")

from src.utils import memory_reducer


input_path = join("data", "preprocessed")
output_path = join("data", "pivoted")

logger.info("Pivot started")
logger.info("Input path: {}", input_path)
logger.info("Output path: {}", output_path)


def load_preprocessed_data():
    Consumptions_TRAIN = pd.read_parquet(join(input_path, "Consumptions_TRAIN.pqt"))
    Consumptions_TEST = pd.read_parquet(join(input_path, "Consumptions_TEST.pqt"))
    Requests_TRAIN = pd.read_parquet(join(input_path, "Requests_TRAIN.pqt"))
    Requests_TEST = pd.read_parquet(join(input_path, "Requests_TEST.pqt"))
    Records_TRAIN = pd.read_parquet(join(input_path, "Records_TRAIN.pqt"))
    Records_TEST = pd.read_parquet(join(input_path, "Records_TEST.pqt"))
    PowerThefts = pd.read_parquet(join(input_path, "PowerThefts_TRAIN.pqt"))
    # Consumptions_mod = pd.read_parquet(join("data", "cleaned_data", "Consumptions_monthly.pqt"))

    Consumptions = pd.concat([Consumptions_TRAIN, Consumptions_TEST])
    Requests = pd.concat([Requests_TRAIN, Requests_TEST])
    Records = pd.concat([Records_TRAIN, Records_TEST])

    del Consumptions_TRAIN
    del Consumptions_TEST
    del Requests_TRAIN
    del Requests_TEST
    del Records_TRAIN
    del Records_TEST

    output = (Consumptions, Requests, Records, PowerThefts)
    return output


def prepare_preprocessed_data(Consumptions, Requests, Records, PowerThefts, Consumptions_mod=None):

    Consumptions = memory_reducer(Consumptions)
    Requests = memory_reducer(Requests)
    Records = memory_reducer(Records)
    PowerThefts = memory_reducer(PowerThefts)
    # Consumptions_mod = memory_reducer(Consumptions_mod)

    start_date = Period("2018-01")
    end_date = Period("2022-10")

    Consumptions = Consumptions[Consumptions["period"].between(start_date, end_date)]
    # Consumptions_mod = Consumptions_mod[Consumptions_mod["period"].between(start_date, end_date)]
    Requests = Requests[Requests["period"].between(start_date, end_date)]
    PowerThefts = PowerThefts[PowerThefts["period"].between(start_date, end_date)]

    output = (Consumptions, Requests, Records, PowerThefts)
    return output


def create_combinations_df(dfs):
    base_df = pd.concat([df[["ACCT_NBR", "SUCCESSOR", "period"]] for df in dfs])
    base_df = base_df.groupby(["ACCT_NBR", "SUCCESSOR"], as_index=False).agg(
        {"period": ["min", "max"]}
    )

    base_df.columns = ["ACCT_NBR", "SUCCESSOR", "min_month", "max_month"]
    base_df["period"] = base_df[["min_month", "max_month"]].apply(
        lambda x: pd.period_range(x[0], x[1]), axis=1
    )
    base_df = base_df[["ACCT_NBR", "SUCCESSOR", "period"]].explode("period")
    return base_df


def save_pivoted(pivoted_df):
    pivoted_df.to_parquet(join(output_path, "data.pqt"), index=False)
    logger.info("Save pivoted data")


def merge_all_tables(Consumptions, Requests, Records, PowerThefts, Consumptions_mod=None):
    """Merge all tables"""

    return (
        create_combinations_df([Consumptions, Requests, PowerThefts])
        .merge(Consumptions, on=["ACCT_NBR", "SUCCESSOR", "period"], how="left")
        # .merge(Consumptions_mod, on=["ACCT_NBR", "SUCCESSOR", "period"], how="left")
        .merge(Requests, on=["ACCT_NBR", "SUCCESSOR", "period"], how="left")
        .merge(Records, on=["ACCT_NBR", "SUCCESSOR"], how="left")
        .merge(PowerThefts, on=["ACCT_NBR", "SUCCESSOR", "period"], how="left")
    )


if __name__ == "__main__":
    processed = load_preprocessed_data()
    processed = prepare_preprocessed_data(*processed)

    pivoted = merge_all_tables(*processed)
    save_pivoted(pivoted)
    logger.info("Pivot complete.")
