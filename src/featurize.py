from loguru import logger
import pandas as pd

from os.path import join
import sys

sys.path.append(".")

from src.utils import memory_reducer

input_path = join("data", "pivoted")
output_path = join("data", "features")

logger.info("Featurize started")
logger.info("Input path: {}", input_path)
logger.info("Output path: {}", output_path)


def feature_engineering(data):
    df = data.copy()
    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    return df

def fillnas(data):
    df = data.copy()
    df["VOLTAGE"] = df["VOLTAGE"].map({"LOW": 0, "MED": 1})
    df = df.fillna(
        value={
            "consumption": 1,
            "newCon": 0,
            "discon": 0,
            "compl": 0,
            "recon": 0,
            "reprPause": 0,
            "reprChange": 0,
            "theft": 0,
            "VOLTAGE": 0,
            "PARNO": 0,
            "XRHSH": 1,
            "CONTRACT_CAPACITY": 12,
            "ACCT_CONTROL": 0,
            "ACCT_WGS84_X": 0,
            "ACCT_WGS84_Y": 0,
        }
    )
    return df


def load_pivoted_data():
    data_path = join(input_path, "data.pqt")
    data = pd.read_parquet(data_path)
    return data


def load_test_users():
    data_path = join(input_path, "test_users.pqt")
    test_users = pd.read_parquet(data_path).drop_duplicates()
    test_users["dataset"] = 00
    return test_users


def preprocess_pivoted_data(data, test_users):
    data = feature_engineering(data)
    data = fillnas(data)
    data = memory_reducer(data)

    data = data.merge(test_users, on=["ACCT_NBR", "SUCCESSOR"], how="outer")

    data = fillnas(data)
    data = memory_reducer(data)
    return data


def split_data(data):
    data.loc[(data["dataset"].isnull()) & (data["ACCT_NBR"] % 3 == 0), "dataset"] = 10
    data.loc[(data["dataset"].isnull()) & (data["ACCT_NBR"] % 3 == 1), "dataset"] = 20
    data.loc[(data["dataset"].isnull()) & (data["ACCT_NBR"] % 3 == 2), "dataset"] = 30
    logger.info(f"Split counts: {data['dataset'].value_counts()}")
    return memory_reducer(data)


def save_dataset(dataset):
    logger.info("Save features")
    path_to_save = join(output_path, "data.pqt")
    dataset.to_parquet(path_to_save)


def create_dataset(data, window_size=12):
    index_cols = ["ACCT_NBR", "SUCCESSOR", "period"]
    time_features_names = [
        "consumption",
        "newCon",
        "discon",
        "compl",
        "recon",
        "reprPause",
        "reprChange",
    ]

    not_time_features_names = [
        "cons_succ_mean",
        "cons_acct_mean",
        "theft_succ_sum",
        "theft_acct_nbr_sum",
        "theft",
    ]
    extra_features = [
        "VOLTAGE",
        "PARNO",
        "XRHSH",
        "CONTRACT_CAPACITY",
        "ACCT_CONTROL",
        "ACCT_WGS84_X",
        "ACCT_WGS84_Y",
        "year",
        "month",
        "dataset",
    ]

    df = data.sort_values("period").copy()
    groups_successor = df.groupby(["ACCT_NBR", "SUCCESSOR"], as_index=False)
    groups_acct_nbr = df.groupby("ACCT_NBR", as_index=False)

    df["cons_succ_mean"] = groups_successor["consumption"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["cons_acct_mean"] = groups_acct_nbr["consumption"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["consumption"] = (df["consumption"] / df["cons_succ_mean"]).fillna(0)
    df.loc[df["consumption"] < 0, "consumption"] = -1
    logger.info("DONE: cons_succ_mean and cons_acct_mean")
    df["theft_succ_sum"] = groups_successor["theft"].cumsum()
    df["theft_acct_nbr_sum"] = groups_acct_nbr["theft"].cumsum()

    logger.info("DONE: theft_succ_sum and theft_acct_nbr_sum")
    df = memory_reducer(df)
    lag_features_names = list()
    for lag in range(1, window_size):
        logger.info(f"{lag=}", end=" ")
        lag_features = (
            groups_successor[time_features_names].shift(lag).fillna(-2).add_suffix(f"_{lag}")
        )
        memory_reducer(lag_features)
        df = pd.concat([df, lag_features], axis=1)
        lag_features_names += lag_features.columns.tolist()

    logger.info("DONE: lag_features")
    columns_to_save = (
        index_cols
        + not_time_features_names
        + time_features_names
        + lag_features_names
        + extra_features
    )
    df = df[columns_to_save]
    return df


if __name__ == "__main__":
    pivoted = load_pivoted_data()
    test_users = load_test_users()

    dataset = preprocess_pivoted_data(pivoted, test_users)
    dataset = split_data(dataset)
    dataset = create_dataset(dataset, window_size=12)
    save_dataset(dataset)
    logger.info("Featurize complete.")
