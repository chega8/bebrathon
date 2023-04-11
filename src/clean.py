import random
import argparse
import yaml
from loguru import logger
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

from os.path import join
import sys

sys.path.append(".")

from src.utils import date_handler, coord_preprocessing

input_path = join("data", "raw")
output_path = join("data", "cleaned")

logger.info("Cleaning data started")
logger.info("Input path: {}", input_path)
logger.info("Output path: {}", output_path)

date_columns_all = set(
    [
        "START_DATE",
        "END_DATE",
        "MEASUREMENT_DATE",
        "REQUEST_DATE",
        "COMPLETION_DATE",
        "HMANAF",
        "INITIAL_DETECTION_DATE",
        "DETECTION_DATE",
    ]
)

coord_columns_all = set(["ACCT_WGS84_X", "ACCT_WGS84_Y"])

dfs_desc = {
    "Records_TRAIN": {"train": True, "sort_key": ""},
    "Representations_TRAIN": {"train": True, "sort_key": "START_DATE"},
    "Consumptions_TRAIN": {"train": True, "sort_key": "MEASUREMENT_DATE"},
    "Requests_TRAIN": {"train": True, "sort_key": "REQUEST_DATE"},
    "PowerThefts_TRAIN": {"train": True, "sort_key": "HMANAF"},
    "Records_TEST": {"train": False, "sort_key": ""},
    "Representations_TEST": {"train": False, "sort_key": "START_DATE"},
    "Consumptions_TEST": {"train": False, "sort_key": "MEASUREMENT_DATE"},
    "Requests_TEST": {"train": False, "sort_key": "REQUEST_DATE"},
}


class DataFrame:
    def __init__(self, df_name, df_desc, input_path):
        self.df_name = df_name
        self.train = df_desc.get("train")
        self.sort_key = df_desc.get("sort_key")
        logger.info(f"Load {df_name}")
        self.df = pd.read_csv(join(input_path, f"{df_name}.csv"), sep="|", nrows=None)
        self.date_columns = date_columns_all & set(self.df.columns)
        self.coord_columns = coord_columns_all & set(self.df.columns)

    def sort_values(self):
        if self.sort_key:
            self.df = self.df.sort_values(self.sort_key)
            return self

    def get_unique_ACCT_NBRs(self):
        return self.df["ACCT_NBR"].unique()

    def date_handler(self):
        for col in self.date_columns:
            self.df[col] = self.df[col].apply(date_handler)
        return self

    def coord_handler(self):
        for col in self.coord_columns:
            self.df[col] = self.df[col].apply(coord_preprocessing)
        return self


class DataSet:
    def __init__(self, dfs_desc):
        self.dfs_desc = dfs_desc
        self.dataset = dict()

    def __getitem__(self, key):
        return self.dataset[key].df

    def load_dataframes(self, input_path):
        for df_name, values in self.dfs_desc.items():
            self.dataset[df_name] = DataFrame(df_name, values, input_path)
        return self

    def save_dataframes(self, new_data_path=output_path):
        for df_name, values in self.dfs_desc.items():
            path_to_save = join(new_data_path, f"{df_name}.pqt")
            logger.info(f"Save {path_to_save}")
            self.dataset[df_name].df.to_parquet(path_to_save, index=False)
        return self

    def date_handler(self):
        for df_name, values in self.dataset.items():
            self.dataset[df_name].date_handler()
        return self

    def sort_values(self):
        for df_name, values in self.dataset.items():
            self.dataset[df_name].sort_values()
        return self

    def coord_handler(self):
        for df_name, values in self.dataset.items():
            self.dataset[df_name].coord_handler()
        return self

    def le_ACCT_NBR(self):
        ACCT_NBRs = set()
        for df_name, values in self.dataset.items():
            ACCT_NBRs.update(self.dataset[df_name].get_unique_ACCT_NBRs())

        self.le = LabelEncoder()
        self.le.fit(list(ACCT_NBRs))
        del ACCT_NBRs

        for df_name, values in self.dataset.items():
            df = self.dataset[df_name].df
            df["ACCT_NBR"] = self.le.transform(df["ACCT_NBR"])
        return self

    def save_label_encoder(self, output_path):
        path_to_save = join(output_path, "ACCT_NBR_label_encoder.pkl")
        logger.info(f"Save {path_to_save}")
        with open(path_to_save, "wb") as f:
            pickle.dump(self.le, f)
        return self


def clean():
    """Clean dataset"""
    datasets = (
        DataSet(dfs_desc=dfs_desc)
        .load_dataframes(input_path)
        .date_handler()
        .sort_values()
        .coord_handler()
        .le_ACCT_NBR()
        .save_dataframes()
        .save_label_encoder(output_path)
    )


if __name__ == "__main__":
    clean()
    logger.info("Cleaning complete.")
