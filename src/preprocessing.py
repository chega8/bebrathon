from loguru import logger
import pandas as pd
import numpy as np

from os.path import join
import sys

sys.path.append(".")

from src.utils import memory_reducer

project_path = "/content/drive/MyDrive/DEDDIE_Datathon"
input_path = join(project_path, "npavin_data", "1.cleaned_data")
output_path = join(project_path, "npavin_data", "2.preprocessed_data")

logger.info("Preprocessing started")
logger.info("Input path: {}", input_path)
logger.info("Output path: {}", output_path)

dfs_name = [
    "Records_TRAIN",
    "Representations_TRAIN",
    "Consumptions_TRAIN",
    "Requests_TRAIN",
    "PowerThefts_TRAIN",
    "Records_TEST",
    "Representations_TEST",
    "Consumptions_TEST",
    "Requests_TEST",
    # "Consumptions_monthly",
]


class Coords:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat
        self.n_points = len(lon)

    def sub_median(self):
        self.lon_median = np.median(self.lon)
        self.lat_median = np.median(self.lat)
        return (self.lon - self.lon_median, self.lat - self.lat_median)

    def add_median(self):
        return (self.lon + self.lon_median, self.lat + self.lat_median)

    def normalize(self):
        self.lon, self.lat = self.sub_median()
        return self

    def bias(self, bias):
        self.lon = self.lon + bias[1]
        self.lat = self.lat + bias[0]
        return self

    def scale(self, scale):
        self.lon, self.lat = self.sub_median()
        self.lon = self.lon / scale[1]
        self.lat = self.lat / scale[0]
        self.lon, self.lat = self.add_median()
        return self

    def mirroring(self):
        self.lon = -self.lon
        return self

    def rotate(self, angle):
        import math

        self.lon, self.lat = self.sub_median()

        angle = math.radians(angle)
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        self.lon, self.lat = (
            self.lon * cos_angle - self.lat * sin_angle,
            self.lon * sin_angle + self.lat * cos_angle,
        )
        self.lon, self.lat = self.add_median()
        return self


def load_cleaned_data():
    datasets = dict()
    for df_name in dfs_name:
        data = pd.read_parquet(join(input_path, f"{df_name}.pqt"))
        data = memory_reducer(data)
        datasets[df_name] = data
        del data
    return datasets


def prepare_raw_datasets(datasets):
    df = datasets["Records_TRAIN"][
        ["ACCT_NBR", "ACCT_WGS84_X", "ACCT_WGS84_Y", "CONTRACT_CAPACITY"]
    ]
    logger.info(f"{len(df)}")
    df = df.dropna(subset=["ACCT_WGS84_X", "ACCT_WGS84_Y"]).drop_duplicates().copy()
    logger.info(f"len(df)")

    coords = Coords(lon=df["ACCT_WGS84_X"].values.copy(), lat=df["ACCT_WGS84_Y"].values.copy())

    coords = (
        coords.normalize()
        .mirroring()
        .scale([1.2, 1])
        .rotate(221.5)
        .bias([23.726, 37.939])
        .scale([0.9, 1.1])
    )

    df["lon"] = coords.lon
    df["lat"] = coords.lat
    df = df.set_index("ACCT_NBR")


def consumption_preproc(data):
    df = data[["ACCT_NBR", "SUCCESSOR", "CSS_MS_HS_USE", "MEASUREMENT_DATE"]].copy()
    df["period"] = pd.to_datetime(df["MEASUREMENT_DATE"]).dt.to_period("M")
    df = df.groupby(["ACCT_NBR", "SUCCESSOR", "period"], as_index=False)["CSS_MS_HS_USE"].sum()
    df = df.rename(columns={"CSS_MS_HS_USE": "consumption"})
    return df


def requests_preproc(data):
    requests = data[["ACCT_NBR", "SUCCESSOR", "REQUEST_TYPE", "REQUEST_DATE"]]
    completes = data[["ACCT_NBR", "SUCCESSOR", "COMPL_REQUEST_STATUS", "COMPLETION_DATE"]]

    requests.columns = ["ACCT_NBR", "SUCCESSOR", "event", "period"]
    completes.columns = ["ACCT_NBR", "SUCCESSOR", "event", "period"]

    requests = requests.dropna(subset=["period"])
    completes = completes.dropna(subset=["period"]).fillna({"event": "compl"})

    requests["period"] = pd.to_datetime(requests["period"]).dt.to_period("M")
    completes["period"] = pd.to_datetime(completes["period"]).dt.to_period("M")

    df = pd.concat([requests, completes]).sort_values("period").reset_index(drop=False)
    event_dtype = pd.CategoricalDtype(
        categories=["newCon", "discon", "compl", "recon", "reprPause", "reprChange"]
    )
    df["event"] = df["event"].astype(event_dtype)
    dummy = pd.get_dummies(df["event"])
    df = df.join(dummy)
    df = df.groupby(["ACCT_NBR", "SUCCESSOR", "period"])[
        ["newCon", "discon", "compl", "recon", "reprPause", "reprChange"]
    ].sum()
    return df.reset_index()


def records_preproc(data):

    return data


def power_thefts_preproc(data):
    df = data[["ACCT_NBR", "SUCCESSOR", "INITIAL_DETECTION_DATE", "DETECTION_DATE"]].copy()
    df["INITIAL_DETECTION_DATE"] = pd.to_datetime(df["INITIAL_DETECTION_DATE"]).dt.to_period("M")
    df["DETECTION_DATE"] = pd.to_datetime(df["DETECTION_DATE"]).dt.to_period("M")
    df = df.dropna()
    df["period"] = df[["INITIAL_DETECTION_DATE", "DETECTION_DATE"]].apply(
        lambda x: pd.period_range(x[0], x[1]), axis=1
    )
    df = df[["ACCT_NBR", "SUCCESSOR", "period"]].copy()
    df = df.explode("period")
    df["theft"] = 1
    return df


def get_test_users(datasets):
    test_1 = (
        datasets["Records_TEST"][["ACCT_NBR", "SUCCESSOR"]]
        .set_index(["ACCT_NBR", "SUCCESSOR"])
        .drop_duplicates()
    )
    test_2 = (
        datasets["Representations_TEST"][["ACCT_NBR", "SUCCESSOR"]]
        .set_index(["ACCT_NBR", "SUCCESSOR"])
        .drop_duplicates()
    )
    test_3 = (
        datasets["Consumptions_TEST"][["ACCT_NBR", "SUCCESSOR"]]
        .set_index(["ACCT_NBR", "SUCCESSOR"])
        .drop_duplicates()
    )
    test_4 = (
        datasets["Requests_TEST"][["ACCT_NBR", "SUCCESSOR"]]
        .set_index(["ACCT_NBR", "SUCCESSOR"])
        .drop_duplicates()
    )

    test_1["Records"] = 1
    test_2["Representations"] = 2
    test_3["Consumptions"] = 4
    test_4["Requests"] = 8

    test_users = (
        test_1.join(test_2, how="outer")
        .join(test_3, how="outer")
        .join(test_4, how="outer")
        .fillna(0)
        .astype(int)
    )

    # print('Here we can understand, where our datasets intercept:')
    # display(test_users.sum(axis=1).value_counts())

    test_users["Records"] /= 1
    test_users["Representations"] /= 2
    test_users["Consumptions"] /= 4
    test_users["Requests"] /= 8
    # print('\nAmount of records in each table:')
    # display(test_users.sum())
    # print('\nAmount of absent values in each tables:')
    # display((test_users==0).sum())
    return test_users


def save_test_users(test_users):
    (
        test_users.reset_index()[["ACCT_NBR", "SUCCESSOR"]]
        .drop_duplicates()
        .to_parquet(join(project_path, "npavin_data", "3.pivoted_data", "test_users.pqt"))
    )


def preprocess(datasets):
    """Preprocess cleaned data"""

    Consumptions_TRAIN = consumption_preproc(datasets["Consumptions_TRAIN"])
    Consumptions_TEST = consumption_preproc(datasets["Consumptions_TEST"])
    Requests_TRAIN = requests_preproc(datasets["Requests_TRAIN"])
    Requests_TEST = requests_preproc(datasets["Requests_TEST"])
    Records_TRAIN = records_preproc(datasets["Records_TRAIN"])
    Records_TEST = records_preproc(datasets["Records_TEST"])
    PowerThefts_TRAIN = power_thefts_preproc(datasets["PowerThefts_TRAIN"])

    output = (
        Consumptions_TRAIN,
        Consumptions_TEST,
        Requests_TRAIN,
        Requests_TEST,
        Records_TRAIN,
        Records_TEST,
        PowerThefts_TRAIN,
    )
    return output


def save_preprocessed(
    Consumptions_TRAIN,
    Consumptions_TEST,
    Requests_TRAIN,
    Requests_TEST,
    Records_TRAIN,
    Records_TEST,
    PowerThefts_TRAIN,
):

    logger.info("Saving preprocessed data")
    Consumptions_TRAIN.to_parquet(join(output_path, "Consumptions_TRAIN.pqt"), index=False)
    Requests_TRAIN.to_parquet(join(output_path, "Requests_TRAIN.pqt"), index=False)
    Records_TRAIN.to_parquet(join(output_path, "Records_TRAIN.pqt"), index=False)
    PowerThefts_TRAIN.to_parquet(join(output_path, "PowerThefts_TRAIN.pqt"), index=False)
    Consumptions_TEST.to_parquet(join(output_path, "Consumptions_TEST.pqt"), index=False)
    Requests_TEST.to_parquet(join(output_path, "Requests_TEST.pqt"), index=False)
    Records_TEST.to_parquet(join(output_path, "Records_TEST.pqt"), index=False)


if __name__ == "__main__":
    cleaned = load_cleaned_data()
    test_users = get_test_users(cleaned)
    processed = preprocess(cleaned)

    save_preprocessed(*processed)

    save_test_users(test_users)
    logger.info("Preprocessing complete.")
