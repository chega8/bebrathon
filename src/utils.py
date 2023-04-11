import pandas as pd
import numpy as np
import re
from datetime import date
from loguru import logger
from pandas.api.types import is_numeric_dtype
import numpy as np


def date_handler(date_str: str) -> date:
    date_masks = [
        r"^\d{1,4}-\d{1,2}-\d{1,2}$",
        r"^\d{1,2}\.\d{1,2}\.\d{1,4}$",
        r"^\d{1,2}\/\d{1,2}\/\d{1,2}$",
        r"^\d{1,4}\/\d{1,2}\/\d{1,4}$",
        r"^\d{8}$",
        r"^\d{1,2}-\d{1,2}-\d{1,4}$",
        r"^\d{1,2}\/\d{6}$",
    ]
    if pd.isnull(date_str):
        return
    date_str = str(date_str)
    if re.match(date_masks[0], date_str):
        y, m, d = date_str.split("-")
    elif re.match(date_masks[1], date_str):
        d, m, y = date_str.split(".")
    elif re.match(date_masks[2], date_str):
        d, m, y = date_str.split("/")
        y = "20" + y
    elif re.match(date_masks[3], date_str):
        d, m, y = date_str.split("/")
    elif re.match(date_masks[4], date_str):
        d = date_str[0:2]
        m = date_str[2:4]
        y = date_str[4:8]
    elif re.match(date_masks[5], date_str):
        d, m, y = date_str.split("-")
    elif re.match(date_masks[6], date_str):
        d, my = date_str.split("/")
        m, y = my[:2], my[2:]
    else:
        logger.debug(date_str)
        return "Unknown date format"
    y, m, d = map(int, [y, m, d])
    return date(y, m, d)


def coord_preprocessing(coord: str) -> float:
    if coord == "0":
        return
    return float(coord.replace(",", "."))


def memory_reducer(df):
    mem_before = df.memory_usage().sum() / 1024 / 1024
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            if ((df[col] % 1).sum() == 0) and (df[col].isnull().sum() == 0):
                min_ = df[col].min()
                max_ = max(df[col].max(), abs(min_))
                if min_ >= 0:
                    if max_ < 2**8:
                        df[col] = df[col].astype(np.uint8)
                    elif max_ < 2**16:
                        df[col] = df[col].astype(np.uint16)
                    elif max_ < 2**32:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                if min_ < 0:
                    if max_ < 2**7:
                        df[col] = df[col].astype(np.int8)
                    elif max_ < 2**15:
                        df[col] = df[col].astype(np.int16)
                    elif max_ < 2**31:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

            else:
                df[col] = df[col].astype(np.float32)

    mem_after = df.memory_usage().sum() / 1024 / 1024
    logger.debug(f"Memory saved: {(mem_before-mem_after)/mem_before*100:.2f}%")
    return df
