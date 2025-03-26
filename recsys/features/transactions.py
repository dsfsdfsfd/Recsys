import numpy as np
import pandas as pd
import polars as pl
from hopsworks import udf


def convert_article_id_to_str(df: pl.DataFrame) -> pl.Series:
    "Convert the 'article_id' column to string type"
    return df["article_id"].cast(pl.Utf8)

def convert_t_dat_to_datetime(df: pl.DataFrame) -> pl.Series:
    "Convert the t_dat columns to datetime type"
    return pl.from_pandas(pd.to_datetime(df["t_dat"].to_pandas()))

def get_year_feature(df: pl.DataFrame) -> pl.Series:
    "Extract year from the 't_dat'."
    return df['t_dat'].dt.year()

def get_month_feature(df: pl.DataFrame) -> pl.Series:
    "Extract month from the 't_dat'."
    return df['t_dat'].dt.month()

def get_day_feature(df: pl.DataFrame) -> pl.Series:
    "Extract day from the 't_dat'."
    return df['t_dat'].dt.day()

def get_day_of_week_feature(df: pl.DataFrame) -> pl.Series:
    "Extract day of week from the 't_dat'."
    return df['t_dat'].dt.weekday()

def cal_month_sin_cos(month: pl.Series) -> pl.DataFrame:
    "cal sine and code values for the month to capture cyclical patterns. "
    C = 2 * np.pi / 12
    return pl.DataFrame(
        {
            "month_sin": month.apply(lambda x: np.sin(x * C)),
            "month_cos": month.apply(lambda x: np.cos(x * C)),
        }
    )

def convert_t_dat_to_epoch_milliseconds(df: pl.DataFrame) -> pl.Series:
    "Convert the 't_dat' column to epoch milliseconds."
    return df["t_dat"].cast(pl.Int64) // 1_000_000

@udf(return_type = float, mode="pandas")
def month_sin(month :pd.Series):
    "On-demand transformation function that sine of month for cyclical feature encoding."
    return np.sin(month * (2 * np.pi / 12))

@udf(return_type = float, mode="pandas")
def month_cos(month :pd.Series):
    return np.cos(month * (2 * np.pi / 12))
    
def compute_features_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    1.Converts 'article_id' to string type.
    2. Converts 't_dat' to datetime type.
    3. Extracts year, month, day, and day of week from 't_dat'.
    4. Calculates sine and cosine of the month for cyclical feature encoding.
    5. Converts 't_dat' to epoch milliseconds.
    """
    return (
        df.with_columns(
            [
                pl.col("article_id").cast(pl.Utf8).alias("article_id"),
            ]
        )
        .with_columns(
            [
                pl.col("t_dat").dt.year().alias("year"),
                pl.col("t_dat").dt.month().alias("month"),
                pl.col("t_dat").dt.day().alias("day"),
                pl.col("t_dat").dt.weekday().alias("day_of_week"),
            ]
        )
        .with_columns([(pl.col("t_dat").cast(pl.Int64) // 1_000_000).alias("t_dat")])
    )
