import random

import polars as pl

from recsys.config import CustomDatasetSize

class DatasetSampler:
    _SIZES = {
        CustomDatasetSize.LARGE: 50_000,
        CustomDatasetSize.MEDIUM: 5_000,
        CustomDatasetSize.SMALL: 1_000,
    }

    def __init__(self, size: CustomDatasetSize) -> None:
        self._size = size

    @classmethod
    def get_supported_sizes(cls) -> dict:
        return cls._SIZES
    
    def sample(
        self, customers_df: pl.DataFrame, transations_df: pl.DataFrame
    ) -> dict[str, pl.DataFrame]:
        random.seed(27)

        n_customers = self._SIZES[self._size]
        customers_df = customers_df.sample(n=n_customers)
        transations_df = transations_df.join(
            customers_df.select("customer_id"), on="customer_id"
        )
        return {"customers": customers_df, "transactions": transations_df}

def fill_missing_club_member_status(df: pl.DataFrame) -> pl.DataFrame:
    "Fill missing values in the 'club_member_status' column with 'ABSENT'. "
    return df.with_columns(pl.col("club_member_status").fill_null("ABSENT"))

def drop_na_age(df: pl.DataFrame) -> pl.DataFrame:
    "Drop rows with null values in the 'age' column"
    return df.drop_nulls(subset=["age"])

def creat_age_group() -> pl.Expr:
    "Creat an expression to categorize age into groups. "
    return (
        pl.when(pl.col("age").is_between(0, 18))
        .then(pl.lit("0-18"))
        .when(pl.col("age").is_between(19, 25))
        .then(pl.lit("19-25"))
        .when(pl.col("age").is_between(26, 35))
        .then(pl.lit("26-35"))
        .when(pl.col("age").is_between(36, 45))
        .then(pl.lit("36-45"))
        .when(pl.col("age").is_between(46, 55))
        .then(pl.lit("46-55"))
        .when(pl.col("age").is_between(56, 65))
        .then(pl.lit("56-65"))
        .otherwise(pl.lit("66+"))
    ).alias("age_group")

def compute_features_customers(
        df: pl.DataFrame, drop_null_age: bool=False
) -> pl.DataFrame:
    """
    1. Checks for required columns in the input DataFrame.
    2. Fill missing with "ABSENT".
    3. Drops rows with missing age value.
    4. Creat an age groups
    5. Cast the 'age' to fl64
    6. Selects and orders specific columns in the ouput.
    """
    df = (
        df.pipe(fill_missing_club_member_status)
        .pipe(drop_na_age)
        .with_columns([creat_age_group(), pl.col("age").cast(pl.Float64)])
        .select(
            ["customer_id", "club_member_status", "age", "postal_code", "age_group"]
        )
    )
    if drop_null_age is True:
        df = df.drop_nulls(subset=["age"])

    return df
