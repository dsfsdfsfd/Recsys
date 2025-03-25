import random

import polars as pl
from loguru import logger

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
        logger.info(f"Sampling {n_customers} customers.")
        customers_df = customers_df.sample(n=n_customers)

        logger.info(
            f"Number of transactions for all the customers: {transations_df.height}"
        )
        transations_df = transations_df.join(
            customers_df.select("customer_id"), on="customer_id"
        )
        logger.info(
            f"Number of transactions for the {n_customers} sampled customers: {transations_df.height}"
        )

        return {"customers": customers_df, "transactions": transations_df}

def fill_missing_club_member_status(df: pl.DataFrame) -> pl.DataFrame:
    "Fill missing values in the 'club_member_status' column with 'ABSENT'. "
    return df.with_columns(pl.col("club_member_status").fill_null("ABSENT"))

def drop_na_age(df: pl.DataFrame) -> pl.DataFrame:
    "Drop rows with null values in the 'age' column"
    return df.drop_nulls(subset=["age"])


