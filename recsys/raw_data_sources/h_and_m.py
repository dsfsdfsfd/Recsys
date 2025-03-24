import polars as pl

def extract_articles_df() -> pl.DataFrame:
    return pl.read_csv("/home/u22/Recsys/recsys/raw_data_sources/dataset/articles.csv", try_parse_dates=True)

def extract_customers_df() -> pl.DataFrame:
    return pl.read_csv("/home/u22/Recsys/recsys/raw_data_sources/dataset/customers.csv", try_parse_dates=True)

def extract_transactions_df() -> pl.DataFrame:
    return pl.read_csv("/home/u22/Recsys/recsys/raw_data_sources/dataset/transactions_train.csv", try_parse_dates=True)