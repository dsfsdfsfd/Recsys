from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict

class CustomDatasetSize(Enum):
    LARGE = 'LARGE'
    MEDIUM = 'MEDIUM'
    SMALL = 'SMALL'

class Setting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    MLFLOW_TRACKING_USERNAME: str
    MLFLOW_TRACKING_PASSWORD: str
    #feature engineering
    CUSTOM_DATA_SIZE: CustomDatasetSize = CustomDatasetSize.SMALL
    FEATURES_EMBEDDING_MODEL_ID: str = '/home/u22/Recsys/recsys/raw_data_sources/dataset/all-MiniLM-L12-v2/local_model'

setting = Setting()