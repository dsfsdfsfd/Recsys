from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict

class CustomDatasetSize(Enum):
    LARGE = 'LARGE'
    MEDIUM = 'MEDIUM'
    SMALL = 'SMALL'

class Setting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    #feature engineering
    CUSTOM_DATA_SIZE: CustomDatasetSize = CustomDatasetSize.SMALL
    FEATURES_EMBEDDING_MODEL_ID: str = "all-MiniLM-L6-v2"


setting = Setting()