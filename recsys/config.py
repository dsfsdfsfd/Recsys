from enum import Enum
from pydantic import SecretStr

from pydantic_settings import BaseSettings, SettingsConfigDict

class CustomDatasetSize(Enum):
    LARGE = 'LARGE'
    MEDIUM = 'MEDIUM'
    SMALL = 'SMALL'

class Setting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    MLFLOW_TRACKING_USERNAME: str | None = None
    MLFLOW_TRACKING_PASSWORD: SecretStr | None = None
    MLFLOW_S3_ENDPOINT_URL: str | None = None
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    MLFLOW_TRACKING_URI: str | None = None
    EXPERIMENT_ID: int | None = None

    #feature engineering
    CUSTOM_DATA_SIZE: CustomDatasetSize = CustomDatasetSize.SMALL
    FEATURES_EMBEDDING_MODEL_ID: str  | None = None
    FEAST_REPO_PATH: str='/home/u22/Recsys'

settings = Setting()