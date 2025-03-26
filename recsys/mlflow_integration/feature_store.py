from mlflow.server.auth.client import AuthServiceClient
from recsys.config import setting

def get_feature_store():
    client = AuthServiceClient("tracking_uri")
    client.create_user(setting.MLFLOW_TRACKING_USERNAME, setting.MLFLOW_TRACKING_PASSWORD)
    return project, project.get_feature_store()