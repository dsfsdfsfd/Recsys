import mlflow
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Int32, String, Float32, Array
from feast.value_type import ValueType
import logging
import pandas as pd
from datetime import timedelta
import numpy as np

logger = logging.getLogger(__name__)

from recsys.mlflow_integration import constants
from recsys.config import settings
from recsys.features.transactions import month_cos, month_sin

# Cấu hình Feast Feature Store
def get_feature_store():
    if settings.MLFLOW_TRACKING_URI:
        logger.info("Đặt URI theo dõi MLflow từ biến môi trường.")
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    try:
        feast_store = FeatureStore(repo_path=settings.FEAST_REPO_PATH if hasattr(settings, 'FEAST_REPO_PATH') else ".")
        logger.info("Khởi tạo Feast Feature Store thành công.")
    except Exception as e:
        logger.info(f"Khởi tạo Feast Feature Store thất bại, dùng cấu hình mặc định: {str(e)}")
        feast_store = FeatureStore(repo_path=".")
    
    return mlflow, feast_store

# Định nghĩa Feature View cho Customers
def create_customers_feature_view(fs, df: pd.DataFrame):
    customer = Entity(name="customer_id", join_keys=["customer_id"], value_type=ValueType.INT32)
    
    customers_fv = FeatureView(
        name="customers",
        description="Customers data including age and postal code",
        entities=[customer],
        ttl=timedelta(days=365),  # Thời gian sống của feature
        schema=[
            Field(name="age", dtype=Int32),
            Field(name="club_member_status", dtype=String),
            Field(name="postal_code", dtype=String),
        ],
        online=True,  # Kích hoạt online store
    )
    
    fs.apply([customer, customers_fv])
    fs.materialize_incremental(df=df, end_date=pd.Timestamp.now())
    return customers_fv

# Định nghĩa Feature View cho Articles (với embedding)
def create_articles_feature_view(fs, df: pd.DataFrame, articles_description_embedding_dim: int):
    article = Entity(name="article_id", join_keys=["article_id"], value_type=ValueType.INT32)
    
    articles_fv = FeatureView(
        name="articles",
        description="Fashion items data including type of item, visual description and category",
        entities=[article],
        ttl=timedelta(days=365),
        schema=[
            Field(name="product_type_name", dtype=String),
            Field(name="graphical_appearance_name", dtype=String),
            Field(name="index_group_name", dtype=String),
            Field(name="embeddings", dtype=Array(Float32)),  # Embedding là mảng Float32
        ],
        online=True,
    )
    
    fs.apply([article, articles_fv])
    fs.materialize_incremental(df=df, end_date=pd.Timestamp.now())
    return articles_fv

# Định nghĩa Feature View cho Transactions
def create_transactions_feature_view(fs, df: pd.DataFrame):
    transaction = Entity(name="transaction", join_keys=["customer_id", "article_id"], value_type=ValueType.STRING)
    
    trans_fv = FeatureView(
        name="transactions",
        description="Transactions data including customer, item, price, sales channel and transaction date",
        entities=[transaction],
        ttl=timedelta(days=365),
        schema=[
            Field(name="price", dtype=Float32),
            Field(name="sales_channel_id", dtype=Int32),
            Field(name="t_dat", dtype=String),  # Event time
            Field(name="month_sin", dtype=Float32),
            Field(name="month_cos", dtype=Float32),
        ],
        online=True,
        timestamp_field="t_dat",
    )
    
    fs.apply([transaction, trans_fv])
    fs.materialize_incremental(df=df, end_date=pd.Timestamp.now())
    return trans_fv

# Định nghĩa Feature View cho Interactions
def create_interactions_feature_view(fs, df: pd.DataFrame):
    interaction = Entity(name="interaction", join_keys=["customer_id", "article_id"], value_type=ValueType.STRING)
    
    interactions_fv = FeatureView(
        name="interactions",
        description="Customer interactions with articles including purchases, clicks, and ignores",
        entities=[interaction],
        ttl=timedelta(days=365),
        schema=[
            Field(name="t_dat", dtype=String),
            Field(name="interaction_type", dtype=String),  # Ví dụ: purchase, click, ignore
        ],
        online=True,
        timestamp_field="t_dat",
    )
    
    fs.apply([interaction, interactions_fv])
    fs.materialize_incremental(df=df, end_date=pd.Timestamp.now())
    return interactions_fv

# Định nghĩa Feature View cho Ranking
def create_ranking_feature_view(fs, df: pd.DataFrame):
    ranking = Entity(name="ranking", join_keys=["customer_id", "article_id"], value_type=ValueType.STRING)
    
    rank_fv = FeatureView(
        name="ranking",
        description="Derived feature group for ranking",
        entities=[ranking],
        ttl=timedelta(days=365),
        schema=[
            Field(name="rank_score", dtype=Float32),
        ],
        online=True,
    )
    
    fs.apply([ranking, rank_fv])
    fs.materialize_incremental(df=df, end_date=pd.Timestamp.now())
    return rank_fv

# Định nghĩa Feature View cho Candidate Embeddings
def create_candidate_embeddings_feature_view(fs, df: pd.DataFrame):
    article = Entity(name="article_id", join_keys=["article_id"], value_type=ValueType.INT32)
    
    candidate_fv = FeatureView(
        name="candidate_embeddings",
        description="Embeddings for each article",
        entities=[article],
        ttl=timedelta(days=365),
        schema=[
            Field(name="embeddings", dtype=Array(Float32)),
        ],
        online=True,
    )
    
    fs.apply([article, candidate_fv])
    fs.materialize_incremental(df=df, end_date=pd.Timestamp.now())
    return candidate_fv

#########################
##### Feature Views #####
#########################

def create_retrieval_feature_view(fs):
    # Lấy các FeatureView đã định nghĩa
    trans_fv = fs.get_feature_view("transactions", version=1)
    customers_fv = fs.get_feature_view("customers", version=1)
    articles_fv = fs.get_feature_view("articles", version=1)
    
    retrieval_fv = FeatureView(
        name="retrieval",
        entities=["customer_id", "article_id"],
        ttl=timedelta(days=365),
        schema=[
            Field(name="price", dtype=Float32),
            Field(name="month_sin", dtype=Float32),
            Field(name="month_cos", dtype=Float32),
            Field(name="age", dtype=Int32),
            Field(name="club_member_status", dtype=String),
            Field(name="garment_group_name", dtype=String),
            Field(name="index_group_name", dtype=String),
        ],
        online=True,
    )
    
    fs.apply([retrieval_fv])
    return retrieval_fv

def create_ranking_feature_views(fs):
    rank_fv = fs.get_feature_view("ranking", version=1)
    trans_fv = fs.get_feature_view("transactions", version=1)
    
    ranking_fv = FeatureView(
        name="ranking_view",
        entities=["customer_id", "article_id"],
        ttl=timedelta(days=365),
        schema=[
            Field(name="rank_score", dtype=Float32),
            Field(name="month_sin", dtype=Float32),
            Field(name="month_cos", dtype=Float32),
        ],
        online=True,
    )
    
    fs.apply([ranking_fv])
    return ranking_fv

# Hàm chạy thử
if __name__ == "__main__":
    mlflow, fs = get_feature_store()
    
    # Dữ liệu giả lập
    customers_df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "age": [25, 30, 35],
        "club_member_status": ["ACTIVE", "INACTIVE", "ACTIVE"],
        "postal_code": ["12345", "67890", "54321"]
    })
    
    customers_fv = create_customers_feature_view(fs, customers_df)
    print("Đã tạo customers feature view:", customers_fv.name)