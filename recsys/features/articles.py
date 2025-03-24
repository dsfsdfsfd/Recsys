import polars as pl

def get_article_id(df: pl.DataFrame) -> pl.Series:
    "Extracts and return the article_id column as a string."
    return df["article_id"].cast(pl.Utf8)

def create_prod_name_length(df: pl.DataFrame) -> pl.Series:
    "Creates a new column 'prod_name_length' representing the length of 'prod_name'. "
    return df["prod_name"].str.len_chars()

def create_article_description(row):
    description = f"{row['prod_name']} - {row['product_type_name']} in {row['product_group_name']}"
    description += f"\nAppearance: {row['graphical_appearance_name']}"
    description += f"\nColor: {row['perceived_colour_value_name']} {row['perceived_colour_master_name']} ({row['colour_group_name']})"
    description += f"\nCategory: {row['index_group_name']} - {row['section_name']} - {row['garment_group_name']}"

    if row["detail_desc"]:
        description += f"\Detail: {row['detail_desc']}"
    
    return description

def compute_features_articles(df: pl.DataFrame) -> pl.DataFrame:
    "Prepares the input df by creating new features and droppign specific columns"
    df = df.with_columns(
        [
            get_article_id(df).alias("article_id"),
            create_prod_name_length(df).alias("prob_name_length"),
            pl.struct(df.columns)
            .map_elements(create_article_description)
            .alias("article_description"),
        ]
    )
    #img urls
    df = df.with_columns(image_url=pl.col('article_id').map_elements(get_image_url))

    #drop null
    df = df.select([col for col in df.columns if not df[col].is_null().any()])

    #remove 'detail_desc'
    columns_to_drop = ['detail_desc', 'detail_desc_length']
    existing_columns = df.columns
    columns_to_keep = [col for col in existing_columns if col not in columns_to_drop]
    
    return df.select(columns_to_keep)

def get_image_url(article_id):
    url_start = "https://repo.hops.works/dev/jdowling/h-and-m/images/0"
    article_id_str = str(article_id)

    folder = article_id_str[:2]
    image_name = article_id_str

    return f"{url_start}{folder}/0{image_name}.jpg"