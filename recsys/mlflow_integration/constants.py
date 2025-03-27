from feast import Field
from feast.types import Int64, String, Float64, Array

### Post ingestion format.###

customer_feature_descriptions = [
    {"name": "customer_id", "description": "Unique identifier for each customer."},
    {
        "name": "club_member_status",
        "description": "Membership status of the customer in the club.",
    },
    {"name": "age", "description": "Age of the customer."},
    {
        "name": "postal_code",
        "description": "Postal code associated with the customer's address.",
    },
    {"name": "age_group", "description": "Categorized age group of the customer."},
]

transactions_feature_descriptions = [
    {"name": "t_dat", "description": "Timestamp of the data record."},
    {"name": "customer_id", "description": "Unique identifier for each customer."},
    {"name": "article_id", "description": "Identifier for the purchased article."},
    {"name": "price", "description": "Price of the purchased article."},
    {"name": "sales_channel_id", "description": "Identifier for the sales channel."},
    {"name": "year", "description": "Year of the transaction."},
    {"name": "month", "description": "Month of the transaction."},
    {"name": "day", "description": "Day of the transaction."},
    {"name": "day_of_week", "description": "Day of the week of the transaction."},
    {
        "name": "month_sin",
        "description": "Sine of the month used for seasonal patterns.",
    },
    {
        "name": "month_cos",
        "description": "Cosine of the month used for seasonal patterns.",
    },
]

interactions_feature_descriptions = [
    {"name": "t_dat", "description": "Timestamp of the interaction."},
    {"name": "customer_id", "description": "Unique identifier for each customer."},
    {
        "name": "article_id",
        "description": "Identifier for the article that was interacted with.",
    },
    {
        "name": "interaction_score",
        "description": "Type of interaction: 0 = ignore, 1 = click, 2 = purchase.",
    },
    {
        "name": "prev_article_id",
        "description": "Previous article that the customer interacted with, useful for sequential recommendation patterns.",
    },
]

ranking_feature_descriptions = [
    {"name": "customer_id", "description": "Unique identifier for each customer."},
    {"name": "article_id", "description": "Identifier for the purchased article."},
    {"name": "age", "description": "Age of the customer."},
    {"name": "product_type_name", "description": "Name of the product type."},
    {"name": "product_group_name", "description": "Name of the product group."},
    {
        "name": "graphical_appearance_name",
        "description": "Name of the graphical appearance.",
    },
    {"name": "colour_group_name", "description": "Name of the colour group."},
    {
        "name": "perceived_colour_value_name",
        "description": "Name of the perceived colour value.",
    },
    {
        "name": "perceived_colour_master_name",
        "description": "Name of the perceived colour master.",
    },
    {"name": "department_name", "description": "Name of the department."},
    {"name": "index_name", "description": "Name of the index."},
    {"name": "index_group_name", "description": "Name of the index group."},
    {"name": "section_name", "description": "Name of the section."},
    {"name": "garment_group_name", "description": "Name of the garment group."},
    {
        "name": "label",
        "description": "Label indicating whether the article was purchased (1) or not (0).",
    },
]

### Pre ingestion format. ###

article_feature_description = [
    Field(
        name="article_id", dtype=String, description="Identifier for the article."
    ),
    Field(
        name="product_code",
        dtype=Int64,
        description="Code associated with the product.",
    ),
    Field(name="prod_name", dtype=String, description="Name of the product."),
    Field(
        name="product_type_no",
        dtype=Int64,
        description="Number associated with the product type.",
    ),
    Field(
        name="product_type_name", dtype=String, description="Name of the product type."
    ),
    Field(
        name="product_group_name",
        dtype=String,
        description="Name of the product group.",
    ),
    Field(
        name="graphical_appearance_no",
        dtype=Int64,
        description="Number associated with graphical appearance.",
    ),
    Field(
        name="graphical_appearance_name",
        dtype=String,
        description="Name of the graphical appearance.",
    ),
    Field(
        name="colour_group_code",
        dtype=Int64,
        description="Code associated with the colour group.",
    ),
    Field(
        name="colour_group_name", dtype=String, description="Name of the colour group."
    ),
    Field(
        name="perceived_colour_value_id",
        dtype=Int64,
        description="ID associated with perceived colour value.",
    ),
    Field(
        name="perceived_colour_value_name",
        dtype=String,
        description="Name of the perceived colour value.",
    ),
    Field(
        name="perceived_colour_master_id",
        dtype=Int64,
        description="ID associated with perceived colour master.",
    ),
    Field(
        name="perceived_colour_master_name",
        dtype=String,
        description="Name of the perceived colour master.",
    ),
    Field(
        name="department_no",
        dtype=Int64,
        description="Number associated with the department.",
    ),
    Field(
        name="department_name", dtype=String, description="Name of the department."
    ),
    Field(
        name="index_code", dtype=String, description="Code associated with the index."
    ),
    Field(name="index_name", dtype=String, description="Name of the index."),
    Field(
        name="index_group_no",
        dtype=Int64,
        description="Number associated with the index group.",
    ),
    Field(
        name="index_group_name", dtype=String, description="Name of the index group."
    ),
    Field(
        name="section_no",
        dtype=Int64,
        description="Number associated with the section.",
    ),
    Field(name="section_name", dtype=String, description="Name of the section."),
    Field(
        name="garment_group_no",
        dtype=Int64,
        description="Number associated with the garment group.",
    ),
    Field(
        name="garment_group_name",
        dtype=String,
        description="Name of the garment group.",
    ),
    Field(
        name="prod_name_length",
        dtype=Int64,
        description="Length of the product name.",
    ),
    Field(
        name="article_description",
        dtype=String,
        description="Description of the article.",
    ),
    Field(
        name="embeddings",
        dtype=Array(Float64),
        description="Vector embeddings of the article description.",
    ),
    Field(name="image_url", dtype=String, description="URL of the product image."),
]