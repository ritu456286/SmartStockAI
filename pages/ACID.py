from google.cloud import bigquery
import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import os

service_account_key = "key.json"

# Create credentials using the service account key
credentials = service_account.Credentials.from_service_account_file(
    service_account_key
)

# Initialize BigQuery client with the credentials
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Define the BigQuery dataset and table
dataset_id = "codevip-2.NikeSales1"
table_id = "nike_sales_table"

# Function to fetch data from BigQuery
def fetch_data_from_bigquery():
    query = f"""
        SELECT * FROM `{dataset_id}.{table_id}`
        LIMIT 10
    """
    df = client.query(query).to_dataframe()
    return df

# Function to insert data into BigQuery
def insert_into_bigquery(invoice_date, product, region, retailer, sales_method, state, price_per_unit, total_sales, units_sold):
    # Create a temporary staging table (if not already created)
    staging_table = f"{dataset_id}.nike_sales_staging"
    
    # Format the values correctly for SQL insertion
    insert_query = f"""
        INSERT INTO `{staging_table}` (invoice_date, product, region, retailer, sales_method, state, price_per_unit, total_sales, units_sold)
        VALUES ('{invoice_date}', '{product}', '{region}', '{retailer}', '{sales_method}', '{state}', {price_per_unit}, {total_sales}, {units_sold})
    """
    client.query(insert_query).result()  # Execute the insert query
    print("Inserted into staging table.")

# Function to commit the changes to the main table
def commit_to_main_table():
    # Merge data from staging table to the main table
    merge_query = f"""
        MERGE INTO `{dataset_id}.{table_id}` AS main
        USING `{dataset_id}.nike_sales_staging` AS staging
        ON main.invoice_date = staging.invoice_date
           AND main.product = staging.product
           AND main.retailer = staging.retailer
        WHEN MATCHED THEN
            UPDATE SET main.total_sales = staging.total_sales, main.units_sold = staging.units_sold
        WHEN NOT MATCHED THEN
            INSERT (invoice_date, product, region, retailer, sales_method, state, price_per_unit, total_sales, units_sold)
            VALUES (staging.invoice_date, staging.product, staging.region, staging.retailer, staging.sales_method, staging.state, staging.price_per_unit, staging.total_sales, staging.units_sold)
    """
    client.query(merge_query).result()  # Execute the merge query
    print("Merge completed successfully!")

# Function to roll back by deleting data in the staging table
def rollback_staging_table():
    delete_query = f"DELETE FROM `{dataset_id}.nike_sales_staging` WHERE 1=1"
    client.query(delete_query).result()  # Delete the data in the staging table
    print("Staging table data deleted.")



# Streamlit app components to interact with the user
st.title("Nike Sales Data - ACID-like Transactions")

# Display the data
df = fetch_data_from_bigquery()
st.write(df)

# Form to input new sales data
with st.form(key='new_sales_form'):
    st.subheader("Enter New Sales Data")

    invoice_date = st.date_input('Invoice Date')
    product = st.text_input('Product')
    region = st.text_input('Region')
    retailer = st.text_input('Retailer')
    sales_method = st.text_input('Sales Method')
    state = st.text_input('State')
    price_per_unit = st.number_input('Price per Unit', min_value=0)
    total_sales = st.number_input('Total Sales', min_value=0)
    units_sold = st.number_input('Units Sold', min_value=0)

    submit_button = st.form_submit_button(label='Insert New Data')

    if submit_button:
        insert_into_bigquery(invoice_date, product, region, retailer, sales_method, state, price_per_unit, total_sales, units_sold)
        st.success("Data inserted into staging table. Now commit to the main table.")

# Display commit and rollback buttons outside the form
st.subheader("Actions")

if st.button("Commit Changes to Main Table"):
    commit_to_main_table()
    st.success("Changes committed to the main table.")

if st.button("Rollback Changes"):
    rollback_staging_table()
    st.success("Changes rolled back.")