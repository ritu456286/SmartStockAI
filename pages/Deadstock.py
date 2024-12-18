import pandas as pd
import streamlit as st
import os
import google.generativeai as genai



# Configure the API key using environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    # model_name="gemini-2.0-flash-exp",
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Function to read and process CSV data
def process_data(file):
    df = pd.read_csv(file)
    
    # Ensure 'invoice_date' column exists and is in datetime format
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['invoice_date'])
        df['invoice_date'] = df['invoice_date'].dt.strftime('%d-%m-%Y')  # Format to 'DD-MM-YYYY'
    else:
        st.error("The uploaded CSV must have an 'invoice_date' column.")
        st.stop()

    # Define deadstock logic (products with units sold less than 15 are considered deadstock)
    threshold = 15
    df['is_deadstock'] = df['units_sold'] < threshold  # Adjust threshold as needed

    return df

# Function to get insights for a selected deadstock product
def get_insights_for_deadstock(product_info):
    prompt = f"""
    Analyze the following product data and generate insights in a concise bullet-point format:
    
    Product: {product_info['product']}
    Region: {product_info['region']}
    Retailer: {product_info['retailer']}
    Sales Method: {product_info['sales_method']}
    State: {product_info['state']}
    Price per Unit: {product_info['price_per_unit']}
    Total Sales: {product_info['total_sales']}
    Units Sold: {product_info['units_sold']}
    
    Provide insights on:
    - Key issues
    - Root causes
    - Immediate actions
    - Long-Term Solutions
    
    Format the insights as concise bullet points, limiting to the top 2 points for each topic.
    """

    chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
    response = chat_session.send_message(prompt)
    return response.text

# Function to display insights
def display_insights(insights):
    st.write("### Insights:")
    st.write(insights)

# Streamlit UI for file upload and processing
st.title("Deadstock Management and Insights")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

if uploaded_file is not None:
    # Process the data and display the table
    df = process_data(uploaded_file)
    
    # Display the processed data
    st.write("Processed Sales Data:")
    st.write(df)

    # Date range selection
    st.subheader("Filter by Date Range:")
    min_date = df['invoice_date'].min()
    max_date = df['invoice_date'].max()

    start_date = st.date_input("Start Date", min_value=pd.to_datetime(min_date, format='%d-%m-%Y'), 
                               max_value=pd.to_datetime(max_date, format='%d-%m-%Y'), 
                               value=pd.to_datetime(min_date, format='%d-%m-%Y'))
    end_date = st.date_input("End Date", min_value=pd.to_datetime(min_date, format='%d-%m-%Y'), 
                             max_value=pd.to_datetime(max_date, format='%d-%m-%Y'), 
                             value=pd.to_datetime(max_date, format='%d-%m-%Y'))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
    else:
        # Filter data within the selected date range
        filtered_df = df[(pd.to_datetime(df['invoice_date'], format='%d-%m-%Y') >= start_date) & 
                         (pd.to_datetime(df['invoice_date'], format='%d-%m-%Y') <= end_date)]
        
        # Extract Deadstock Products within the date range
        deadstock_df = filtered_df[filtered_df['is_deadstock'] == True]
        st.subheader("Deadstock Products in Selected Date Range:")
        
        # Remove duplicates based on a combination of key fields
        unique_deadstock_df = deadstock_df.drop_duplicates(
            subset=['product', 'region', 'retailer', 'sales_method', 'state']
        )

        st.write(unique_deadstock_df)
        st.write(f"Total Deadstock Products: {unique_deadstock_df.shape[0]}")

        # Dropdown for selecting a product
        selected_product = st.selectbox("Select a Deadstock Product:", 
                                       unique_deadstock_df['product'].tolist(), index=None)

        # Button below dropdown
        if st.button("Generate Insights"):
            if selected_product:
                selected_product_info = unique_deadstock_df[unique_deadstock_df['product'] == selected_product].iloc[0]
                product_info = selected_product_info.to_dict()
                
                # Get insights from Gemini
                insights = get_insights_for_deadstock(product_info)
                
                # Display the insights
                display_insights(insights)
            else:
                st.warning("Please select a product to generate insights.")

        # Button for generating insights for all products
        st.subheader("Generate Insights for All Products:")
        if st.button("Generate Insights for All Deadstock Products"):
            for _, row in unique_deadstock_df.iterrows():
                product_info = row.to_dict()  # Convert the row to a dictionary
                
                # Get insights from Gemini
                insights = get_insights_for_deadstock(product_info)
                
                # Display the insights
                st.subheader(f"Insights for {row['product']}:")
                display_insights(insights)
