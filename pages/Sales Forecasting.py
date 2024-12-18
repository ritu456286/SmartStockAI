# import streamlit as st
# from google.cloud import bigquery
# import pandas as pd
# from google.oauth2 import service_account
# import plotly.graph_objects as go

# # Path to your service account key file
# service_account_key = "key.json"

# # Create credentials using the service account key
# credentials = service_account.Credentials.from_service_account_file(
#     service_account_key
# )

# # Initialize BigQuery client with the credentials
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# # Title of the Streamlit App
# st.title("Forecast Visualization for Products")

# # Product options and corresponding model names
# product_model_mapping = {
#     "Women's Athletic Footwear": "units_arima_model1_womenAF",
#     "Men's Apparel": "units_arima_model1_menApparel",
#     "Women's Street Footwear": "units_arima_model1_womenStreetF",
#     "Men's Athletic Footwear": "units_arima_model1_menAF",
#     "Women's Apparel": "units_arima_model1_womenA",
#     "Men's Street Footwear": "units_arima_model1_menStreetF",
#     "ALL": "units_arima_model1"
# }

# # User input: Product selection
# selected_product = st.selectbox(
#     "Select Product:",
#     options=list(product_model_mapping.keys()),
#     index=0  # Default to the first option
# )

# # User input: Forecast Horizon
# forecast_horizon = st.number_input(
#     "Select Forecast Horizon (Number of Days):",
#     min_value=1,
#     max_value=60,
#     value=15,  # Default value is 15
#     step=1
# )

# # Submit Button: Executes the query when clicked
# if st.button("Run Forecast"):
#     # Fetch the model name based on the selected product
#     model_name = product_model_mapping[selected_product]

#     # Construct the query
#     query = f"""
#     SELECT
#       *
#     FROM
#       ML.FORECAST(MODEL `codevip-2.NikeSales1.{model_name}`,
#                   STRUCT({forecast_horizon} AS horizon, 0.9 AS confidence_level))
#     """

#     # Run the query and convert the result into a pandas DataFrame
#     try:
#         st.info("Fetching forecast data from BigQuery...")
#         df = client.query(query).to_dataframe()

#         # Convert forecast_timestamp to datetime format
#         df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])

#         # Display the DataFrame as a table
#         st.subheader("Forecast Data")
#         st.dataframe(df)

#         # Plot Forecast and Confidence Intervals using Plotly
#         st.subheader("Forecast Visualization")
        
#         # Create a Plotly figure
#         fig = go.Figure()

#         # Add Forecast Line
#         fig.add_trace(go.Scatter(
#             x=df["forecast_timestamp"],
#             y=df["forecast_value"],
#             mode='lines+markers',
#             name="Forecast Value",
#             line=dict(color='blue', width=2),
#             hovertemplate="<b>Date:</b> %{x}<br><b>Forecast Value:</b> %{y}<extra></extra>"
#         ))

#         # Add Confidence Interval as a shaded area
#         fig.add_trace(go.Scatter(
#             x=df["forecast_timestamp"],
#             y=df["confidence_interval_lower_bound"],
#             mode='lines',
#             name="Confidence Interval Lower Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=df["forecast_timestamp"],
#             y=df["confidence_interval_upper_bound"],
#             mode='lines',
#             name="Confidence Interval Upper Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             fill='tonexty',  # Fill area between the lines
#             fillcolor='rgba(135,206,235,0.3)',
#             hoverinfo='skip',  # Skip hoverinfo for the shaded area
#             showlegend=False
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             title=f"Forecast for {selected_product}",
#             xaxis_title="Date",
#             yaxis_title="Forecast Value",
#             template="plotly_dark",  # Optional: Change to a dark theme for better visibility
#             hovermode="x unified",  # This makes the hover info show on the same vertical line
#             showlegend=True
#         )

#         # Display the chart in Streamlit
#         st.plotly_chart(fig)

#     except Exception as e:
#         st.error(f"Error fetching or processing data: {e}")
# import pandas as pd
# import streamlit as st
# import os
# import google.generativeai as genai
# from google.oauth2 import service_account
# from google.cloud import bigquery

# service_account_key = "key.json"

# # Create credentials using the service account key
# credentials = service_account.Credentials.from_service_account_file(
#     service_account_key
# )


# # Initialize BigQuery client with the credentials
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)


# # Define the BigQuery dataset and table
# dataset_id = "codevip-2.NikeSales1"
# table_id = "nike_sales_table"

# # Function to fetch data from BigQuery
# def fetch_data_from_bigquery_by_product(prod_name):
#     query = f"""
#         SELECT * FROM `{dataset_id}.{table_id}` WHERE product={prod_name}

#     """
#     df = client.query(query).to_dataframe()
#     return df


# # Configure the API key using environment variable
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Define the model configuration
# generation_config = {
#     "temperature": 1,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#     model_name="gemini-2.0-flash-exp",
#     generation_config=generation_config,
# )

# # Function to get pred_df (Placeholder for the actual function)
# def get_pred_df():
#     # Assume this function returns the pred_df DataFrame with the required columns
#     # For testing purposes, I'll mock the DataFrame.
#     pred_df = client.query(query).to_dataframe()

#         # Convert forecast_timestamp to datetime format
#     pred_df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
#     return pred_df


# # Function to get product information (Placeholder for the actual function)
# def get_product_info(selected_product):
#     # Assume this function returns a DataFrame with product info for the selected product.
#     # For testing purposes, we'll mock this DataFrame.
#     df = fetch_data_from_bigquery_by_product(selected_product)
#     return df

# # Function to flag deadstock based on forecast_value
# def check_deadstock(pred_df, threshold):
#     # Calculate the average forecast value
#     avg_forecast_value = pred_df['forecast_value'].mean()
    
#     # If average forecast value is less than the threshold, return 'Yes' (deadstock), else 'No'
#     return "Yes" if avg_forecast_value < threshold else "No"

# # Function to get insights for a selected deadstock product
# def get_insights_for_deadstock(product_info, pred_df):
#     prompt = f"""
#     Analyze the following product data and generate insights in a concise bullet-point format:
    
#     Product: {product_info['product']}
#     Region: {product_info['region']}
#     Retailer: {product_info['retailer']}
#     Sales Method: {product_info['sales_method']}
#     State: {product_info['state']}
#     Price per Unit: {product_info['price_per_unit']}
#     Total Sales: {product_info['total_sales']}
#     Units Sold: {product_info['units_sold']}
#     Forecast Data: {pred_df[['forecast_timestamp', 'forecast_value']].to_dict(orient='records')}
    
#     Provide insights on:
#     - Key issues
#     - Root causes
#     - Immediate actions
#     - Long-Term Solutions
    
#     Format the insights as concise bullet points, limiting to the top 2 points for each topic.
#     """

#     chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
#     response = chat_session.send_message(prompt)
#     return response.text

# # Function to display insights
# def display_insights(insights):
#     st.write("### Insights:")
#     st.write(insights)

# # Streamlit UI for product selection, prediction data, and insights
# st.title("Deadstock Management and Insights")

# # Define threshold for deadstock
# threshold = 15  # This can be adjusted as needed

# # Fetch the predicted data and product info
# pred_df = get_pred_df()
# selected_product = st.selectbox("Select Product", ['Example Product'])

# # Check if the product is deadstock
# is_deadstock = check_deadstock(pred_df, threshold)
# st.write(f"Is the selected product '{selected_product}' deadstock? {is_deadstock}")

# # If it's deadstock, generate insights
# if is_deadstock == "Yes":
#     st.subheader("Deadstock Insights")
    
#     # Fetch product info
#     product_info_df = get_product_info(selected_product)
#     product_info = product_info_df.iloc[0].to_dict()  # Convert the first row to a dictionary
    
#     # Generate insights from Gemini
#     insights = get_insights_for_deadstock(product_info, pred_df)
    
#     # Display the insights
#     display_insights(insights)


# import streamlit as st
# from google.cloud import bigquery
# import pandas as pd
# from google.oauth2 import service_account
# import plotly.graph_objects as go
# import google.generativeai as genai
# import os

# # Set up API key for Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Path to your service account key file
# service_account_key = "key.json"

# # Create credentials using the service account key
# credentials = service_account.Credentials.from_service_account_file(
#     service_account_key
# )

# # Initialize BigQuery client with the credentials
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# # Title of the Streamlit App
# st.title("Forecast Visualization for Products and Deadstock Management")

# # Product options and corresponding model names
# product_model_mapping = {
#     "Women's Athletic Footwear": "units_arima_model1_womenAF",
#     "Men's Apparel": "units_arima_model1_menApparel",
#     "Women's Street Footwear": "units_arima_model1_womenStreetF",
#     "Men's Athletic Footwear": "units_arima_model1_menAF",
#     "Women's Apparel": "units_arima_model1_womenA",
#     "Men's Street Footwear": "units_arima_model1_menStreetF",
#     "ALL": "units_arima_model1"
# }

# # User input: Product selection
# selected_product = st.selectbox(
#     "Select Product:",
#     options=list(product_model_mapping.keys()),
#     index=0  # Default to the first option
# )

# # User input: Forecast Horizon
# forecast_horizon = st.number_input(
#     "Select Forecast Horizon (Number of Days):",
#     min_value=1,
#     max_value=60,
#     value=15,  # Default value is 15
#     step=1
# )

# # User input: Threshold for Deadstock
# threshold = st.number_input(
#     "Enter Threshold for Deadstock (Units Sold Below Threshold):",
#     min_value=1,
#     value=15,  # Default threshold value is 15
#     step=1
# )

# # Function to get pred_df from BigQuery (the forecast data)
# def get_pred_df(model_name, forecast_horizon):
#     query = f"""
#     SELECT
#       forecast_timestamp,
#       forecast_value,
#       standard_error,
#       confidence_level,
#       prediction_interval_lower_bound,
#       prediction_interval_upper_bound,
#       confidence_interval_lower_bound,
#       confidence_interval_upper_bound
#     FROM
#       ML.FORECAST(MODEL `codevip-2.NikeSales1.{model_name}`,
#                   STRUCT({forecast_horizon} AS horizon, 0.9 AS confidence_level))
#     """

#     # Execute the query and return the results as a DataFrame
#     df = client.query(query).to_dataframe()
#     return df

# # Function to get additional data for the selected product (for insights)
# def get_product_info_df(selected_product):
#     # This is a placeholder function. You can implement it to fetch the required product info from your DB or another source.
#     # For now, it will return a simple example DataFrame with product info.
#     data = {
#         "product": [selected_product],
#         "category": ["Footwear"],  # Example category
#         "region": ["North America"],
#         "retailer": ["NikeStore"],
#         "price_per_unit": [100],
#         "total_sales": [20000],
#         "units_sold": [1000],
#     }
#     return pd.DataFrame(data)

# Function to generate insights from Gemini based on the pred_df and product info

# # Submit Button: Executes the query when clicked
# if st.button("Run Forecast"):
#     # Fetch the model name based on the selected product
#     model_name = product_model_mapping[selected_product]

#     # Fetch the prediction data (pred_df) using the provided model name and forecast horizon
#     try:
#         st.info("Fetching forecast data from BigQuery...")
#         pred_df = get_pred_df(model_name, forecast_horizon)

#         # Convert forecast_timestamp to datetime format
#         pred_df["forecast_timestamp"] = pd.to_datetime(pred_df["forecast_timestamp"])

#         # Display the DataFrame as a table
#         st.subheader("Forecast Data")
#         st.dataframe(pred_df)

#         # Calculate if the product will become deadstock based on the threshold
#         avg_forecast_value = pred_df['forecast_value'].mean()
#         is_deadstock = "Yes" if avg_forecast_value < threshold else "No"

#         # Display the deadstock flag
#         st.subheader(f"Deadstock Status: {is_deadstock}")

#         # Plot Forecast and Confidence Intervals using Plotly
#         st.subheader("Forecast Visualization")
        
#         # Create a Plotly figure
#         fig = go.Figure()

#         # Add Forecast Line
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["forecast_value"],
#             mode='lines+markers',
#             name="Forecast Value",
#             line=dict(color='blue', width=2),
#             hovertemplate="<b>Date:</b> %{x}<br><b>Forecast Value:</b> %{y}<extra></extra>"
#         ))

#         # Add Confidence Interval as a shaded area
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_lower_bound"],
#             mode='lines',
#             name="Confidence Interval Lower Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_upper_bound"],
#             mode='lines',
#             name="Confidence Interval Upper Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             fill='tonexty',  # Fill area between the lines
#             fillcolor='rgba(135,206,235,0.3)',
#             hoverinfo='skip',  # Skip hoverinfo for the shaded area
#             showlegend=False
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             title=f"Forecast for {selected_product}",
#             xaxis_title="Date",
#             yaxis_title="Forecast Value",
#             template="plotly_dark",  # Optional: Change to a dark theme for better visibility
#             hovermode="x unified",  # This makes the hover info show on the same vertical line
#             showlegend=True
#         )

#         # Display the chart in Streamlit
#         st.plotly_chart(fig)

#         # If the product is predicted to become deadstock, show the "Ideas to minimize loss" button
#         if is_deadstock == "Yes":
#             st.button("Ideas to Minimize Loss from Deadstocks", on_click=lambda: get_insights_for_deadstock(pred_df, get_product_info_df(selected_product)))

#     except Exception as e:
#         st.error(f"Error fetching or processing data: {e}")

















# def get_insights_for_deadstock(pred_df, product_info_df):
#     # Extract relevant product information
#     product_info = product_info_df.iloc[0].to_dict()  # Convert the first row to a dictionary

#     # Create the prompt for Gemini to analyze
#     prompt = f"""
#     Analyze the following product data and forecast data to generate insights:
    
#     Product: {product_info['product']}
#     Category: {product_info['category']}
#     Region: {product_info['region']}
#     Retailer: {product_info['retailer']}
#     Price per Unit: {product_info['price_per_unit']}
#     Total Sales: {product_info['total_sales']}
#     Units Sold: {product_info['units_sold']}
    
#     Forecast Data:
#     {pred_df[['forecast_timestamp', 'forecast_value']].to_dict(orient='records')}
    
#     Given this data, determine if the product might become deadstock in the next forecast horizon.
#     Provide insights on:
#     - Key Issues
#     - Root Causes
#     - Immediate Actions
#     - Long-Term Solutions
    
#     Format the insights as concise bullet points.
#     """

#     # Send the request to Gemini
#     chat_session = genai.GenerativeModel(
#         model_name="gemini-2.0-flash-exp",
#         generation_config={"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192}
#     ).start_chat()

#     response = chat_session.send_message(prompt)
#     return response.text



# import streamlit as st
# from google.cloud import bigquery
# import pandas as pd
# from google.oauth2 import service_account
# import plotly.graph_objects as go
# import google.generativeai as genai
# import os

# # Set up API key for Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Path to your service account key file
# service_account_key = "key.json"

# # Create credentials using the service account key
# credentials = service_account.Credentials.from_service_account_file(
#     service_account_key
# )

# # Initialize BigQuery client with the credentials
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# # Title of the Streamlit App
# st.title("Forecast Visualization for Products and Deadstock Management")

# # Product options and corresponding model names
# product_model_mapping = {
#     "Women's Athletic Footwear": "units_arima_model1_womenAF",
#     "Men's Apparel": "units_arima_model1_menApparel",
#     "Women's Street Footwear": "units_arima_model1_womenStreetF",
#     "Men's Athletic Footwear": "units_arima_model1_menAF",
#     "Women's Apparel": "units_arima_model1_womenA",
#     "Men's Street Footwear": "units_arima_model1_menStreetF",
#     "ALL": "units_arima_model1"
# }

# # User input: Product selection and Forecast Horizon
# col1, col2, col3 = st.columns([1, 2, 1])

# with col1:
#     # User input: Product selection
#     selected_product = st.selectbox(
#         "Select Product:",
#         options=list(product_model_mapping.keys()),
#         index=0  # Default to the first option
#     )

# with col2:
#     # User input: Forecast Horizon
#     forecast_horizon = st.number_input(
#         "Select Forecast Horizon (Number of Days):",
#         min_value=1,
#         max_value=60,
#         value=15,  # Default value is 15
#         step=1
#     )

# with col3:
#     # User input: Threshold for Deadstock
#     threshold = st.number_input(
#         "Threshold (Units Sold Below):",
#         min_value=1,
#         value=15,  # Default threshold value is 15
#         step=1
#     )

# # Function to get pred_df from BigQuery (the forecast data)
# def get_pred_df(model_name, forecast_horizon):
#     query = f"""
#     SELECT
#       forecast_timestamp,
#       forecast_value,
#       standard_error,
#       confidence_level,
#       prediction_interval_lower_bound,
#       prediction_interval_upper_bound,
#       confidence_interval_lower_bound,
#       confidence_interval_upper_bound
#     FROM
#       ML.FORECAST(MODEL `codevip-2.NikeSales1.{model_name}`,
#                   STRUCT({forecast_horizon} AS horizon, 0.9 AS confidence_level))
#     """

#     # Execute the query and return the results as a DataFrame
#     df = client.query(query).to_dataframe()
#     return df

# # Function to get additional data for the selected product (for insights)
# def get_product_info_df(selected_product):
#     # This is a placeholder function. You can implement it to fetch the required product info from your DB or another source.
#     # For now, it will return a simple example DataFrame with product info.
#     data = {
#         "product": [selected_product],
#         "category": ["Footwear"],  # Example category
#         "region": ["North America"],
#         "retailer": ["NikeStore"],
#         "price_per_unit": [100],
#         "total_sales": [20000],
#         "units_sold": [1000],
#     }
#     return pd.DataFrame(data)


# # Function to create the prompt for one product
# def create_prompt_for_single_product(pred_df, product_info_df):
#     # Extract product details for the selected product
#     product_info = product_info_df.iloc[0].to_dict()  # Convert the first row to a dictionary
    
#     # Extract forecasted data for the selected product
#     forecast_info = pred_df.iloc[0].to_dict()  # Convert the first row of predictions to a dictionary

#     # Construct the prompt for analysis
#     prompt = f"""
#     You are an advanced analytics model designed to help businesses minimize losses due to deadstock. Your task is to analyze the sales performance and forecast for a specific product to determine whether it is likely to become deadstock and suggest strategies to minimize losses.

#     **Product Information:**
#     The product details are as follows:
#     {product_info}

#     **Forecast Information:**
#     The forecasted data for this product, based on predicted sales and stock levels, is as follows:
#     {forecast_info}

#     **Objective:**
#     - Analyze the predicted sales vs. the current stock level to identify if the product is likely to become deadstock (unsold inventory).
#     - Consider the trends in sales predictions over the coming period.
#     - Identify potential risks of deadstock based on forecast data and stock levels.

#     **Analysis and Recommendations:**
#     1. Identify if the product is predicted to have excess stock compared to forecasted sales (deadstock).
#     2. Based on the analysis, suggest strategies to minimize deadstock:
#         - Can the price be adjusted to stimulate sales?
#         - Should marketing or promotional campaigns be considered?
#         - Can the stock be transferred to locations with higher demand or more favorable sales conditions?
#         - Any other strategies based on the provided data to reduce potential losses?

#     Please provide a detailed analysis of the product's sales forecast and actionable recommendations to minimize deadstock and losses.
#     """

#     return prompt

# # Function to generate insights from Gemini based on the pred_df and product info
# def get_insights_for_deadstock(pred_df, product_info_df):

#     # Debugging: Print product and forecast data
#     print("Generating insights for:", product_info_df.head())
#     print("Forecast Data:", pred_df.head())
#      # Create the prompt for Gemini to analyze the selected product
#     prompt = create_prompt_for_single_product(pred_df, product_info_df)
#     # Extract relevant product information
#     # product_info = product_info_df.iloc[0].to_dict()  # Convert the first row to a dictionary

#     # Create the prompt for Gemini to analyze

#     # Send the request to Gemini
#     chat_session = genai.GenerativeModel(
#         model_name="gemini-2.0-flash-exp",
#         generation_config={"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192}
#     ).start_chat()

#     response = chat_session.send_message(prompt)
#     return response.text

# # Submit Button: Executes the query when clicked
# if st.button("Run Forecast"):
#     # Fetch the model name based on the selected product
#     model_name = product_model_mapping[selected_product]

#     # Fetch the prediction data (pred_df) using the provided model name and forecast horizon
#     try:
#         st.info("Fetching forecast data from BigQuery...")
#         pred_df = get_pred_df(model_name, forecast_horizon)

#         # Convert forecast_timestamp to datetime format
#         pred_df["forecast_timestamp"] = pd.to_datetime(pred_df["forecast_timestamp"])

#         # Display the DataFrame as a table
#         st.subheader("Forecast Data")
#         st.dataframe(pred_df)

#         # Calculate if the product will become deadstock based on the threshold
#         avg_forecast_value = pred_df['forecast_value'].mean()
#         is_deadstock = "Yes" if avg_forecast_value < threshold else "No"

#         # Display the deadstock flag
#         st.subheader(f"Deadstock Status: {is_deadstock}")

#         # Plot Forecast and Confidence Intervals using Plotly
#         st.subheader("Forecast Visualization")
        
#         # Create a Plotly figure
#         fig = go.Figure()

#         # Add Forecast Line
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["forecast_value"],
#             mode='lines+markers',
#             name="Forecast Value",
#             line=dict(color='blue', width=2),
#             hovertemplate="<b>Date:</b> %{x}<br><b>Forecast Value:</b> %{y}<extra></extra>"
#         ))

#         # Add Confidence Interval as a shaded area
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_lower_bound"],
#             mode='lines',
#             name="Confidence Interval Lower Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_upper_bound"],
#             mode='lines',
#             name="Confidence Interval Upper Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             fill='tonexty',  # Fill area between the lines
#             fillcolor='rgba(135,206,235,0.3)',
#             hoverinfo='skip',  # Skip hoverinfo for the shaded area
#             showlegend=False
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             title=f"Forecast for {selected_product}",
#             xaxis_title="Date",
#             yaxis_title="Forecast Value",
#             template="plotly_dark",  # Optional: Change to a dark theme for better visibility
#             hovermode="x unified",  # This makes the hover info show on the same vertical line
#             showlegend=True
#         )

#         # Display the chart in Streamlit
#         st.plotly_chart(fig)

#         # If the product is predicted to become deadstock, show the "Ideas to minimize loss" button
#         if is_deadstock == "Yes":
#             if st.button("Ideas to Minimize Loss from Deadstocks"):
#                 # Fetch the product info and generate insights
#                 product_info_df = get_product_info_df(selected_product)
#                 print(product_info_df.head())
#                 insights = get_insights_for_deadstock(pred_df, product_info_df)
#                 st.subheader("Gemini Insights:")
#                 st.write(insights)

#     except Exception as e:
#         st.error(f"Error fetching or processing data: {e}")

# import streamlit as st
# from google.cloud import bigquery
# import pandas as pd
# from google.oauth2 import service_account
# import plotly.graph_objects as go
# import os

# # Path to your service account key file
# service_account_key = "key.json"

# # Create credentials using the service account key
# credentials = service_account.Credentials.from_service_account_file(
#     service_account_key
# )

# # Initialize BigQuery client with the credentials
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# # Title of the Streamlit App
# st.title("Forecast Visualization for Products and Deadstock Management")

# # Product options and corresponding model names
# product_model_mapping = {
#     "Women's Athletic Footwear": "units_arima_model1_womenAF",
#     "Men's Apparel": "units_arima_model1_menApparel",
#     "Women's Street Footwear": "units_arima_model1_womenStreetF",
#     "Men's Athletic Footwear": "units_arima_model1_menAF",
#     "Women's Apparel": "units_arima_model1_womenA",
#     "Men's Street Footwear": "units_arima_model1_menStreetF",
#     "ALL": "units_arima_model1"
# }

# # User input: Product selection and Forecast Horizon
# col1, col2, col3 = st.columns([1, 2, 1])

# with col1:
#     # User input: Product selection
#     selected_product = st.selectbox(
#         "Select Product:",
#         options=list(product_model_mapping.keys()),
#         index=0  # Default to the first option
#     )

# with col2:
#     # User input: Forecast Horizon
#     forecast_horizon = st.number_input(
#         "Select Forecast Horizon (Number of Days):",
#         min_value=1,
#         max_value=60,
#         value=15,  # Default value is 15
#         step=1
#     )

# with col3:
#     # User input: Threshold for Deadstock
#     threshold = st.number_input(
#         "Threshold (Units Sold Below):",
#         min_value=1,
#         value=15,  # Default threshold value is 15
#         step=1
#     )

# # Function to get pred_df from BigQuery (the forecast data)
# def get_pred_df(model_name, forecast_horizon):
#     query = f"""
#     SELECT
#       forecast_timestamp,
#       forecast_value,
#       standard_error,
#       confidence_level,
#       prediction_interval_lower_bound,
#       prediction_interval_upper_bound,
#       confidence_interval_lower_bound,
#       confidence_interval_upper_bound
#     FROM
#       ML.FORECAST(MODEL `codevip-2.NikeSales1.{model_name}`,
#                   STRUCT({forecast_horizon} AS horizon, 0.9 AS confidence_level))
#     """

#     # Execute the query and return the results as a DataFrame
#     df = client.query(query).to_dataframe()
#     return df

# # Function to get additional data for the selected product (for insights)
# def get_product_info_df(selected_product):
#     # This is a placeholder function. You can implement it to fetch the required product info from your DB or another source.
#     # For now, it will return a simple example DataFrame with product info.
#     data = {
#         "product": [selected_product],
#         "category": ["Footwear"],  # Example category
#         "region": ["North America"],
#         "retailer": ["NikeStore"],
#         "price_per_unit": [100],
#         "total_sales": [20000],
#         "units_sold": [1000],
#     }
#     return pd.DataFrame(data)

# # Function to generate insights without Gemini
# def get_insights_for_deadstock(pred_df, product_info_df):
#     # Debugging: Print product and forecast data
#     print("Generating insights for:", product_info_df.head())
#     print("Forecast Data:", pred_df.head())
    
#     # Simple logic to determine insights based on the forecast
#     avg_forecast_value = pred_df['forecast_value'].mean()
#     insights = []

#     if avg_forecast_value < threshold:
#         insights.append("Key Issues: Likely to become deadstock due to low forecast sales.")
#         insights.append("Root Causes: Insufficient demand or overstock.")
#         insights.append("Immediate Actions: Consider price adjustment, targeted promotions, or transferring stock.")
#         insights.append("Long-Term Solutions: Reassess inventory management and forecasting models.")
#     else:
#         insights.append("The product is unlikely to become deadstock based on the forecast.")
    
#     return "\n".join(insights)

# # Submit Button: Executes the query when clicked
# if st.button("Run Forecast"):
#     # Fetch the model name based on the selected product
#     model_name = product_model_mapping[selected_product]

#     # Fetch the prediction data (pred_df) using the provided model name and forecast horizon
#     try:
#         st.info("Fetching forecast data from BigQuery...")
#         pred_df = get_pred_df(model_name, forecast_horizon)

#         # Convert forecast_timestamp to datetime format
#         pred_df["forecast_timestamp"] = pd.to_datetime(pred_df["forecast_timestamp"])

#         # Display the DataFrame as a table
#         st.subheader("Forecast Data")
#         st.dataframe(pred_df)

#         # Calculate if the product will become deadstock based on the threshold
#         avg_forecast_value = pred_df['forecast_value'].mean()
#         is_deadstock = "Yes" if avg_forecast_value < threshold else "No"

#         # Display the deadstock flag
#         st.subheader(f"Deadstock Status: {is_deadstock}")

#         # Plot Forecast and Confidence Intervals using Plotly
#         st.subheader("Forecast Visualization")
        
#         # Create a Plotly figure
#         fig = go.Figure()

#         # Add Forecast Line
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["forecast_value"],
#             mode='lines+markers',
#             name="Forecast Value",
#             line=dict(color='blue', width=2),
#             hovertemplate="<b>Date:</b> %{x}<br><b>Forecast Value:</b> %{y}<extra></extra>"
#         ))

#         # Add Confidence Interval as a shaded area
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_lower_bound"],
#             mode='lines',
#             name="Confidence Interval Lower Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_upper_bound"],
#             mode='lines',
#             name="Confidence Interval Upper Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             fill='tonexty',  # Fill area between the lines
#             fillcolor='rgba(135,206,235,0.3)',
#             hoverinfo='skip',  # Skip hoverinfo for the shaded area
#             showlegend=False
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             title=f"Forecast for {selected_product}",
#             xaxis_title="Date",
#             yaxis_title="Forecast Value",
#             template="plotly_dark",  # Optional: Change to a dark theme for better visibility
#             hovermode="x unified",  # This makes the hover info show on the same vertical line
#             showlegend=True
#         )

#         # Display the chart in Streamlit
#         st.plotly_chart(fig)

#         # If the product is predicted to become deadstock, show the "Ideas to minimize loss" button
#         if is_deadstock == "Yes":
#             if st.button("Ideas to Minimize Loss from Deadstocks"):
#                 # Fetch the product info and generate insights
#                 product_info_df = get_product_info_df(selected_product)
#                 insights = get_insights_for_deadstock(pred_df, product_info_df)
#                 st.subheader("Insights:")
#                 st.write(insights)

#     except Exception as e:
#         st.error(f"Error fetching or processing data: {e}")

## good code with fixed notifications

# import streamlit as st
# from google.cloud import bigquery
# import pandas as pd
# from google.oauth2 import service_account
# import plotly.graph_objects as go
# import os

# # Path to your service account key file
# service_account_key = "key.json"

# # Create credentials using the service account key
# credentials = service_account.Credentials.from_service_account_file(
#     service_account_key
# )

# # Initialize BigQuery client with the credentials
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# # Title of the Streamlit App
# st.title("Forecast Visualization for Products and Deadstock Management")

# # Product options and corresponding model names
# product_model_mapping = {
#     "Women's Athletic Footwear": "units_arima_model1_womenAF",
#     "Men's Apparel": "units_arima_model1_menApparel",
#     "Women's Street Footwear": "units_arima_model1_womenStreetF",
#     "Men's Athletic Footwear": "units_arima_model1_menAF",
#     "Women's Apparel": "units_arima_model1_womenA",
#     "Men's Street Footwear": "units_arima_model1_menStreetF",
#     "ALL": "units_arima_model1"
# }

# # User input: Product selection and Forecast Horizon
# col1, col2, col3 = st.columns([1, 2, 1])

# with col1:
#     # User input: Product selection
#     selected_product = st.selectbox(
#         "Select Product:",
#         options=list(product_model_mapping.keys()),
#         index=0  # Default to the first option
#     )

# with col2:
#     # User input: Forecast Horizon
#     forecast_horizon = st.number_input(
#         "Select Forecast Horizon (Number of Days):",
#         min_value=1,
#         max_value=60,
#         value=15,  # Default value is 15
#         step=1
#     )

# with col3:
#     # User input: Threshold for Deadstock
#     threshold = st.number_input(
#         "Threshold (Units Sold Below):",
#         min_value=1,
#         value=15,  # Default threshold value is 15
#         step=1
#     )

# # Function to get pred_df from BigQuery (the forecast data)
# def get_pred_df(model_name, forecast_horizon):
#     query = f"""
#     SELECT
#       forecast_timestamp,
#       forecast_value,
#       standard_error,
#       confidence_level,
#       prediction_interval_lower_bound,
#       prediction_interval_upper_bound,
#       confidence_interval_lower_bound,
#       confidence_interval_upper_bound
#     FROM
#       ML.FORECAST(MODEL `codevip-2.NikeSales1.{model_name}`,
#                   STRUCT({forecast_horizon} AS horizon, 0.9 AS confidence_level))
#     """

#     # Execute the query and return the results as a DataFrame
#     df = client.query(query).to_dataframe()
#     return df

# # Function to get additional data for the selected product (for insights)
# def get_product_info_df(selected_product):
#     data = {
#         "product": [selected_product],
#         "category": ["Footwear"],  # Example category
#         "region": ["North America"],
#         "retailer": ["NikeStore"],
#         "price_per_unit": [100],
#         "total_sales": [20000],
#         "units_sold": [1000],
#     }
#     return pd.DataFrame(data)

# # Function to generate insights for deadstock
# def get_insights_for_deadstock(pred_df, product_info_df):
#     avg_forecast_value = pred_df['forecast_value'].mean()
#     insights = []

#     if avg_forecast_value < threshold:
#         insights.append("Key Issues: Likely to become deadstock due to low forecast sales.")
#         insights.append("Root Causes: Insufficient demand or overstock.")
#         insights.append("Immediate Actions: Consider price adjustment, targeted promotions, or transferring stock.")
#         insights.append("Long-Term Solutions: Reassess inventory management and forecasting models.")
#     else:
#         insights.append("The product is unlikely to become deadstock based on the forecast.")
    
#     return "\n".join(insights)

# # Submit Button: Executes the query when clicked
# if st.button("Run Forecast"):
#     model_name = product_model_mapping[selected_product]

#     try:
#         st.info("Fetching forecast data from BigQuery...")
#         pred_df = get_pred_df(model_name, forecast_horizon)

#         # Convert forecast_timestamp to datetime format
#         pred_df["forecast_timestamp"] = pd.to_datetime(pred_df["forecast_timestamp"])

#         # Display the DataFrame as a table
#         st.subheader("Forecast Data")
#         st.dataframe(pred_df)

#         # Calculate if the product will become deadstock based on the threshold
#         avg_forecast_value = pred_df['forecast_value'].mean()
#         is_deadstock = "Yes" if avg_forecast_value < threshold else "No"

#         # Display the deadstock flag
#         if is_deadstock == "Yes":
#             st.warning("This product is predicted to become deadstock.")
#         else:
#             st.success("This product is unlikely to become deadstock.")

#         # Display the deadstock status in text
#         st.subheader(f"Deadstock Status: {is_deadstock}")

#         # Plot Forecast and Confidence Intervals using Plotly
#         st.subheader("Forecast Visualization")
        
#         # Create a Plotly figure
#         fig = go.Figure()

#         # Add Forecast Line
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["forecast_value"],
#             mode='lines+markers',
#             name="Forecast Value",
#             line=dict(color='blue', width=2),
#             hovertemplate="<b>Date:</b> %{x}<br><b>Forecast Value:</b> %{y}<extra></extra>"
#         ))

#         # Add Confidence Interval as a shaded area
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_lower_bound"],
#             mode='lines',
#             name="Confidence Interval Lower Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             showlegend=False
#         ))
#         fig.add_trace(go.Scatter(
#             x=pred_df["forecast_timestamp"],
#             y=pred_df["confidence_interval_upper_bound"],
#             mode='lines',
#             name="Confidence Interval Upper Bound",
#             line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
#             fill='tonexty',  # Fill area between the lines
#             fillcolor='rgba(135,206,235,0.3)',
#             hoverinfo='skip',  # Skip hoverinfo for the shaded area
#             showlegend=False
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             title=f"Forecast for {selected_product}",
#             xaxis_title="Date",
#             yaxis_title="Forecast Value",
#             template="plotly_dark",  # Optional: Change to a dark theme for better visibility
#             hovermode="x unified",  # This makes the hover info show on the same vertical line
#             showlegend=True
#         )

#         # Display the chart in Streamlit
#         st.plotly_chart(fig)

#         # If the product is predicted to become deadstock, show the "Ideas to minimize loss" button
        

#     except Exception as e:
#         st.error(f"Error fetching or processing data: {e}")


import streamlit as st
from google.cloud import bigquery
import pandas as pd
from google.oauth2 import service_account
import plotly.graph_objects as go
import os

# Path to your service account key file
service_account_key = "key.json"

# Create credentials using the service account key
credentials = service_account.Credentials.from_service_account_file(
    service_account_key
)

# Initialize BigQuery client with the credentials
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Title of the Streamlit App
st.title("Forecast Visualization for Products and Deadstock Management")

# Product options and corresponding model names
product_model_mapping = {
    "Women's Athletic Footwear": "units_arima_model1_womenAF",
    "Men's Apparel": "units_arima_model1_menApparel",
    "Women's Street Footwear": "units_arima_model1_womenStreetF",
    "Men's Athletic Footwear": "units_arima_model1_menAF",
    "Women's Apparel": "units_arima_model1_womenA",
    "Men's Street Footwear": "units_arima_model1_menStreetF",
    "ALL": "units_arima_model1"
}

# User input: Product selection and Forecast Horizon
col1, col2, col3 = st.columns([2, 2, 2])  # Adjusted column widths for better alignment

with col1:
    # User input: Product selection
    selected_product = st.selectbox(
        "Select Product:",
        options=list(product_model_mapping.keys()),
        index=0  # Default to the first option
    )

with col2:
    # User input: Forecast Horizon
    forecast_horizon = st.number_input(
        "Select Forecast Horizon (Number of Days):",
        min_value=1,
        max_value=60,
        value=15,  # Default value is 15
        step=1
    )

with col3:
    # User input: Threshold for Deadstock
    threshold = st.number_input(
        "Threshold (Units Sold Below):",
        min_value=1,
        value=15,  # Default threshold value is 15
        step=1
    )

# Function to get pred_df from BigQuery (the forecast data)
def get_pred_df(model_name, forecast_horizon):
    query = f"""
    SELECT
      forecast_timestamp,
      forecast_value,
      standard_error,
      confidence_level,
      prediction_interval_lower_bound,
      prediction_interval_upper_bound,
      confidence_interval_lower_bound,
      confidence_interval_upper_bound
    FROM
      ML.FORECAST(MODEL `codevip-2.NikeSales1.{model_name}`,
                  STRUCT({forecast_horizon} AS horizon, 0.9 AS confidence_level))
    """

    # Execute the query and return the results as a DataFrame
    df = client.query(query).to_dataframe()
    return df

# Function to get additional data for the selected product (for insights)
def get_product_info_df(selected_product):
    data = {
        "product": [selected_product],
        "category": ["Footwear"],  # Example category
        "region": ["North America"],
        "retailer": ["NikeStore"],
        "price_per_unit": [100],
        "total_sales": [20000],
        "units_sold": [1000],
    }
    return pd.DataFrame(data)

# Function to generate insights for deadstock
def get_insights_for_deadstock(pred_df, product_info_df):
    avg_forecast_value = pred_df['forecast_value'].mean()
    insights = []

    if avg_forecast_value < threshold:
        insights.append("Key Issues: Likely to become deadstock due to low forecast sales.")
        insights.append("Root Causes: Insufficient demand or overstock.")
        insights.append("Immediate Actions: Consider price adjustment, targeted promotions, or transferring stock.")
        insights.append("Long-Term Solutions: Reassess inventory management and forecasting models.")
    else:
        insights.append("The product is unlikely to become deadstock based on the forecast.")
    
    return "\n".join(insights)

# Submit Button: Executes the query when clicked
if st.button("Run Forecast"):
    model_name = product_model_mapping[selected_product]

    try:
        st.info("Fetching forecast data from BigQuery...")
        pred_df = get_pred_df(model_name, forecast_horizon)

        # Convert forecast_timestamp to datetime format
        pred_df["forecast_timestamp"] = pd.to_datetime(pred_df["forecast_timestamp"])

        # Display the DataFrame as a table
        st.subheader("Forecast Data")
        st.dataframe(pred_df)

        # Calculate if the product will become deadstock based on the threshold
        avg_forecast_value = pred_df['forecast_value'].mean()
        is_deadstock = "Yes" if avg_forecast_value < threshold else "No"

        # Display the deadstock flag
        if is_deadstock == "Yes":
            st.warning("This product is predicted to become deadstock.")
        else:
            st.success("This product is unlikely to become deadstock.")

        # Display the deadstock status in text
        st.subheader(f"Deadstock Status: {is_deadstock}")

        # Plot Forecast and Confidence Intervals using Plotly
        st.subheader("Forecast Visualization")
        
        # Create a Plotly figure
        fig = go.Figure()

        # Add Forecast Line
        fig.add_trace(go.Scatter(
            x=pred_df["forecast_timestamp"],
            y=pred_df["forecast_value"],
            mode='lines+markers',
            name="Forecast Value",
            line=dict(color='blue', width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Forecast Value:</b> %{y}<extra></extra>"
        ))

        # Add Confidence Interval as a shaded area
        fig.add_trace(go.Scatter(
            x=pred_df["forecast_timestamp"],
            y=pred_df["confidence_interval_lower_bound"],
            mode='lines',
            name="Confidence Interval Lower Bound",
            line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=pred_df["forecast_timestamp"],
            y=pred_df["confidence_interval_upper_bound"],
            mode='lines',
            name="Confidence Interval Upper Bound",
            line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
            fill='tonexty',  # Fill area between the lines
            fillcolor='rgba(135,206,235,0.3)',
            hoverinfo='skip',  # Skip hoverinfo for the shaded area
            showlegend=False
        ))

        # Update layout for the plot
        fig.update_layout(
            title=f"Forecast for {selected_product}",
            xaxis_title="Date",
            yaxis_title="Forecast Value",
            template="plotly_dark",  # Optional: Change to a dark theme for better visibility
            hovermode="x unified",  # This makes the hover info show on the same vertical line
            showlegend=True
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # If the product is predicted to become deadstock, show the "Ideas to minimize loss" button
        

    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")
