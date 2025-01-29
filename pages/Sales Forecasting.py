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
