# SmartStockAI: Predictive Inventory & Deadstock Management

SmartStockAI is an AI-driven solution designed to optimize inventory management by predicting inventory trends, identifying potential deadstock risks, and generating actionable solutions to minimize losses. By leveraging advanced machine learning models and large language models (LLMs), SmartStockAI transforms traditional inventory management into a dynamic, data-driven process that adapts to changing market conditions and consumer behaviors.

## Features

- **Demand Forecasting**: Utilizes the ARIMA_PLUS model in BigQuery ML to predict future sales and identify potential deadstock.
- **Unstructured Data Analysis**: Processes unstructured data, such as customer feedback and vendor notes, using the Gemini 2.0 LLM to extract actionable insights.
- **Actionable Recommendations**: Generates strategies to prevent stock from becoming deadstock, helping businesses reduce waste and improve efficiency.
- **Interactive Visualization**: Provides user-friendly interfaces through Streamlit and Looker Studio for visualizing forecasts and insights.

## Architecture

The high-level architecture of SmartStockAI integrates several Google Cloud tools and services:

- **BigQuery ML**: Implements the ARIMA_PLUS model for demand forecasting.
- **Gemini 2.0 LLM**: Processes unstructured data to generate actionable insights.
- **Cloud SQL**: Stores structured relational data.
- **BigQuery**: Serves as the central analytics engine for large-scale data processing.
- **Streamlit**: Provides an interactive frontend for visualizing forecasts and insights.
- **Looker Studio**: Offers collaborative dashboards for deeper analysis.

## Prerequisites

Before implementing SmartStockAI, ensure you have the following:

- **Google Cloud Platform Tools**:
  - Cloud Storage
  - Cloud SQL
  - BigQuery
  - BigQuery ML
  - Looker Studio
- **Machine Learning Frameworks and Models**:
  - ARIMA_PLUS Model
- **Gemini 2.0 API**
- **Visualization and Deployment Tool**:
  - Streamlit
- **Basic Knowledge Required**:
  - SQL Queries
  - Machine Learning Concepts
  - Python Programming
  - Google Cloud Platform (GCP)

## Getting Started

1. **Data Acquisition**: Obtain inventory data, such as the Nike Sales dataset available on Kaggle.
2. **Data Upload**: Upload the dataset to Google Cloud Storage.
3. **BigQuery Integration**: Enable BigQuery and connect it to Cloud SQL for federated queries to fetch real-time data.
4. **Model Implementation**: Use BigQuery ML to apply the ARIMA_PLUS model for demand forecasting.
5. **Unstructured Data Processing**: Integrate the Gemini 2.0 LLM to process unstructured data and generate insights.
6. **Visualization**: Develop interactive dashboards using Streamlit and Looker Studio to display forecasts and recommendations.

## Resources

- **BigQuery ML ARIMA_PLUS Model**: [Overview](https://cloud.google.com/vertex-ai/docs/tabular-data/forecasting-arima/overview)
- **Google Cloud Storage Documentation**: [Cloud Storage](https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv)
- **Looker Studio Documentation**: [Looker Studio](https://lookerstudio.google.com/)

## Acknowledgments
Thanks to Code Vipassana team for organizing the in-person event.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

