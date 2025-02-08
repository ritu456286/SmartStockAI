# SmartStockAI: Predictive Inventory & Deadstock Management  

![SmartStockAI](https://your-image-link-here.com) <!-- Add a relevant banner image if available -->

## 🚀 Overview
SmartStockAI is an AI-driven solution that optimizes inventory management by predicting inventory trends, identifying potential deadstock risks, and generating actionable insights to minimize losses. Leveraging **advanced machine learning models** and **large language models (LLMs)**, it transforms traditional inventory management into a dynamic, data-driven process adaptable to changing market conditions and consumer behaviors.

🏆 **Winner of Google Build and Blog Marathon '24**

## 🌐 Live Demo
🎥 **[Watch Demo Video](https://www.youtube.com/watch?v=6f0wTAcmsTo)**  
🚀 **Deployment:** Initially deployed on `Cloud Run`, but due to cloud charges, it has been removed. All functionalities are showcased in the demo video.  
📝 **Read the Full Story:** **[Medium Blog](https://medium.com/google-cloud/smartstockai-predictive-inventory-deadstock-management-ea8cb0556081)**

---

## 🔥 Features

✅ **Demand Forecasting** - Uses the **ARIMA_PLUS** model in **BigQuery ML** to predict future sales and identify potential deadstock.  
✅ **Unstructured Data Analysis** - Utilizes **Gemini 2.0 LLM** to extract insights from customer feedback and vendor notes.  
✅ **Actionable Recommendations** - Generates strategies to **reduce waste** and **improve efficiency** in inventory management.  
✅ **Interactive Visualization** - Provides dashboards via **Streamlit** and **Looker Studio** to visualize forecasts and insights.  

---

## 🏗️ Architecture

SmartStockAI integrates multiple **Google Cloud** services for a seamless, scalable solution:

- **BigQuery ML** - Implements the **ARIMA_PLUS** model for demand forecasting.
- **Gemini 2.0 LLM** - Processes unstructured data to generate insights.
- **Cloud SQL** - Stores structured relational data.
- **BigQuery** - Serves as the core analytics engine for large-scale data processing.
- **Streamlit** - Provides an interactive frontend for data visualization.
- **Looker Studio** - Offers collaborative dashboards for deeper analysis.

![Architecture Diagram](https://your-architecture-image-link.com) <!-- Add an architecture diagram if available -->

---

## 📌 Prerequisites

Before implementing SmartStockAI, ensure you have the following:

### 🔹 **Google Cloud Platform (GCP) Services**
- Cloud Storage  
- Cloud SQL  
- BigQuery  
- BigQuery ML  
- Looker Studio  

### 🔹 **Machine Learning Models**
- ARIMA_PLUS Model (for demand forecasting)

### 🔹 **APIs & Tools**
- **Gemini 2.0 API** (for unstructured data analysis)
- **Streamlit** (for visualization)

### 🔹 **Required Knowledge**
- SQL Queries  
- Machine Learning Concepts  
- Python Programming  
- Google Cloud Platform (GCP)  

---

## ⚡ Getting Started

Follow these steps to set up and run SmartStockAI:

1️⃣ **Data Acquisition** - Obtain inventory data (e.g., the Nike Sales dataset from Kaggle).  
2️⃣ **Data Upload** - Upload the dataset to **Google Cloud Storage**.  
3️⃣ **BigQuery Integration** - Enable **BigQuery** and connect it to **Cloud SQL** for real-time data retrieval.  
4️⃣ **Model Implementation** - Apply the **ARIMA_PLUS** model in **BigQuery ML** for demand forecasting.  
5️⃣ **Unstructured Data Processing** - Integrate **Gemini 2.0 LLM** for analyzing customer feedback and vendor notes.  
6️⃣ **Visualization** - Develop interactive dashboards using **Streamlit** and **Looker Studio**.  

---

## 📚 Resources

🔗 **[BigQuery ML ARIMA_PLUS Model](https://cloud.google.com/vertex-ai/docs/tabular-data/forecasting-arima/overview)**  
🔗 **[Google Cloud Storage Documentation](https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv)**  
🔗 **[Looker Studio Documentation](https://lookerstudio.google.com/)**  

---

## 🙌 Acknowledgments

A huge thanks to **Code Vipassana** for organizing the in-person event! 🎉

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

### 📩 Have Questions?
Feel free to **open an issue** or **reach out** via [LinkedIn/Twitter/GitHub Discussions]!

---

⭐ **If you find this project useful, don't forget to give it a star!** ⭐

