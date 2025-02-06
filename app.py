import streamlit as st

# Set Streamlit page layout to "wide" to make use of the entire screen width
st.set_page_config(layout="wide")

def main():
    st.title("SmartStockAI: Predictive Inventory & Deadstock Management")

    # Replace with your actual Looker Studio dashboard embed URL
    looker_url = "https://lookerstudio.google.com/embed/u/0/reporting/6c25c091-c34a-442e-aada-5d32aa80aa4b/page/p_awg7t1jynd"

    # Use Streamlit's built-in iframe function to embed the dashboard
    st.markdown(
        f"""
        <iframe src="{looker_url}" 
                style="width: 100%; height: 95vh; border: none;"
                allowfullscreen>
        </iframe>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()


