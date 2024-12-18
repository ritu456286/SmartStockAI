
# import streamlit as st

# # Set page configuration
# st.set_page_config(page_title="Nike Sales App", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     /* App background */
#     .stApp {
#         background-image: url('https://www.reuters.com/resizer/v2/https%3A%2F%2Fcloudfront-us-east-2.images.arcpublishing.com%2Freuters%2FYC333JKSXFIDBPSJ5ST2T25XWE.jpg?auth=5a97cf4f8ad5e1dec6c4fd9eeb269b04f3be55c68f3f91430503bd5f024e8f93');
#         background-size: cover;
        
#     }

#     /* Header styling */
#     .header {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         padding: 10px 20px;
#         background-color: rgba(255, 255, 255, 0.8);
#         border-radius: 10px;
#         box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
#     }
#     .logo img {
#         height: 60px;
#     }
#     .icons {
#         display: flex;
#         gap: 15px;
#     }
#     .icons img {
#         height: 30px;
#         cursor: pointer;
#     }

#     /* Sales quotes styling */
#     .sales-quote {
#         text-align: center;
#         margin-top: 40px;
#         font-size: 50px;
#         color: white;
#         font-weight: bold;
       
#     }

   

#     /* Four images in a row with animation */
#     .image-row {
#         display: flex;
#         justify-content: center;
#         gap: 20px;
#         margin-top: 60px;
#     }
#     .image-row img {
#         width: 150px;
#         height: 150px;
#         border-radius: 10px;
#         box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
#         transition: transform 0.3s ease;
#     }
#     .image-row img:hover {
#         transform: scale(1.1) rotate(3deg);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Header with Nike logo and icons
# st.markdown(
#     """
#     <div class="header">
#         <div class="logo">
#             <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg" alt="Nike Logo">
#         </div>
#         <div class="icons">
#             <img src="https://img.icons8.com/ios-glyphs/30/search--v1.png" alt="Search">
#             <img src="https://img.icons8.com/ios-glyphs/30/user--v1.png" alt="User">
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # Sales quote
# st.markdown(
#     """
#     <p class="sales-quote">
#         "Nike continues to inspire and drive excellence in athletes around the world. 
#         With every step, every push, and every challenge, our products empower you to 
#         achieve greatness. As we break new records, we rise together with the relentless 
#         spirit of innovation, passion, and performance. Just Do It."
#     </p>
#     """,
#     unsafe_allow_html=True
# )





# # Four images in a row with animation
# st.markdown(
#     """
#     <div class="image-row">
#         <img src="https://images.pexels.com/photos/1478442/pexels-photo-1478442.jpeg?cs=srgb&dl=pexels-craytive-1478442.jpg&fm=jpg" alt="Nike Shoe 1">
#         <img src="https://c.static-nike.com/a/images/f_auto,cs_srgb/w_1536,c_limit/g1ljiszo4qhthfpluzbt/123-joyride-cdp-apla-xa-xp.jpg" alt="Nike Shoe 2">
#         <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe629LceauQtxtRlGy6A9HagYGkB8vVmphvQ&s" alt="Nike Shoe 3">
#         <img src="https://cdn.pixabay.com/photo/2019/01/26/22/48/nike-3957127_640.jpg" alt="Nike Shoe 4">
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

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


