# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import for Google Gemini
# from dotenv import load_dotenv
# import os
# import json

# # Loading of the environment variables
# load_dotenv()

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')  # Updated to use GOOGLE_API_KEY
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     # Initialize the Google Gemini instance
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",  # Use the appropriate Gemini model
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     # Creating the Pandas DataFrame agent with the LLM
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
#     # Prompt with query guidelines and formatting
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
    
#     # Run the prompt through the agent and capture the response.
#     result = agent.run(prompt)
    
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         st.write(response_dict["answer"])

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig)

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig)

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig)

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig)

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Function to try different encodings and read CSV file
# def read_csv_with_encodings(file):
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']  # List of encodings to try
#     for encoding in encodings:
#         try:
#             file.seek(0)  # Reset the file pointer to the beginning
#             df = pd.read_csv(file, encoding=encoding)
#             return df
#         except Exception as e:
#             # Continue to the next encoding if the current one fails
#             continue
#     return f"Error reading CSV file with provided encodings: {e}"

# # Setting the layout configuration for the Streamlit page
# st.set_page_config(page_title="CSV ChatApp", layout='wide', page_icon="📊")

# # CSS starts
# st.markdown("""
#     <style>
#         .stButton>button {
#             color: white;
#             background-color: #4CAF50;
#             padding: 10px 24px;
#             font-size: 16px;
#             border-radius: 10px;
#         }
#         .stTextArea textarea {
#             background-color: #f0f0f5;
#             font-size: 18px;
#             padding: 10px;
#         }
#         .sidebar .sidebar-content {
#             background-color: #f7f9fc;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Header and description
# st.title("Car Sales Advertisements")
# st.markdown("""
#     Welcome to the CSV ChatApp!
    
#     Select a file, enter your query, and get instant insights!
# """)

# # Sidebar for file upload and instructions
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # If files are uploaded, proceed to display options
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
    
#     # Load and display the selected CSV file
#     data = read_csv_with_encodings(input_csvs[selected_index])
#     if isinstance(data, pd.DataFrame):
#         st.subheader(f"Preview of {selected_file}")
#         st.dataframe(data.head(10), use_container_width=True)
        
#         # Query input section
#         st.subheader("Enter your query")
#         input_text = st.text_area("What do you want to know?", placeholder="Type your question here...")
        
#         # Chat button and results
#         if input_text:
#             if st.button("Chat with CSV"):
#                 st.info(f"Your Query: {input_text}")
#                 with st.spinner("Processing..."):
#                     result = chat_with_csv(data, input_text)
#                 decoded_response = decode_response(result)
#                 write_answer(decoded_response)
#     else:
#         st.error(data)
# else:
#     st.sidebar.info("Upload one or more CSV files to get started.")

# # Footer
# st.markdown("""
#     ---
#      Powered by LLM and Streamlit
# """)



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import for Google Gemini
# from dotenv import load_dotenv
# import os
# import json

# # Loading of the environment variables
# load_dotenv()

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')  # Updated to use GOOGLE_API_KEY
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     # Initialize the Google Gemini instance
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",  # Use the appropriate Gemini model
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     # Creating the Pandas DataFrame agent with the LLM
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
#     # Prompt with query guidelines and formatting
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
    
#     # Run the prompt through the agent and capture the response.
#     result = agent.run(prompt)
    
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         st.write(response_dict["answer"])

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig)

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig)

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig)

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig)

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Function to try different encodings and read CSV file
# def read_csv_with_encodings(file):
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']  # List of encodings to try
#     for encoding in encodings:
#         try:
#             file.seek(0)  # Reset the file pointer to the beginning
#             df = pd.read_csv(file, encoding=encoding)
#             return df
#         except Exception as e:
#             # Continue to the next encoding if the current one fails
#             continue
#     return f"Error reading CSV file with provided encodings: {e}"

# # Setting the layout configuration for the Streamlit page
# st.set_page_config(page_title="CSV ChatApp", layout='wide', page_icon="📊")

# # CSS starts
# st.markdown("""
#     <style>
#         .stButton>button {
#             color: white;
#             background-color: #4CAF50;
#             padding: 10px 24px;
#             font-size: 16px;
#             border-radius: 10px;
#         }
#         .stTextArea textarea {
#             background-color: #f0f0f5;
#             font-size: 18px;
#             padding: 10px;
#         }
#         .sidebar .sidebar-content {
#             background-color: #f7f9fc;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Header and description
# st.title("Car Sales Advertisements")
# st.markdown("""
#     Welcome to the CSV ChatApp!
    
#     Select a file, enter your query, and get instant insights!
# """)

# # Sidebar for file upload and instructions
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Initialize conversation history in session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# # If files are uploaded, proceed to display options
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
    
#     # Load and display the selected CSV file
#     data = read_csv_with_encodings(input_csvs[selected_index])
#     if isinstance(data, pd.DataFrame):
#         st.subheader(f"Preview of {selected_file}")
#         st.dataframe(data.head(10), use_container_width=True)
        
#         # Query input section
#         st.subheader("Enter your query")
#         input_text = st.text_area("What do you want to know?", placeholder="Type your question here...")
        
#         # Chat button and results
#         if input_text:
#             if st.button("Chat with CSV"):
#                 st.info(f"Your Query: {input_text}")
#                 with st.spinner("Processing..."):
#                     result = chat_with_csv(data, input_text)
#                 decoded_response = decode_response(result)
#                 write_answer(decoded_response)
                
#                 # Store the query and answer in the conversation history
#                 st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})
        
#         # Display conversation history
#         if st.session_state.conversation_history:
#             st.subheader("Conversation History:")
#             for idx, convo in enumerate(st.session_state.conversation_history):
#                 st.markdown(f"**Q{idx+1}:** {convo['query']}")
#                 write_answer(convo['response'])
#                 st.markdown("---")
#     else:
#         st.error(data)
# else:
#     st.sidebar.info("Upload one or more CSV files to get started.")

# # Footer
# st.markdown("""
#     ---
#      Powered by LLM and Streamlit
# """)

#  FINAL CODE


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import for Google Gemini
# from dotenv import load_dotenv
# import os
# import json

# # Loading of the environment variables
# load_dotenv()

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')  # Updated to use GOOGLE_API_KEY
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     # Initialize the Google Gemini instance
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",  # Use the appropriate Gemini model
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     # Creating the Pandas DataFrame agent with the LLM
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
#     # Prompt with query guidelines and formatting
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
    
#     # Run the prompt through the agent and capture the response.
#     result = agent.run(prompt)
    
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         st.write(response_dict["answer"])

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="bar_chart")

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="line_chart")

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig, key="pie_chart")

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="scatter_chart")

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Function to try different encodings and read CSV file
# def read_csv_with_encodings(file):
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']  # List of encodings to try
#     for encoding in encodings:
#         try:
#             file.seek(0)  # Reset the file pointer to the beginning
#             df = pd.read_csv(file, encoding=encoding)
#             return df
#         except Exception as e:
#             # Continue to the next encoding if the current one fails
#             continue
#     return f"Error reading CSV file with provided encodings: {e}"

# # Setting the layout configuration for the Streamlit page
# st.set_page_config(page_title="CSV ChatApp", layout='wide', page_icon="📊")

# # CSS starts
# st.markdown("""
#     <style>
#         .stButton>button {
#             color: white;
#             background-color: #4CAF50;
#             padding: 10px 24px;
#             font-size: 16px;
#             border-radius: 10px;
#         }
#         .stTextArea textarea {
#             background-color: #f0f0f5;
#             font-size: 18px;
#             padding: 10px;
#         }
#         .sidebar .sidebar-content {
#             background-color: #f7f9fc;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Header and description
# st.title("Car Sales Advertisements")
# st.markdown("""
#     Welcome to the CSV ChatApp!
    
#     Select a file, enter your query, and get instant insights!
# """)

# # Sidebar for file upload and instructions
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Initialize conversation history in session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# # If files are uploaded, proceed to display options
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
    
#     # Load and display the selected CSV file
#     data = read_csv_with_encodings(input_csvs[selected_index])
#     if isinstance(data, pd.DataFrame):
#         st.subheader(f"Preview of {selected_file}")
#         st.dataframe(data.head(10), use_container_width=True)
        
#         # Query input section
#         st.subheader("Enter your query")
#         input_text = st.text_area("What do you want to know?", placeholder="Type your question here...")
        
#         # Chat button and results
#         if input_text:
#             if st.button("Chat with CSV"):
#                 st.info(f"Your Query: {input_text}")
#                 with st.spinner("Processing..."):
#                     result = chat_with_csv(data, input_text)
#                 decoded_response = decode_response(result)
#                 write_answer(decoded_response)
                
#                 # Store the query and answer in the conversation history
#                 st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})
        
#         # Display conversation history
#         if st.session_state.conversation_history:
#             st.subheader("Conversation History:")
#             for idx, convo in enumerate(st.session_state.conversation_history):
#                 st.markdown(f"**Q{idx+1}:** {convo['query']}")
#                 write_answer(convo['response'])
#                 st.markdown("---")
#     else:
#         st.error(data)
# else:
#     st.sidebar.info("Upload one or more CSV files to get started.")

# # Footer
# st.markdown("""
#     ---
#      Powered by LLM and Streamlit
# """)



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# import json
# import speech_recognition as sr
# import pyttsx3

# # Loading environment variables
# load_dotenv()

# # Initialize text-to-speech engine
# tts_engine = pyttsx3.init()

# # Function for text-to-speech
# def speak_text(text):
#     tts_engine.say(text)
#     tts_engine.runAndWait()

# # Function for speech-to-text
# def listen_to_query():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak your query now.")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             query = recognizer.recognize_google(audio)
#             st.success(f"You said: {query}")
#             return query
#         except sr.UnknownValueError:
#             st.error("Sorry, I could not understand the audio.")
#             return None
#         except sr.RequestError:
#             st.error("Could not request results from the speech recognition service.")
#             return None

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
    
#     result = agent.run(prompt)
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         answer = response_dict["answer"]
#         st.write(answer)
#         speak_text(answer)  # Speak the answer

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="bar_chart")

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="line_chart")

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig, key="pie_chart")

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="scatter_chart")

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Streamlit UI setup
# st.title("Car Sales Advertisements with Voice Assistance")
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Initialize session state for conversation history
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# input_text = None
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
#     df = pd.read_csv(input_csvs[selected_index])

#     st.subheader(f"Preview of {selected_file}")
#     st.dataframe(df.head(10))

#     # Voice or Text Query Input
#     query_mode = st.radio("How would you like to input your query?", ["Type", "Speak"])

#     if query_mode == "Type":
#         input_text = st.text_area("What do you want to know?", placeholder="Type your question here...")
#     else:
#         if st.button("Speak Now"):
#             input_text = listen_to_query()

#     if input_text:
#         st.info(f"Your Query: {input_text}")
#         with st.spinner("Processing..."):
#             result = chat_with_csv(df, input_text)
#         decoded_response = decode_response(result)
#         write_answer(decoded_response)

#         # Store conversation history
#         st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})

#     # Conversation History
#     if st.session_state.conversation_history:
#         st.subheader("Conversation History:")
#         for idx, convo in enumerate(st.session_state.conversation_history):
#             st.markdown(f"**Q{idx+1}:** {convo['query']}")
#             write_answer(convo['response'])
# else:
#     st.sidebar.info("Upload a CSV file to get started.")


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# import json
# import speech_recognition as sr
# import pyttsx3
# import threading

# # Loading environment variables
# load_dotenv()

# # Initialize text-to-speech engine
# tts_engine = pyttsx3.init()

# # Function for text-to-speech
# def speak_text(text):
#     # Define a helper function to run the speech in a separate thread
#     def speak():
#         tts_engine.say(text)
#         tts_engine.runAndWait()
    
#     # Start a new thread for the speak function
#     threading.Thread(target=speak).start()

# # Function for speech-to-text
# def listen_to_query():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak your query now.")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             query = recognizer.recognize_google(audio)
#             st.success(f"You said: {query}")
#             return query
#         except sr.UnknownValueError:
#             st.error("Sorry, I could not understand the audio.")
#             return None
#         except sr.RequestError:
#             st.error("Could not request results from the speech recognition service.")
#             return None

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
    
#     result = agent.run(prompt)
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         answer = response_dict["answer"]
#         st.write(answer)
#         speak_text(answer)  # Speak the answer

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="bar_chart")

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="line_chart")

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig, key="pie_chart")

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="scatter_chart")

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Streamlit UI setup
# st.title("Car Sales Advertisements with Voice Assistance")
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Initialize session state for conversation history
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# input_text = None
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
#     df = pd.read_csv(input_csvs[selected_index])

#     st.subheader(f"Preview of {selected_file}")
#     st.dataframe(df.head(10))

#     # Voice or Text Query Input
#     query_mode = st.radio("How would you like to input your query?", ["Type", "Speak"])

#     if query_mode == "Type":
#         input_text = st.text_area("What do you want to know?", placeholder="Type your question here...")
#     else:
#         if st.button("Speak Now"):
#             input_text = listen_to_query()

#     if input_text:
#         st.info(f"Your Query: {input_text}")
#         with st.spinner("Processing..."):
#             result = chat_with_csv(df, input_text)
#         decoded_response = decode_response(result)
#         write_answer(decoded_response)

#         # Store conversation history
#         st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})

#     # Conversation History
#     if st.session_state.conversation_history:
#         st.subheader("Conversation History:")
#         for idx, convo in enumerate(st.session_state.conversation_history):
#             st.markdown(f"**Q{idx+1}:** {convo['query']}")
#             write_answer(convo['response'])
# else:
#     st.sidebar.info("Upload a CSV file to get started.")




#PERFECT WORKING FINAL CODE 


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# import json
# import speech_recognition as sr
# import pyttsx3
# import threading
# import queue

# # Loading environment variables
# load_dotenv()

# # Initialize text-to-speech engine
# tts_engine = pyttsx3.init()

# # Create a queue for communication between threads
# speech_queue = queue.Queue()

# # Function for text-to-speech
# def speak_text(text):
#     # Put the text to the queue, the main thread will process it
#     speech_queue.put(text)

# def process_speech():
#     while True:
#         # Get the text from the queue and process it
#         text = speech_queue.get()
#         if text == "STOP":
#             break
#         tts_engine.say(text)
#         tts_engine.runAndWait()

# # Start the speech processing in a separate thread
# speech_thread = threading.Thread(target=process_speech, daemon=True)
# speech_thread.start()

# # Function for speech-to-text
# def listen_to_query():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak your query now.")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             query = recognizer.recognize_google(audio)
#             st.success(f"You said: {query}")
#             return query
#         except sr.UnknownValueError:
#             st.error("Sorry, I could not understand the audio.")
#             return None
#         except sr.RequestError:
#             st.error("Could not request results from the speech recognition service.")
#             return None

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
    
#     result = agent.run(prompt)
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         answer = response_dict["answer"]
#         st.write(answer)
#         speak_text(answer)  # Add text to queue for speech synthesis

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="bar_chart")

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="line_chart")

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig, key="pie_chart")

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="scatter_chart")

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Streamlit UI setup
# st.title("Car Sales Advertisements with Voice Assistance")
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Initialize session state for conversation history
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# input_text = None
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
#     df = pd.read_csv(input_csvs[selected_index])

#     st.subheader(f"Preview of {selected_file}")
#     st.dataframe(df.head(10))

#     # Voice or Text Query Input
#     query_mode = st.radio("How would you like to input your query?", ["Type", "Speak"])

#     if query_mode == "Type":
#         input_text = st.text_area("What do you want to know?", placeholder="Type your question here...")
#     else:
#         if st.button("Speak Now"):
#             input_text = listen_to_query()

#     if input_text:
#         st.info(f"Your Query: {input_text}")
#         with st.spinner("Processing..."):
#             result = chat_with_csv(df, input_text)
#         decoded_response = decode_response(result)
#         write_answer(decoded_response)

#         # Store conversation history
#         st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})

#     # Conversation History
#     if st.session_state.conversation_history:
#         st.subheader("Conversation History:")
#         for idx, convo in enumerate(st.session_state.conversation_history):
#             st.markdown(f"**Q{idx+1}:** {convo['query']}")
#             write_answer(convo['response'])
# else:
#     st.sidebar.info("Upload a CSV file to get started.")








# with 1.5 corrected


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# import json
# import speech_recognition as sr
# import pyttsx3
# import threading
# import queue

# # Loading environment variables
# load_dotenv()

# # Initialize text-to-speech engine
# tts_engine = pyttsx3.init()

# # Create a queue for communication between threads
# speech_queue = queue.Queue()

# # Function for text-to-speech
# def speak_text(text):
#     speech_queue.put(text)

# def process_speech():
#     while True:
#         text = speech_queue.get()
#         if text == "STOP":
#             break
#         tts_engine.say(text)
#         tts_engine.runAndWait()

# # Start the speech processing in a separate thread
# speech_thread = threading.Thread(target=process_speech, daemon=True)
# speech_thread.start()

# # Function for speech-to-text
# def listen_to_query():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak your query now.")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             query = recognizer.recognize_google(audio)
#             st.success(f"You said: {query}")
#             return query
#         except sr.UnknownValueError:
#             st.error("Sorry, I could not understand the audio.")
#             return None
#         except sr.RequestError:
#             st.error("Could not request results from the speech recognition service.")
#             return None

# # Function to chat with CSV data
# def chat_with_csv(df, query):
#     gemini_api_key = os.environ.get('GOOGLE_API_KEY')
#     if not gemini_api_key:
#         return "GOOGLE_API_KEY environment variable not set."
    
#     llm = ChatGoogleGenerativeAI(
#         google_api_key=gemini_api_key,
#         model="gemini-pro",
#         temperature=0.2,
#         convert_system_message_to_human=True
#     )
    
#     agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
#     prompt = (
#         """
#         Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

#         1. If the query requires a table, format your answer like this:
#            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

#         2. For a bar chart, respond like this:
#            {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         3. For a line chart, your reply should look like this:
#            {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         4. For a pie chart, respond like this:
#            {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

#         5. For a scatter plot, respond like this:
#            {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

#         6. For a plain question that doesn't need a chart or table, your response should be:
#            {"answer": "Your answer goes here"}

#         7. If the answer is not known or available, respond with:
#            {"answer": "I do not know."}

#         Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
#         For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

#         Now, let's tackle the query step by step. Here's the query for you to work on: 
#         """
#         + query
#     )
#     result = agent.run(prompt)
#     return result

# # Function to decode the response
# def decode_response(response: str) -> dict:
#     if isinstance(response, dict):
#         return response
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {"answer": response}

# # Function to write the answer
# def write_answer(response_dict: dict):
#     if "answer" in response_dict:
#         answer = response_dict["answer"]
#         st.write(answer)
#         speak_text(answer)

#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="bar_chart")

#     if "line" in response_dict:
#         data = response_dict["line"]
#         fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="line_chart")

#     if "pie" in response_dict:
#         data = response_dict["pie"]
#         fig = px.pie(names=data['labels'], values=data['values'])
#         st.plotly_chart(fig, key="pie_chart")

#     if "scatter" in response_dict:
#         data = response_dict["scatter"]
#         fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
#         st.plotly_chart(fig, key="scatter_chart")

#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# # Predefined queries
# HARD_CODED_QUERIES = [
#     "What is the total sales for each product?",
#     "Show me the top 5 products with the highest sales.",
#     "Which region has the most sales?",
#     "What is the monthly trend of sales for the year 2023?",
# ]

# # Streamlit UI setup
# st.title("Car Sales Advertisements with Voice Assistance")
# st.sidebar.header("Upload CSV Files")
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Initialize session state for conversation history
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []

# input_text = None
# if input_csvs:
#     selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)
#     df = pd.read_csv(input_csvs[selected_index])

#     st.subheader(f"Preview of {selected_file}")
#     st.dataframe(df.head(10))

#     # Display hardcoded queries as buttons
#     st.subheader("Quick Queries:")
#     col1, col2 = st.columns(2)
#     for i, query in enumerate(HARD_CODED_QUERIES):
#         if i % 2 == 0:
#             if col1.button(query):
#                 input_text = query
#         else:
#             if col2.button(query):
#                 input_text = query

#     # Voice or Text Query Input
#     query_mode = st.radio("How would you like to input your query?", ["Type", "Speak"])

#     if query_mode == "Type":
#         user_input = st.text_area("What do you want to know?", placeholder="Type your question here...")
#         if user_input:
#             input_text = user_input
#     else:
#         if st.button("Speak Now"):
#             input_text = listen_to_query()

#     # Process query if available
#     if input_text:
#         st.info(f"Your Query: {input_text}")
#         with st.spinner("Processing..."):
#             result = chat_with_csv(df, input_text)
#         decoded_response = decode_response(result)
#         write_answer(decoded_response)

#         # Store conversation history
#         st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})

#     # Display conversation history
#     if st.session_state.conversation_history:
#         st.subheader("Conversation History:")
#         for idx, convo in enumerate(st.session_state.conversation_history):
#             st.markdown(f"**Q{idx+1}:** {convo['query']}")
#             write_answer(convo['response'])
# else:
#     st.sidebar.info("Upload a CSV file to get started.")


#with 2.0
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json
import speech_recognition as sr
import pyttsx3
import threading
import queue

# Loading environment variables
load_dotenv()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Create a queue for communication between threads
speech_queue = queue.Queue()

# Function for text-to-speech
def speak_text(text):
    speech_queue.put(text)

def process_speech():
    while True:
        text = speech_queue.get()
        if text == "STOP":
            break
        tts_engine.say(text)
        tts_engine.runAndWait()

# Start the speech processing in a separate thread
speech_thread = threading.Thread(target=process_speech, daemon=True)
speech_thread.start()

# Function for speech-to-text
def listen_to_query():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak your query now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Could not request results from the speech recognition service.")
            return None

# Function to chat with CSV data
def chat_with_csv(df, query):
    gemini_api_key = os.environ.get('GOOGLE_API_KEY')
    if not gemini_api_key:
        return "GOOGLE_API_KEY environment variable not set."
    
    llm = ChatGoogleGenerativeAI(
        google_api_key=gemini_api_key,
        model="gemini-1.5-flash",
        temperature=0.2,
        convert_system_message_to_human=True
    )
    
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

        3. For a line chart, your reply should look like this:
           {"line": {"x": ["A", "B", "C", ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

        4. For a pie chart, respond like this:
           {"pie": {"labels": ["A", "B", "C", ...], "values": [25, 24, 10, ...]}}

        5. For a scatter plot, respond like this:
           {"scatter": {"x": [1, 2, 3, ...], "y": [25, 24, 10, ...], "x_label": "X Axis Label", "y_label": "Y Axis Label"}}

        6. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        7. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )
    result = agent.run(prompt)
    return result

# Function to decode the response
def decode_response(response: str) -> dict:
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"answer": response}

# Function to write the answer
def write_answer(response_dict: dict):
    if "answer" in response_dict:
        answer = response_dict["answer"]
        st.write(answer)
        speak_text(answer)

    if "bar" in response_dict:
        data = response_dict["bar"]
        fig = px.bar(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
        st.plotly_chart(fig, key=f"bar_chart_{str(data['x'])}_{str(data['y'])}")

    if "line" in response_dict:
        data = response_dict["line"]
        fig = px.line(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
        st.plotly_chart(fig, key=f"line_chart_{str(data['x'])}_{str(data['y'])}")

    if "pie" in response_dict:
        data = response_dict["pie"]
        fig = px.pie(names=data['labels'], values=data['values'])
        st.plotly_chart(fig, key=f"pie_chart_{str(data['labels'])}_{str(data['values'])}")

    if "scatter" in response_dict:
        data = response_dict["scatter"]
        fig = px.scatter(x=data['x'], y=data['y'], labels={'x': data['x_label'], 'y': data['y_label']})
        st.plotly_chart(fig, key=f"scatter_chart_{str(data['x'])}_{str(data['y'])}")

    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

# Predefined queries
HARD_CODED_QUERIES = [
    "What is the total sales for each product?",
    "Show me the top 5 products with the highest sales.",
    "Which region has the most sales?",
    "What is the monthly trend of sales for the year 2021?",
]

# Streamlit UI setup
st.title("Nike Sales data analyzer with Voice Assistance")
st.sidebar.header("Upload CSV Files")
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

input_text = None
if input_csvs:
    selected_file = st.sidebar.selectbox("Select a CSV file to display", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)
    df = pd.read_csv(input_csvs[selected_index])

    st.subheader(f"Preview of {selected_file}")
    st.dataframe(df.head(10))

    # Display hardcoded queries as buttons
    st.subheader("FAQ")
    col1, col2 = st.columns(2)
    for i, query in enumerate(HARD_CODED_QUERIES):
        if i % 2 == 0:
            if col1.button(query):
                input_text = query
        else:
            if col2.button(query):
                input_text = query

    # Voice or Text Query Input
    query_mode = st.radio("How would you like to input your query?", ["Type", "Speak"])

    if query_mode == "Type":
        user_input = st.text_area("What do you want to know?", placeholder="Type your question here...")
        if user_input:
            input_text = user_input
    else:
        if st.button("Speak Now"):
            input_text = listen_to_query()

    # Process query if available
    if input_text:
        st.info(f"Your Query: {input_text}")
        with st.spinner("Processing..."):
            result = chat_with_csv(df, input_text)
        decoded_response = decode_response(result)
        write_answer(decoded_response)

        # Store conversation history
        st.session_state.conversation_history.append({"query": input_text, "response": decoded_response})

    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History:")
        for idx, convo in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{idx+1}:** {convo['query']}")
            write_answer(convo['response'])
else:
    st.sidebar.info("Upload a CSV file to get started.")
