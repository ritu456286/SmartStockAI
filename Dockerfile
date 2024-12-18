# Use lightweight Python 3.11 image
FROM python:3.11-slim

# Install system dependencies for pyttsx3 and eSpeak
RUN apt-get update && \
    apt-get install -y espeak libespeak1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the service account JSON key to the container (adjust path for your local system)
# If you are on Windows, use Unix-style paths with relative paths, or copy the file explicitly into the container during the build
# COPY key.json key.json

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# ENV GOOGLE_APPLICATION_CREDENTIALS=key.json


# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install psycopg2-binary==2.9.9 && pip install --no-cache-dir -r requirements.txt

# Copy all project files to the container
COPY . .

# Expose port 8080 for Streamlit
EXPOSE 8080

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
