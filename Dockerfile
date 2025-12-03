# Use a lightweight Python base image (matches your Python 3.11 environment)
FROM python:3.11-slim

# Set the working directory to the project root
WORKDIR /app

# Install system dependencies (if needed for ML libraries like XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching during builds)
COPY streamlit_app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including models, data, and app)
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit (optional, for headless mode)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the app (from project root, as per README)
CMD ["streamlit", "run", "streamlit_app/app.py"]
