FROM python:3.9-slim

WORKDIR /app

# Copy requirements from parent directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Frontend code
COPY front/ .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
