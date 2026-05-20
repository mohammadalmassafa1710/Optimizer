FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies (including coinor-cbc in case standard pulp solver requires it, 
# although PuLP on linux usually downloads CBC automatically, installing coinor-cbc ensures maximum compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port and run server
EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
