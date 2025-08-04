# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_API_KEY=AIzaSyD83yavELzObxwvUrQy8VyZcZ9-TzlFWlU
ENV MONGO_URI=mongodb+srv://ayush:123%40123@cluster0.rrv507u.mongodb.net/Project0?retryWrites=true&w=majority
ENV MONGO_DB_NAME=Project0
ENV DEBUG_MODE=False
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Run the application
CMD ["python", "main.py"]