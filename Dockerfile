FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app_debug.py app.py
COPY test_api.py .

# Copy .env file
COPY .secretcontainer/ .secretcontainer/

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Run application
CMD ["python", "app.py"]