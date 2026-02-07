FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (added curl for health check)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create references.json if it doesn't exist
RUN if [ ! -f references.json ]; then echo '[]' > references.json; fi

# Expose port 7860 (REQUIRED by Hugging Face Spaces - changed from 8001)
EXPOSE 7860

# Health check (fixed to use curl and port 7860)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application (CHANGED PORT to 7860 for Hugging Face)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]