# Use Python 3.11 slim image for efficient deployments
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy UV lockfile, pyproject.toml, and README.md first for better caching
COPY uv.lock pyproject.toml README.md ./

# Install UV package manager
RUN pip install uv

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY simple_api.py ./
COPY .env* ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

# Run the simple API application
CMD ["uv", "run", "uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000"]
