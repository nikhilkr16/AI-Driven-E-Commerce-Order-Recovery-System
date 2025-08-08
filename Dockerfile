# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Generate mock data if not exists\n\
if [ ! -f "data/orders.csv" ]; then\n\
    echo "Generating mock data..."\n\
    python data/mock_data_generator.py\n\
fi\n\
\n\
# Start services based on command\n\
case "$1" in\n\
    "api")\n\
        echo "Starting FastAPI server..."\n\
        uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
        ;;\n\
    "dashboard")\n\
        echo "Starting Streamlit dashboard..."\n\
        streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
        ;;\n\
    "both")\n\
        echo "Starting both API and Dashboard..."\n\
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &\n\
        streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
        ;;\n\
    *)\n\
        echo "Usage: $0 {api|dashboard|both}"\n\
        exit 1\n\
        ;;\n\
esac' > /app/start.sh

# Make startup script executable
RUN chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh", "both"]
