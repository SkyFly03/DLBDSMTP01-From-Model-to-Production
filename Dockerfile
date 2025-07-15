# Dockerfile
# ------------------------------------------------
# Container for running the refund classification Flask API.
# Installs dependencies and runs the app using gunicorn.
# ------------------------------------------------

FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port for the Flask API
EXPOSE 5000

# Run the API using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
