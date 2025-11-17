# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install system dependencies as root
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create and configure non-root user
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

WORKDIR /app

# Copy requirements with proper ownership
COPY --chown=appuser:appuser requirements.txt .

# Switch to non-root user for Python packages
USER appuser

# Install Python dependencies in user space
RUN pip install --user --no-warn-script-location --upgrade pip && \
    pip install --user --no-warn-script-location -r requirements.txt

# Copy application files with proper ownership
COPY --chown=appuser:appuser . .

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application
# Use PORT environment variable if set, otherwise default to 5000
# Note: PORT is typically set by the hosting platform (Heroku, Render, etc.)
# Using shell form to allow environment variable substitution
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 2 --timeout 120 app:app"]

