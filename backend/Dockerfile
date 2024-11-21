# Use Python 3.12 slim image as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV, TensorFlow, Gunicorn, and OpenCV dependencies
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*  # Clean up the apt cache to reduce image size

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the necessary Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app

# Expose the port that Gunicorn will run on (default: 8000)
EXPOSE 8000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the Flask application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
