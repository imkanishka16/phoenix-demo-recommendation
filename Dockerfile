# # Use Python base image
# FROM python:3.12

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     default-jdk \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*
    
# # Set working directory
# WORKDIR /flask-app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt requirements.txt

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Expose port 80
# EXPOSE 80

# # Command to run the application
# CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:80", "app2:app"]


# Use Python base image
FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /flask-app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 80
EXPOSE 80

# Command to run the application with increased timeout
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:80", "--timeout", "120", "app2:app"]