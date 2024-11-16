# Use Python 3.11 for better compatibility
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Upgrade pip and install dependencies with root action suppressed
RUN pip install --upgrade pip --root-user-action=ignore

# Copy the requirements file first (better cache management)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy the entire project
COPY . /app

# Ensure the models directory exists in the container
RUN mkdir -p /app/models

# Set environment variables for Flask
ENV FLASK_ENV=production
ENV FLASK_APP=app.app

# Expose port 5000 for Flask
EXPOSE 5000

# Use Gunicorn to run the app
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:5000"]