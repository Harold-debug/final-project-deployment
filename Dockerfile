# Use Python base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the desired port (optional, adjust if necessary)
EXPOSE 8000

# Start the application with Gunicorn
CMD ["gunicorn", "app.app:app", "--workers", "1", "--timeout", "1200"]