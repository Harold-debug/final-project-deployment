# Use an official Python image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set the environment for Flask
ENV FLASK_ENV=production
ENV FLASK_APP=app.app

# Expose port 5000 for Flask
EXPOSE 5000

# Run the app using Gunicorn
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:5000"]