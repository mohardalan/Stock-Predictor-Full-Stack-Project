# Use the official TensorFlow Docker image as the base image
#FROM tensorflow/tensorflow:latest
FROM tensorflow/tensorflow:2.13.0
# Set the working directory inside the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install the required libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Expose the Flask port
EXPOSE 5000
# Command to run the Flask application
CMD ["python", "app.py"]