# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# requirements.txt만 먼저 복사
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ./Modeling /app

# Define the default command
CMD ["python", "main.py"\
, "--host", "phoenix-mlops-db.cjywquoscxz3.ap-northeast-2.rds.amazonaws.com"\
, "--database", "ethic_db"\
, "--user", "admin"\
, "--password", "moasis0104"\
, "--table", "train"\
, "--epochs", "1"]