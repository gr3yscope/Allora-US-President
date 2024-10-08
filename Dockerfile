# Use an official Python runtime as the base image
FROM python:3.11-slim AS project_env

# Install curl
RUN apt-get update && apt-get install -y curl

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

FROM project_env

COPY . /app/

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "app:app"]
