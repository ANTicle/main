# Pull the official base image
FROM python:3.9.6-slim-buster

# Set work directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# switch to root user
USER root

# Install dependencies
COPY ./requirements.txt /main/requirements.txt
RUN pip install -r requirements.txt

# Copy uncompiled source code
COPY . /main/