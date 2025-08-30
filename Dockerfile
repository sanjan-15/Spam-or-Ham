# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt

COPY app/ ./app/
COPY saved_models/ ./saved_models/

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.app:app"]