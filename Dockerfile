FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords punkt

# Expose port
EXPOSE 8000

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.app:app"]
