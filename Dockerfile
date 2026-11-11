# Lightweight Python container
FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Ensure sample database and index are initialized on build
RUN python3 seed_data.py --offline --force && \
    python3 search_engine.py --force --query "machine learning"

# Expose web server port
EXPOSE 8080

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Start zero-dependency web server
CMD ["python3", "app.py"]

