FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# System dependencies for OpenCV and Kafka
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

CMD ["./start.sh"]