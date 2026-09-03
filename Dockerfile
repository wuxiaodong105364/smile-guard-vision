FROM python:3.11-slim

WORKDIR /app

COPY vision-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ./config.py
COPY src ./src
COPY data/models ./data/models
COPY vision-service ./vision-service

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "vision-service.app:app", "--host", "0.0.0.0", "--port", "8000"]
