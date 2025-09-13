FROM python:3.11-slim
WORKDIR /app
COPY .. /app
RUN pip install --no-cache-dir fastapi uvicorn celery pydantic redis
CMD ["celery", "-A", "apps.workers.celery_app", "worker", "-P", "solo", "--loglevel=INFO"]
