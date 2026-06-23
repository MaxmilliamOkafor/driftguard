# syntax=docker/dockerfile:1
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY driftguard ./driftguard
COPY dashboard ./dashboard
COPY data ./data
EXPOSE 8000 8501
CMD ["uvicorn", "driftguard.serve:app", "--host", "0.0.0.0", "--port", "8000"]
