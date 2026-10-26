FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN bash scripts/download_models.sh && python scripts/train_digit_model.py --strings 2000
ENV TESSDATA_PREFIX=/app/models/tessdata
# no network needed at runtime: the service never calls an external AI API
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
