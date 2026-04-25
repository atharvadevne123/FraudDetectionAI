FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/opt/models PORT=8001 FLASK_ENV=production PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8001"]
