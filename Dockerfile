
FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN pip install --no-cache-dir --timeout=300 -r requirement.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]
