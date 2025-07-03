# Dockerfile
# Extiende la imagen base
FROM python:3.9-slim

# Directorio de trabajo
WORKDIR /app

# Copiar scripts locales de fase-3
COPY fase-3/apirest.py /app/apirest.py
COPY fase-3/client.py /app/client.py
COPY fase-3/requirements.txt /app/requirements.txt

# Copiar scripts de fase-2 desde el directorio hermano
COPY fase-2/predict.py /app/predict.py
COPY fase-2/train.py /app/train.py
COPY fase-2/requirements.txt /app/requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask


# Copiar datos y modelo desde fase-1
COPY fase-1/data/train.csv /app/data/train.csv
COPY fase-1/data/model.joblib /app/data/model.joblib


# Exponer el puerto 5000
EXPOSE 5000

# Comando por defecto
CMD ["python", "/app/apirest.py"]
