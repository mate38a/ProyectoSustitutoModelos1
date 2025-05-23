# Usamos imagen base oficial de Python 3.9 slim para tener un entorno ligero
FROM python:3.9-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los scripts al contenedor
COPY train.py .
COPY predict.py .

# Copiamos el archivo de requerimientos (lo creamos con las librerías necesarias)
COPY requirements.txt .

# Instalamos las librerías necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el directorio /app para que los datos se puedan montar desde fuera
VOLUME ["/app/data"]

# Por defecto ejecutamos bash para poder usar el contenedor interactivamente
CMD ["bash"]
