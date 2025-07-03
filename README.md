# Proyecto de Predicción de Duración de Viajes en Taxis

![Proyecto](https://img.shields.io/badge/Proyecto-Taxi%20Prediction-blue)  
Este proyecto tiene como objetivo predecir la **duración de un viaje en taxi** utilizando un modelo de **Machine Learning**. Para ello, usamos datos reales de viajes de taxis en Nueva York. El modelo que creamos predice cuánto durará un viaje en taxi basado en diferentes características, como la ubicación de recogida, destino, la cantidad de pasajeros, etc.

Este proyecto está dividido en diferentes fases. En esta documentación, cubriremos la **Fase 1** y la **Fase 2** del proyecto como un tutorial detallado.

---

## 📑 Fases del Proyecto

- **Fase 1**: Entrenamiento del modelo y generación de predicciones.  
- **Fase 2**: Contenerización con Docker.  
- **Fase 3**: Implementación de la API REST (próximamente).

---

## 🛠️ Fase 1: Entrenamiento del Modelo

En esta fase, entrenamos un **modelo de Machine Learning** para predecir la duración de un viaje en taxi. A continuación, te explicamos cómo montar el proyecto paso a paso.

### 1. Descargar los datos desde Kaggle
- Ve a [Kaggle - NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data) y descarga los datos de la competencia **"NYC Taxi Trip Duration"**.  
- Los archivos que necesitas son **`train.csv`** y **`test.csv`**. Estos contienen información como el tiempo de duración del viaje, ubicaciones de recogida y destino, y cantidad de pasajeros.

### 2. Subir los datos a Google Colab
- Abre [Google Colab](https://colab.research.google.com/) en tu navegador.  
- Haz clic en "Archivo" > "Subir cuaderno" o crea uno nuevo.  
- En el panel lateral izquierdo, selecciona el ícono de "Subir archivo" y sube **`train.csv`** y **`test.csv`**. Esto te permitirá trabajar con los datos en la nube sin instalar nada localmente.

### 3. Exploración Inicial de los Datos
- Usa Python en Google Colab para cargar los datos con la librería `pandas`. Por ejemplo:  
  ```python
  import pandas as pd
  train_data = pd.read_csv('train.csv')
  test_data = pd.read_csv('test.csv')
  print(train_data.head())
  print(train_data.describe())
  ```
- Verás que el dataset tiene 1,447,614 filas y 11 columnas, con datos como el **ID del viaje**, **coordenadas de recogida y destino**, **cantidad de pasajeros** y **duración del viaje**.

### 4. Limpieza de Datos
- Filtra los valores atípicos en la duración del viaje. Por ejemplo, elimina viajes menores a 1 minuto o mayores a 2 horas (7200 segundos):  
  ```python
  train_data = train_data[(train_data['trip_duration'] >= 60) & (train_data['trip_duration'] <= 7200)]
  ```

### 5. Feature Engineering
- Calcula la distancia entre recogida y destino usando la fórmula **Haversine**. Agrega esta nueva columna al dataset:  
  ```python
  from math import radians, sin, cos, sqrt, atan2
  def haversine(lon1, lat1, lon2, lat2):
      R = 6371  # Radio de la Tierra en km
      lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
      dlon = lon2 - lon1
      dlat = lat2 - lat1
      a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
      c = 2 * atan2(sqrt(a), sqrt(1-a))
      return R * c
  train_data['distance'] = train_data.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
  ```

### 6. Selección de Características
- Elige las columnas relevantes para el modelo: coordenadas de recogida (`pickup_longitude`, `pickup_latitude`), destino (`dropoff_longitude`, `dropoff_latitude`) y la distancia calculada (`distance`).

### 7. Dividir en Entrenamiento y Validación
- Divide los datos en conjuntos de entrenamiento y validación (por ejemplo, 80% entrenamiento, 20% validación):  
  ```python
  from sklearn.model_selection import train_test_split
  X = train_data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'distance']]
  y = train_data['trip_duration']
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### 8. Entrenamiento del Modelo
- Entrena un **Random Forest Regressor**:  
  ```python
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  ```

### 9. Evaluación del Modelo
- Evalúa el modelo con el **RMSE**:  
  ```python
  from sklearn.metrics import mean_squared_error
  y_pred = model.predict(X_val)
  rmse = mean_squared_error(y_val, y_pred, squared=False)
  print(f'RMSE: {rmse}')  # Obtuvimos un RMSE de ~380 segundos
  ```

### 10. Guardar el Modelo
- Guarda el modelo para usarlo después:  
  ```python
  import joblib
  joblib.dump(model, 'modelo_fase1.pkl')
  ```

---

### 📊 Conclusiones de la Fase 1
- Entrenamos un modelo **Random Forest** que predice la duración de viajes en taxi con un error promedio de 380 segundos (RMSE).  
- Preprocesamos los datos, eliminamos valores atípicos y creamos la característica `distance`.  
- Guardamos el modelo en `modelo_fase1.pkl` para reutilizarlo.

---

## 🐳 Fase 2: Contenerización con Docker

En esta fase, contenerizamos el proyecto con **Docker** para que sea portable y fácil de ejecutar en cualquier máquina. Aquí te explicamos cómo montarlo paso a paso.

### 1. Clonar el repositorio de GitHub
- Clona el repositorio donde está el código:  
  ```bash
  git clone https://github.com/tu_usuario/tu_repositorio.git
  cd tu_repositorio
  ```

### 2. Archivos necesarios
- Descarga `train.csv` y `test.csv` desde Google Drive (enlaces ficticios, reemplaza con los reales):  
  - [train.csv](https://drive.google.com/link/to/train.csv)  
  - [test.csv](https://drive.google.com/link/to/test.csv)  
- Crea una carpeta `data` y mueve los archivos ahí:  
  ```bash
  mkdir fase-1/data
  mv /ruta/a/train.csv fase-1/data/
  mv /ruta/a/test.csv fase-1/data/
  ```

### 3. Crear el archivo Dockerfile
- En la raíz del repositorio, crea un archivo llamado `Dockerfile` con este contenido:  
  ```
  FROM python:3.9-slim
  WORKDIR /app
  COPY . /app
  RUN pip install --no-cache-dir -r requirements.txt
  EXPOSE 5000
  CMD ["python", "train.py"]
  ```

### 4. Crear el archivo requirements.txt
- Crea un archivo `requirements.txt` con las dependencias:  
  ```
  pandas
  numpy
  scikit-learn
  joblib
  google-colab
  ```

### 5. Construir la imagen Docker
- Construye la imagen desde la raíz del repositorio:  
  ```bash
  docker build -t taxi_model .
  ```

### 6. Ejecutar el contenedor
- Ejecuta el contenedor, montando la carpeta `data` y corriendo las predicciones:  
  ```bash
  docker run --rm -v "$(pwd)/fase-1/data:/app/data" taxi_model python predict.py /app/data/test.csv /app/data/predictions.csv
  ```
  Esto ejecuta `predict.py`, usando `test.csv` para generar predicciones en `predictions.csv`.

---

### 📊 Conclusiones de la Fase 2
- Contenerizamos el proyecto con Docker, asegurando que funcione consistentemente en cualquier máquina con Docker instalado.  
- El proceso de entrenamiento y predicción ahora es portable y reproducible.

---

# Proyecto de Machine Learning - Fase 3

## Descripción
Este proyecto implementa un modelo predictivo para la duración de viajes de taxi (NYC Taxi Trip Duration). La Fase 3 crea una API REST para realizar predicciones y entrenar el modelo, desplegada en un contenedor Docker.

## Estructura del repositorio
- `fase-1/`: Notebook para entrenamiento y predicción.
- `fase-2/`: Scripts `predict.py`, `train.py`, y Dockerfile base.
- `fase-3/`: Scripts `apirest.py`, `client.py`, Dockerfile para la API, y datos.

## Requisitos
- Docker instalado.
- Archivos `train.csv` y `model.joblib` en `data/`.

## Instrucciones para ejecutar la Fase 3

### 1. Construir el contenedor
```bash
cd fase-3
docker build -t my-api-image:latest .

---

### Notas Finales
- Asegúrate de tener Docker instalado en tu máquina.  
- Si necesitas ajustar el código, edita los archivos en el repositorio o dentro del contenedor.

---

## 📚 Autor

**Luis Mateo Ochoa Agudelo**  
**CC:** 104132979  
**Correo:** mateo.ochoa1@udea.edu.co  
**Carrera:** Ingeniería de Sistemas
