# Proyecto de Predicción de Duración de Viajes en Taxis

Este proyecto tiene como objetivo predecir el tiempo de duración de un viaje en taxi utilizando varios datos, como la ubicación de recogida y destino, la cantidad de pasajeros, entre otros.

## Fase 1: Entrenamiento del Modelo

<details>
  <summary>Haz clic aquí para ver los detalles de la Fase 1</summary>
  
  En la **Fase 1**, el objetivo fue entrenar un modelo de **Machine Learning** para predecir la duración de un viaje en taxi, basado en diferentes características. Los pasos seguidos fueron los siguientes:

  ### Pasos realizados:

  1. **Descargar los datos desde Kaggle**:
     - Los datos fueron obtenidos desde la plataforma Kaggle, que contiene información sobre viajes de taxis, incluyendo la duración del viaje y las características del mismo (ubicación de recogida, destino, cantidad de pasajeros, etc.).
     - Se necesitó un API key de Kaggle para realizar la descarga de los archivos.

  2. **Subir los datos a Google Colab**:
     - Se subieron los archivos CSV a **Google Colab** para entrenar el modelo en la nube, aprovechando los recursos disponibles y facilitando la ejecución sin limitaciones locales.
   
  3. **Preprocesamiento de los datos**:
     - Se limpiaron los datos, se eliminaron los valores nulos y se realizaron las conversiones necesarias, como convertir las fechas en características que puedan ser procesadas por el modelo.

  4. **Entrenamiento del modelo**:
     - Se utilizó un modelo de **Random Forest Regressor** para predecir la duración del viaje.
     - El modelo fue entrenado utilizando las características disponibles (ubicación, cantidad de pasajeros, etc.).

  5. **Guardar el modelo entrenado**:
     - Una vez entrenado el modelo, se guardó utilizando **joblib** para poder usarlo en etapas posteriores de predicción.

  6. **Generación de predicciones**:
     - Se utilizaron los datos de prueba (test.csv) para generar las predicciones de duración de los viajes y se guardaron en un archivo **predictions.csv**.
  
  ### Código utilizado en Google Colab:

  ```python
  # Cargar librerías
  import pandas as pd
  from sklearn.ensemble import RandomForestRegressor
  import joblib

  # Cargar los datos de entrenamiento
  train_data = pd.read_csv("train.csv")

  # Preprocesamiento de los datos
  X = train_data.drop(columns=["trip_duration"])
  y = train_data["trip_duration"]

  # Entrenamiento del modelo
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, y)

  # Guardar el modelo entrenado
  joblib.dump(model, "taxi_model.joblib")

  # Cargar los datos de prueba
  test_data = pd.read_csv("test.csv")

  # Hacer


