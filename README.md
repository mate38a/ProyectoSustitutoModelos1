# Proyecto de Predicción de Duración de Viajes en Taxis

Este proyecto tiene como objetivo predecir la **duración de un viaje en taxi** utilizando un modelo de **Machine Learning**. Para ello, usamos datos reales de viajes de taxis en Nueva York. El modelo que creamos predice cuánto durará un viaje en taxi basado en diferentes características, como la ubicación de recogida, destino, la cantidad de pasajeros, etc.

Este proyecto está dividido en diferentes fases. En esta **Fase 1**, nos enfocamos en el entrenamiento del modelo para predecir la duración del viaje.

## Tabla de Contenidos

<details>
  <summary>Haz clic aquí para ver las fases del proyecto</summary>
  
  - **Fase 1**: Entrenamiento del modelo y generación de predicciones.
  - **Fase 2**: Contenerización con Docker.
  - **Fase 3**: Implementación de la API REST.

</details>

---

## Fase 1: Entrenamiento del Modelo

En esta fase, entrenamos un **modelo de Machine Learning** para predecir la duración de un viaje en taxi. Los datos utilizados para el entrenamiento se descargaron desde [Kaggle - NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data).

### Pasos realizados en la Fase 1:

<details>
  <summary>Haz clic aquí para ver los detalles de la Fase 1</summary>

  ### 1. Descargar los datos desde Kaggle

   Para comenzar, **descargamos los datos** desde Kaggle. Los archivos de datos (`train.csv` y `test.csv`) contienen información sobre los viajes de taxi en Nueva York, como el tiempo de duración, las ubicaciones de recogida y destino, y la cantidad de pasajeros.

   Los datos se descargaron desde [este enlace de Kaggle](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data). Para descargar los archivos, es necesario tener una cuenta en **Kaggle**.

### 2. Subir los datos a Google Colab

   Después de descargar los datos, **subimos los archivos** a Google Colab, donde entrenamos el modelo. Google Colab es una herramienta en línea que permite ejecutar código Python sin necesidad de instalar nada en tu computadora.

   Para cargar los archivos en Colab, utilizamos este código:

   ```python
   from google.colab import files
   files.upload()  # Esto permite que subas los archivos CSV a Colab



