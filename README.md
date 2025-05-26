# Proyecto de Predicción de Duración de Viajes en Taxis

![Proyecto](https://img.shields.io/badge/Proyecto-Taxi%20Prediction-blue)  
Este proyecto tiene como objetivo predecir la **duración de un viaje en taxi** utilizando un modelo de **Machine Learning**. Para ello, usamos datos reales de viajes de taxis en Nueva York. El modelo que creamos predice cuánto durará un viaje en taxi basado en diferentes características, como la ubicación de recogida, destino, la cantidad de pasajeros, etc.

Este proyecto está dividido en diferentes fases. En esta **Fase 1**, nos enfocamos en el entrenamiento del modelo para predecir la duración del viaje.

---

## 📑 Tabla de Contenidos

<details>
  <summary>Haz clic aquí para ver las fases del proyecto</summary>
  
  - **Fase 1**: Entrenamiento del modelo y generación de predicciones.
      ---
    ## Fase 1: Entrenamiento del Modelo
    
    En esta fase, entrenamos un **modelo de Machine Learning** para predecir la duración de un viaje en taxi. Los datos utilizados para el entrenamiento se descargaron desde [Kaggle - NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data).
    
    ### 🛠️ Pasos realizados en la Fase 1:
    
    <details>
      <summary>Haz clic aquí para ver los detalles de la Fase 1</summary>
    
    ### 1. Descargar los datos desde Kaggle
       - Se descargaron los datos de la competencia **"NYC Taxi Trip Duration"** en Kaggle. Los archivos contienen información sobre los viajes de taxi en Nueva York, incluyendo el tiempo de duración del viaje, la ubicación de recogida y destino, y la cantidad de pasajeros.
       - Los archivos descargados fueron **`train.csv`** y **`test.csv`**.
    
    ### 2. Subir los datos a Google Colab
       - Los archivos **`train.csv`** y **`test.csv`** fueron subidos a **Google Colab**, una herramienta en línea que permite ejecutar código Python sin necesidad de instalar nada localmente.
    
    ### 3. Exploración Inicial de los Datos
       - Se cargaron los datos y se realizaron las primeras exploraciones para entender el contenido del dataset, visualizando las primeras filas y obteniendo un resumen de la información y las estadísticas descriptivas.
       - Observamos que el dataset contiene 1,447,614 filas y 11 columnas, incluyendo el **ID del viaje**, **ubicaciones de recogida y destino**, **cantidad de pasajeros**, y **duración del viaje**.
    
    ### 4. Limpieza de Datos
       - Se eliminaron filas con **valores atípicos** en la duración del viaje. Específicamente, se eliminaron los viajes cuya duración era inferior a 1 minuto o superior a 2 horas (7200 segundos), ya que estos no eran representativos.
    
    ### 5. Feature Engineering
       - Se creó una nueva característica llamada **`distance`**, que calcula la distancia entre la ubicación de recogida y la ubicación de destino utilizando la fórmula **Haversine**. Esta fórmula calcula la distancia entre dos puntos geográficos a partir de sus coordenadas de latitud y longitud.
    
    ### 6. Selección de Características
       - Se seleccionaron las características relevantes para el modelo, que incluían las coordenadas de recogida y destino, y la distancia calculada. Estas características fueron utilizadas para entrenar el modelo.
    
    ### 7. Dividir en Entrenamiento y Validación
       - El conjunto de datos fue dividido en dos partes: un conjunto de entrenamiento y un conjunto de validación. Esto nos permitió entrenar el modelo con una parte de los datos y evaluar su rendimiento con la otra parte.
    
    ### 8. Entrenamiento del Modelo
       - Se entrenó un **Random Forest Regressor**, un modelo de Machine Learning que es efectivo para manejar datos complejos como los de este proyecto. Este modelo predice la duración del viaje en taxi basándose en las características de entrada.
    
    ### 9. Evaluación del Modelo
       - Después de entrenar el modelo, se evaluó su rendimiento utilizando el **RMSE** (Root Mean Squared Error), que nos muestra el error promedio de las predicciones en segundos. El modelo obtuvo un **RMSE de 380 segundos**, lo que significa que las predicciones del modelo tienen un error promedio de 380 segundos.
    
    ### 10. Guardar el Modelo
       - Una vez entrenado el modelo, se guardó en un archivo **`modelo_fase1.pkl`** usando **joblib** para poder usarlo en futuras predicciones sin necesidad de reentrenarlo.
    
    ---
    
    ### 📊 Conclusiones de la Fase 1
    
    En esta fase, logramos entrenar un modelo de **Random Forest** que puede predecir la **duración de un viaje en taxi**. Los puntos clave de esta fase son:
    
    1. **Entrenamos el modelo** utilizando los datos descargados de **Kaggle**.
    2. **Realizamos un preprocesamiento** y limpieza de los datos para eliminar los valores atípicos y asegurarnos de que los datos fueran útiles para el modelo.
    3. **Creamos nuevas características** (como la distancia entre la recogida y destino) para mejorar la precisión de las predicciones.
    4. **Evaluamos el modelo** utilizando **RMSE** y obtuvimos un buen desempeño con un error promedio de 380 segundos.
    5. Finalmente, **guardamos el modelo** para poder reutilizarlo en el futuro.
  - **Fase 2**: Contenerización con Docker.
  - **Fase 3**: Implementación de la API REST.

</details>



## 📚 Autor

**Luis Mateo Ochoa Agudelo**  
**CC:** 104132979  
**Correo:** mateo.ochoa1@udea.edu.co  
**Carrera:** Ingeniería de Sistemas





