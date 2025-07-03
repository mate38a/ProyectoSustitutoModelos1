# Proyecto de Predicción de Duración de Viajes en Taxis 🚖

![Proyecto](https://img.shields.io/badge/Proyecto-Taxi%20Prediction-blue)

Este proyecto implementa un modelo predictivo para la **duración de viajes de taxis** en Nueva York (NYC Taxi Trip Duration). Está dividido en tres fases que abordan desde el entrenamiento del modelo hasta la implementación de una API REST. Todos los archivos necesarios ya están incluidos en el repositorio.

---

## 📦 Estructura del Repositorio

```
ProyectoSustitutoModelos1/
├── fase-1/                  # Entrenamiento y predicción del modelo
│   ├── data/
│   │   ├── train.csv        # Datos de entrenamiento
│   │   └── model.joblib     # Modelo entrenado
│   ├── train.py             # Script de entrenamiento
│   └── predict.py           # Script de predicción
│
├── fase-2/                  # Contenerización básica
│   ├── Dockerfile           # Dockerfile para fase 2
│   └── requirements.txt     # Dependencias comunes
│
└── fase-3/                  # API REST
    ├── apirest.py           # Servidor Flask para la API
    ├── client.py            # Cliente de ejemplo para consumir la API
    ├── Dockerfile           # Dockerfile para fase 3
    └── requirements.txt     # Dependencias para la API
```

---

## 🔧 Requisitos Previos

- Docker instalado en tu máquina
- Archivos `train.csv` y `model.joblib` en `fase-1/data/` (ya incluidos)

---

## 📑 Fases del Proyecto

### **Fase 1: Entrenamiento del Modelo**

#### Objetivo
Entrenar un modelo de Machine Learning para predecir la duración de viajes de taxis utilizando datos de NYC.

#### Archivos Incluidos
- `train.py`: Entrena el modelo con `train.csv`
- `predict.py`: Genera predicciones a partir del modelo guardado

#### Pasos Ejecutados
1. **Preprocesamiento**: Limpieza de valores atípicos (viajes menores a 1 minuto o mayores a 2 horas)
2. **Feature Engineering**: Cálculo de distancia entre coordenadas usando la fórmula Haversine
3. **Entrenamiento**: Modelo Random Forest Regressor con RMSE ~380 segundos
4. **Guardado**: Modelo serializado como `model.joblib`

#### Uso Local (Sin Docker)
```bash
cd fase-1
pip install -r ../fase-2/requirements.txt
python train.py
python predict.py data/test.csv data/predictions.csv
```

---

### **Fase 2: Contenerización Básica**

#### Objetivo
Empaquetar el código de entrenamiento y predicción en un contenedor Docker para ejecución portátil.

#### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "train.py"]
```

#### Uso con Docker
```bash
cd ProyectoSustitutoModelos1/
docker build -t taxi-model-fase2 -f fase-2/Dockerfile .
docker run --rm -v "$(pwd)/fase-1/data:/app/data" taxi-model-fase2 python predict.py data/test.csv data/predictions.csv
```

---

### **Fase 3: API REST con Flask**

#### Objetivo
Exponer el modelo como un servicio web mediante una API REST contenerizada.

#### Archivos Incluidos
- `apirest.py`: Servidor Flask con endpoints `/predict` y `/train`
- `client.py`: Cliente de ejemplo para solicitudes POST

#### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "apirest.py"]
```

#### Uso con Docker
```bash
# Construir la imagen
cd ProyectoSustitutoModelos1/
docker build -t taxi-api -f fase-3/Dockerfile .

# Ejecutar el contenedor
docker run -d -p 5000:5000 --name taxi-api-container taxi-api

# Enviar solicitud de predicción (ejemplo)
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"pickup_longitude": -73.994454, "pickup_latitude": 40.750042, "dropoff_longitude": -73.994454, "dropoff_latitude": 40.750042}'
```

---

## ✅ Resultados

| Fase | Descripción | Métrica |
|------|-------------|---------|
| Fase 1 | Modelo entrenado | RMSE ~380 segundos |
| Fase 2 | Contenedor funcional | ✅ |
| Fase 3 | API REST activa | ✅ |

---

## 📚 Autores

**Luis Mateo Ochoa Agudelo**  
**CC:** 104132979  
**Correo:** mateo.ochoa1@udea.edu.co  
**Carrera:** Ingeniería de Sistemas  
**Universidad de Antioquia**  

---

## 📌 Notas Adicionales

- **Fase 1**: El código ya está optimizado y listo para ejecutarse.
- **Fase 2**: El Dockerfile está configurado para ejecutar directamente `train.py` o `predict.py`.
- **Fase 3**: La API está lista para recibir solicitudes POST en `/predict` y `/train`.

---

Este README está diseñado para ser claro, directo y enfocado en la ejecución sin requerir pasos adicionales de creación de archivos. Todos los recursos necesarios ya están incluidos en el repositorio. ¡Solo clona, construye y ejecuta! 🚀