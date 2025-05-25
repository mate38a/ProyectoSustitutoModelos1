#!/usr/bin/env python3
"""
train.py

Entrena un modelo RandomForestRegressor para predecir trip_duration
a partir de datos de taxi.

Uso:
    python train.py ruta_al_train_csv

Salida:
    model.joblib con el modelo entrenado.
"""

import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # Eliminar columna id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Convertir fechas a datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    # Extraer features temporales
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month

    # Borrar las columnas originales de fecha para evitar fugas
    df = df.drop(columns=['pickup_datetime', 'dropoff_datetime'])

    # Codificar variables categóricas
    for col in ['vendor_id', 'store_and_fwd_flag']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]) # codificar las variables categóricas en números enteros (0, 1, 2, ...) y guardarlas en la columna 'col' del DataFrame 'df'

    return df

def main():
    if len(sys.argv) != 2:
        print("Uso: python train.py ruta_al_train_csv")
        sys.exit(1) # si no se pasan argumentos, muestra el mensaje de error y sale del programa

    train_csv = sys.argv[1] # ruta al csv de entrenamiento

    # Cargar datos
    df = pd.read_csv(train_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Preprocesar
    df = preprocess(df)
    print("Preprocesamiento realizado")

    # Separar features y target
    if 'trip_duration' not in df.columns:
        print("Error: la columna 'trip_duration' no está en el dataset")
        sys.exit(1)

    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']

    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("Modelo entrenado")

    # Guardar modelo
    joblib.dump(model, 'data/model.joblib')
    print("Modelo guardado en model.joblib")

if __name__ == '__main__':
    main()
