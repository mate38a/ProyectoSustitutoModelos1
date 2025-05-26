#!/usr/bin/env python3
"""
train.py

Entrena un modelo RandomForestRegressor para predecir trip_duration
usando datos desde /app/data y guarda el modelo en /app/data/model.joblib

Uso:
    python train.py /app/data/train.csv
"""

import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month

    df = df.drop(columns=['pickup_datetime', 'dropoff_datetime'])

    for col in ['vendor_id', 'store_and_fwd_flag']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df

def main():
    if len(sys.argv) != 2:
        print("Uso: python train.py /app/data/train.csv")
        sys.exit(1)

    train_csv = sys.argv[1]

    df = pd.read_csv(train_csv)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    df = preprocess(df)
    print("Preprocesamiento realizado")

    if 'trip_duration' not in df.columns:
        print("Error: columna 'trip_duration' no encontrada")
        sys.exit(1)

    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("Modelo entrenado")

    # Guardar modelo en ruta absoluta dentro del contenedor
    joblib.dump(model, '/app/data/model.joblib')
    print("Modelo guardado en /app/data/model.joblib")

if __name__ == '__main__':
    main()
