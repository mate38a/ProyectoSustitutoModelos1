#!/usr/bin/env python3
"""
predict.py

Script para hacer predicciones de trip_duration con un modelo entrenado.

Uso:
    python predict.py ruta_al_test_csv ruta_al_output_csv

El CSV de test debe contener todas las columnas excepto 'trip_duration',
pero debe incluir 'id' para identificar cada fila.
"""

import sys
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # Guardamos id para salida
    ids = df['id'].copy()

    # Convertir fechas a datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    # Extraer features temporales
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month

    # Borrar las columnas originales
    df = df.drop(columns=['pickup_datetime', 'dropoff_datetime'])

    # Codificar variables categóricas igual que en train
    for col in ['vendor_id', 'store_and_fwd_flag']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Eliminar id para predecir
    df = df.drop(columns=['id'])

    return df, ids

def main():
    if len(sys.argv) != 3:
        print("Uso: python predict.py ruta_al_test_csv ruta_al_output_csv")
        sys.exit(1)

    test_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Cargar datos
    df = pd.read_csv(test_csv)
    print(f"Datos de test cargados: {df.shape[0]} filas")

    # Preprocesar
    X, ids = preprocess(df)
    print("Preprocesamiento realizado")

    # Cargar modelo
    model = joblib.load('model.joblib')

    # Predecir
    preds = model.predict(X)

    # Crear DataFrame para salida
    df_out = pd.DataFrame({'id': ids, 'trip_duration': preds})

    # Guardar CSV
    df_out.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en {output_csv}")

if __name__ == '__main__':
    main()
