#!/usr/bin/env python3
"""
predict.py

Realiza predicciones usando modelo guardado en /app/data/model.joblib
lee datos desde /app/data y guarda predicciones en /app/data

Uso:
    python predict.py /app/data/test.csv /app/data/predictions.csv
"""

import sys
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    ids = df['id'].copy()

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Solo convertir dropoff_datetime si existe
    if 'dropoff_datetime' in df.columns:
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month

    # Eliminar columnas de fecha que existan
    cols_to_drop = ['pickup_datetime']
    if 'dropoff_datetime' in df.columns:
        cols_to_drop.append('dropoff_datetime')
    df = df.drop(columns=cols_to_drop)

    for col in ['vendor_id', 'store_and_fwd_flag']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    df = df.drop(columns=['id'])

    return df, ids


def main():
    if len(sys.argv) != 3:
        print("Uso: python predict.py /app/data/test.csv /app/data/predictions.csv")
        sys.exit(1)

    test_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df = pd.read_csv(test_csv)
    print(f"Datos test cargados: {df.shape[0]} filas")

    X, ids = preprocess(df)
    print("Preprocesamiento realizado")

    model = joblib.load('/app/data/model.joblib')
    preds = model.predict(X)

    df_out = pd.DataFrame({'id': ids, 'trip_duration': preds})
    df_out.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en {output_csv}")

if __name__ == '__main__':
    main()
