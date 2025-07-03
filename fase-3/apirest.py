# apirest.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from predict import predict
from train import train_model

app = Flask(__name__)

# Rutas de archivos dentro del contenedor
MODEL_PATH = "/app/data/model.joblib"
DATA_PATH = "/app/data/train.csv"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint para realizar predicciones con el modelo almacenado.
    Entrada: JSON con datos de entrada (id, pickup_datetime, etc.).
    Salida: JSON con las predicciones.
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({"error": "Se requiere una lista de datos en JSON"}), 400

        # Convertir lista de datos a DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({"error": "Datos vacíos"}), 400

        # Llamar a predict.py para generar predicciones
        predictions = predict(df, MODEL_PATH)
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_endpoint():
    """
    Endpoint para entrenar el modelo con datos proporcionados o estándar.
    Entrada: Opcionalmente, un archivo CSV con datos de entrenamiento.
    Salida: Mensaje de confirmación.
    """
    try:
        if 'file' in request.files:
            file = request.files['file']
            file.save(DATA_PATH)
        elif not os.path.exists(DATA_PATH):
            return jsonify({"error": "No se proporcionó archivo y no existe train.csv"}), 400

        train_model(DATA_PATH, MODEL_PATH)
        return jsonify({"message": "Modelo entrenado y guardado correctamente"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)