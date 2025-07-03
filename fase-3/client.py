# client.py
import requests
import json

BASE_URL = "http://localhost:5000"

def test_predict():
    """
    Prueba el endpoint /predict con un dato de ejemplo.
    """
    sample_data = [
        {
            "id": "id12345",
            "vendor_id": "1",
            "pickup_datetime": "2016-03-14 17:24:55",
            "dropoff_datetime": "2016-03-14 17:32:30",
            "passenger_count": 1,
            "pickup_longitude": -73.98215,
            "pickup_latitude": 40.767937,
            "dropoff_longitude": -73.96463,
            "dropoff_latitude": 40.765602,
            "store_and_fwd_flag": "N"
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print("Respuesta de /predict:", response.json())

def test_train():
    """
    Prueba el endpoint /train enviando un archivo CSV.
    """
    files = {'file': open('/app/data/train.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/train", files=files)
    print("Respuesta de /train:", response.json())

if __name__ == "__main__":
    print("Probando endpoint /predict...")
    test_predict()
    print("\nProbando endpoint /train...")
    test_train()