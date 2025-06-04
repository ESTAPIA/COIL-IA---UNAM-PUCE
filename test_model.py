import os
import joblib

# Utilizamos el modelo XGBoost disponible
file_path = 'modelo_xgboost_optimizado.pkl'

if os.path.exists(file_path):
    print(f"El archivo {file_path} existe.")
    print(f"Tamaño del archivo: {os.path.getsize(file_path)} bytes")
    try:
        modelo = joblib.load(file_path)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
else:
    print(f"El archivo {file_path} no existe.")
