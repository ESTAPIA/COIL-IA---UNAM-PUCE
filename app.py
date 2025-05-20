from flask import Flask, render_template, request
import joblib
import numpy as np

# Inicialización de la aplicación
app = Flask(__name__)

# Mensaje de depuración antes de cargar el modelo
print("Iniciando la aplicación Flask...")

# Cargar el modelo optimizado
try:
    modelo = joblib.load('modelo_random_forest_optimizado.pkl')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

# Página inicial con formulario
@app.route('/')
def index():
    return render_template('formulario.html')

# Ruta que recibe y procesa los datos del formulario
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Obtener valores desde el formulario
        datos = [
            float(request.form['Embarazos']),
            float(request.form['Glucosa']),
            float(request.form['PresionArterial']),
            float(request.form['EspesorPiel']),
            float(request.form['Insulina']),
            float(request.form['IMC']),
            float(request.form['DiabetesPedigree']),
            float(request.form['Edad'])
        ]

        # Convertir a array y hacer predicción
        entrada = np.array([datos])
        resultado = modelo.predict(entrada)[0]

        mensaje = "Resultado: Positivo para Diabetes." if resultado == 1 else "Resultado: Negativo para Diabetes."
        return f"<h2>{mensaje}</h2><br><a href='/'>Volver al formulario</a>"

    except Exception as e:
        return f"<h2>Error en la predicción: {str(e)}</h2>"


# Ejecutar la app en local
if __name__ == '__main__':
    app.run(debug=True)
