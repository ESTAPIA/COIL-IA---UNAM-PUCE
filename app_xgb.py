from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Cargar modelo XGBoost previamente entrenado
modelo = joblib.load("modelo_xgboost_optimizado.pkl")

@app.route('/')
def formulario():
    return render_template("formulario_xgb.html")

@app.route('/predecir', methods=["POST"])
def predecir():
    try:
        # Variables binarias directas (0 o 1)
        campos_binarios = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                           'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                           'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                           'DiffWalk', 'Sex', 'HighMentHlth', 'HighPhysHlth']
        entradas = [int(request.form[campo]) for campo in campos_binarios]

        # Variables numéricas directas
        BMI = float(request.form['BMI'])
        MentHlth = float(request.form['MentHlth'])
        PhysHlth = float(request.form['PhysHlth'])
        entradas.extend([BMI, MentHlth, PhysHlth])

        # Dummies: GenHlth
        genhlth = int(request.form['GenHlth'])
        for i in [2, 3, 4, 5]:
            entradas.append(1 if genhlth == i else 0)

        # Dummies: Age
        age = int(request.form['Age'])
        for i in range(2, 14):
            entradas.append(1 if age == i else 0)

        # Dummies: Education
        edu = int(request.form['Education'])
        for i in range(2, 7):
            entradas.append(1 if edu == i else 0)

        # Dummies: Income
        income = int(request.form['Income'])
        for i in range(2, 9):
            entradas.append(1 if income == i else 0)

        # Preparar input
        arreglo = np.array(entradas).reshape(1, -1)
        prediccion = modelo.predict(arreglo)[0]

        resultado = "Positivo para Diabetes" if prediccion == 1 else "Negativo para Diabetes"
        return render_template("formulario_xgb.html", resultado=resultado)

    except Exception as e:
        return f"Error al procesar la predicción: {e}"

if __name__ == '__main__':
    app.run(debug=True)
