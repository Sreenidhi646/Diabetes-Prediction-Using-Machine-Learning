from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("Diabetes.pkl") 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        gender = 1 if gender.lower() == 'male' else 0

        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])

        smoking_history = request.form['smoking_history']
        smoking_dict = {'never': 0, 'former': 1, 'current': 2, 'ever': 3, 'not current': 4, 'No Info': 5}
        smoking_history = smoking_dict.get(smoking_history.lower(), 5)

        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        glucose = float(request.form['glucose'])

        features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]])
        prediction = model.predict(features)[0]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)