from flask import Flask, render_template, request
import pickle
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load the trained model
# model = joblib.load(open('model5.pkl', 'rb'))
# Load model with error handling

# Load model with better error handling and type checking
def load_model():
    try:
        model_path = 'model5.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if hasattr(model, 'predict'):
                print("✅ Model loaded successfully. Type:", type(model))
                return model
            else:
                print("❌ Loaded object is not a valid model.")
                return None
        else:
            print("❌ Error: model5.pkl file not found!")
            return None
    except Exception as e:
        print(f"❌ Exception while loading model: {e}")
        return None

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def recommend_heart():
    try:
        if model is None:
            return render_template('index.html', error="Model not loaded properly")
        
        # Get input data from the form
        age = float(request.form.get('age'))
        anaemia = float(request.form.get('anaemia'))
        creatinine_phosphokinase = float(request.form.get('creatinine_phosphokinase'))
        diabetes = float(request.form.get('diabetes'))
        ejection_fraction = float(request.form.get('ejection_fraction'))  # FIXED: was 'anaemia'
        high_blood_pressure = float(request.form.get('high_blood_pressure'))
        platelets = float(request.form.get('platelets'))
        serum_creatinine = float(request.form.get('serum_creatinine'))
        serum_sodium = float(request.form.get('serum_sodium'))
        sex = float(request.form.get('sex'))
        smoking = float(request.form.get('smoking'))
        time = float(request.form.get('time'))

        # Arrange input into numpy array
        input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                                ejection_fraction, high_blood_pressure, platelets,
                                serum_creatinine, serum_sodium, sex, smoking, time]])

        # Predict
        prediction = model.predict(input_data)[0]

        # Interpret result
        result_text = "High Risk of Heart Failure 😟" if prediction == 1 else "Low Risk of Heart Failure 🙂"

        return render_template('index.html', result=result_text)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

# Correct main check
if __name__ == "__main__":
    app.run(debug=True)
