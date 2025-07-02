from flask import Flask, request, jsonify
import pandas as pd
import joblib

# تحميل النماذج والمحولات
diagnosis_model = joblib.load('sleep_apnea_finalmodel.pkl')
diagnosis_label_encoder = joblib.load('label_encoder_sleep_apnea_final.pkl')

treatment_model = joblib.load("treatment_model_with_diagnosis.pkl")
treatment_label_encoders = joblib.load("treatment_label_encoders.pkl")

app = Flask(__name__)

# دالة تجهيز البيانات الموحدة
def preprocess_input_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
    if "Snoring" in df.columns:
        df["Snoring"] = df["Snoring"].astype(str).str.strip().str.capitalize()
    if "EEG_Sleep_Stage" in df.columns:
        df["EEG_Sleep_Stage"] = df["EEG_Sleep_Stage"].astype(str).str.strip().str.upper()
    if "Diagnosis_of_SDB" in df.columns:
        df["Diagnosis_of_SDB"] = df["Diagnosis_of_SDB"].astype(str).str.strip().str.capitalize()

    return df

@app.route('/')
def home():
    return "✅ Sleep Apnea Diagnosis & Treatment API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = preprocess_input_data(data)

        prediction = diagnosis_model.predict(df)
        diagnosis = diagnosis_label_encoder.inverse_transform(prediction)[0]

        return jsonify({'prediction': diagnosis})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/treatment', methods=['POST'])
def treatment_predict():
    try:
        data = request.get_json()
        df = preprocess_input_data(data)

        if "Diagnosis_of_SDB" not in df.columns or pd.isnull(df["Diagnosis_of_SDB"][0]):
            diagnosis_pred = diagnosis_model.predict(df)[0]
            diagnosis_label = diagnosis_label_encoder.inverse_transform([diagnosis_pred])[0]
            df["Diagnosis_of_SDB"] = diagnosis_label
            df = preprocess_input_data(df.to_dict(orient="records")[0])

        prediction = treatment_model.predict(df)[0]
        result = {
            col: treatment_label_encoders[col].inverse_transform([prediction[i]])[0]
            for i, col in enumerate(['Treatment_Required', 'CPAP', 'Surgery'])
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        df = preprocess_input_data(data)

        prediction = diagnosis_model.predict(df)
        diagnosis = diagnosis_label_encoder.inverse_transform(prediction)[0]

        warnings = []

        age = data.get("Age", 0)
        bmi = data.get("BMI", 0)
        o2 = data.get("Oxygen_Saturation", 100)
        ahi = data.get("AHI", 0)
        hr = data.get("ECG_Heart_Rate", 0)
        airflow = data.get("Nasal_Airflow", 1.0)
        chest = data.get("Chest_Movement", 1.0)
        snoring = str(data.get("Snoring", "")).strip().lower()
        sleep_stage = str(data.get("EEG_Sleep_Stage", "")).strip().upper()

        if age > 60: warnings.append("⚠️ Age > 60 = higher SDB risk.")
        if bmi >= 35: warnings.append("⚠️ BMI ≥ 35 = severe obesity.")
        elif bmi >= 30: warnings.append("⚠️ BMI ≥ 30 = obesity.")
        elif bmi < 18.5: warnings.append("⚠️ Underweight = check for other disorders.")

        if o2 < 85: warnings.append("⚠️ O2 < 85% = critical hypoxia.")
        elif o2 < 90: warnings.append("⚠️ O2 < 90% = low saturation.")

        if ahi >= 30: warnings.append("⚠️ AHI ≥ 30 = severe sleep apnea.")
        elif ahi >= 15: warnings.append("⚠️ AHI 15–30 = moderate apnea.")
        elif ahi >= 5: warnings.append("⚠️ AHI 5–15 = mild apnea.")
        else: warnings.append("✅ AHI is normal.")

        if hr > 110: warnings.append("⚠️ HR > 110 = possible tachycardia.")
        elif hr > 100: warnings.append("⚠️ HR > 100 = elevated.")
        elif hr < 50: warnings.append("⚠️ HR < 50 = possible bradycardia.")

        if airflow < 0.2: warnings.append("⚠️ Low nasal airflow (< 0.2).")
        elif airflow < 0.3: warnings.append("⚠️ Reduced nasal airflow.")

        if chest < 0.2: warnings.append("⚠️ Weak chest movement.")
        elif chest < 0.3: warnings.append("⚠️ Shallow chest movement.")

        if snoring in ["yes", "true", "y"]:
            warnings.append("⚠️ Snoring reported—possible OSA.")

        if sleep_stage == "REM":
            warnings.append("⚠️ Apneas may worsen in REM sleep.")
        elif sleep_stage == "NREM":
            warnings.append("🌀 Deep sleep (NREM). Some apneas may go undetected.")

        report = {
            "Diagnosis": diagnosis,
            "Summary": {
                "Age": age,
                "Gender": data.get("Gender"),
                "BMI": bmi,
                "Snoring": data.get("Snoring"),
                "Oxygen_Saturation": o2,
                "AHI": ahi,
                "ECG_Heart_Rate": hr,
                "Nasal_Airflow": airflow,
                "Chest_Movement": chest,
                "EEG_Sleep_Stage": data.get("EEG_Sleep_Stage")
            },
            "Symptoms": data.get("Patient_Symptoms", ""),
            "Physician_Notes": data.get("Physician_Notes", ""),
            "Warnings": warnings
        }

        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/full_report', methods=['POST'])
def full_report():
    try:
        data = request.get_json()
        df = preprocess_input_data(data)

        if "Diagnosis_of_SDB" not in df.columns or pd.isnull(df["Diagnosis_of_SDB"][0]):
            diagnosis_pred = diagnosis_model.predict(df)[0]
            diagnosis_label = diagnosis_label_encoder.inverse_transform([diagnosis_pred])[0]
            df["Diagnosis_of_SDB"] = diagnosis_label
            df = preprocess_input_data(df.to_dict(orient="records")[0])
        else:
            diagnosis_label = df["Diagnosis_of_SDB"].iloc[0]

        treatment_pred = treatment_model.predict(df)[0]
        treatment_result = {
            col: treatment_label_encoders[col].inverse_transform([treatment_pred[i]])[0]
            for i, col in enumerate(['Treatment_Required', 'CPAP', 'Surgery'])
        }

        # توليد التحذيرات بناءً على القيم
        age = data.get("Age", 0)
        bmi = data.get("BMI", 0)
        o2 = data.get("Oxygen_Saturation", 100)
        ahi = data.get("AHI", 0)
        hr = data.get("ECG_Heart_Rate", 0)
        airflow = data.get("Nasal_Airflow", 1.0)
        chest = data.get("Chest_Movement", 1.0)

        warnings = []
        if age > 60: warnings.append("⚠️ Age > 60 = elevated risk of SDB.")
        if bmi >= 35: warnings.append("⚠️ BMI ≥ 35 indicates severe obesity.")
        elif bmi >= 30: warnings.append("⚠️ BMI ≥ 30 = obesity.")
        elif bmi < 18.5: warnings.append("⚠️ Underweight could signal metabolic issues.")

        if o2 < 85: warnings.append("⚠️ Oxygen saturation critically low (< 85%).")
        elif o2 < 90: warnings.append("⚠️ Oxygen saturation below normal (< 90%).")

        if ahi >= 30: warnings.append("⚠️ AHI ≥ 30 = severe sleep apnea.")
        elif ahi >= 15: warnings.append("⚠️ AHI between 15–30 = moderate apnea.")
        elif ahi >= 5: warnings.append("⚠️ AHI between 5–15 = mild apnea.")

        if hr > 110: warnings.append("⚠️ High heart rate (> 110 bpm) — possible tachycardia.")
        elif hr < 50: warnings.append("⚠️ Low heart rate (< 50 bpm) — possible bradycardia.")

        if airflow < 0.3: warnings.append("⚠️ Reduced nasal airflow detected.")
        if chest < 0.3: warnings.append("⚠️ Shallow chest movement — monitor for central apnea.")

        report = {
            "Diagnosis": diagnosis_label,
            "Treatment_Plan": treatment_result,
            "Summary": {
                "Age": age,
                "Gender": data.get("Gender"),
                "BMI": bmi,
                "Snoring": data.get("Snoring"),
                "Oxygen_Saturation": o2,
                "AHI": ahi,
                "ECG_Heart_Rate": hr,
                "Nasal_Airflow": airflow,
                "Chest_Movement": chest,
                "EEG_Sleep_Stage": data.get("EEG_Sleep_Stage")
            },
            "Symptoms": data.get("Patient_Symptoms", ""),
            "Physician_Notes": data.get("Physician_Notes", ""),
            "Warnings": warnings
        }

        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host='0.0.0.0', port=port)
