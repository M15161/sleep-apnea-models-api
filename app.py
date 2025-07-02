from flask import Flask, request, jsonify
import pandas as pd
import joblib


diagnosis_model = joblib.load('sleep_apnea_finalmodel.pkl')
diagnosis_label_encoder = joblib.load('label_encoder_sleep_apnea_final.pkl')

treatment_model = joblib.load("treatment_model_with_diagnosis.pkl")
treatment_label_encoders = joblib.load("treatment_label_encoders.pkl")


app = Flask(__name__)

@app.route('/')
def home():
    return "Sleep Apnea Diagnosis & Treatment API is running."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        df["Gender"] = df["Gender"].str.strip().str.capitalize()
        df["Snoring"] = df["Snoring"].str.strip()
        df["EEG_Sleep_Stage"] = df["EEG_Sleep_Stage"].str.strip().str.upper()

        prediction = diagnosis_model.predict(df)
        diagnosis = diagnosis_label_encoder.inverse_transform(prediction)[0]

        return jsonify({'prediction': diagnosis})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/treatment', methods=['POST'])
def treatment_predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        df["Gender"] = df["Gender"].str.strip().str.capitalize()
        df["Snoring"] = df["Snoring"].str.strip()
        df["EEG_Sleep_Stage"] = df["EEG_Sleep_Stage"].str.strip().str.upper()

        if "Diagnosis_of_SDB" not in df.columns or pd.isnull(df["Diagnosis_of_SDB"][0]):
            diagnosis_pred = diagnosis_model.predict(df)[0]
            diagnosis_label = diagnosis_label_encoder.inverse_transform([diagnosis_pred])[0]
            df["Diagnosis_of_SDB"] = diagnosis_label

        prediction = treatment_model.predict(df)[0]
        result = {}
        for i, col in enumerate(['Treatment_Required', 'CPAP', 'Surgery']):
            result[col] = treatment_label_encoders[col].inverse_transform([prediction[i]])[0]

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        df["Gender"] = df["Gender"].str.strip().str.capitalize()
        df["Snoring"] = df["Snoring"].str.strip()
        df["EEG_Sleep_Stage"] = df["EEG_Sleep_Stage"].str.strip().str.upper()

        
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

        #  التحذيرات
        if age > 60:
            warnings.append("⚠️ Patient is over 60. Risk of SDB increases significantly with age.")
        if bmi >= 35:
            warnings.append("⚠️ BMI indicates severe obesity (≥ 35). This is a major risk factor.")
        elif bmi >= 30:
            warnings.append("⚠️ BMI indicates obesity (≥ 30). Consider lifestyle intervention.")
        elif bmi < 18.5:
            warnings.append("⚠️ Underweight may also be associated with certain sleep disorders.")

        if o2 < 85:
            warnings.append("⚠️ Critically low oxygen saturation (< 85%). May indicate serious hypoxia.")
        elif o2 < 90:
            warnings.append("⚠️ Low oxygen saturation (< 90%). Consider further investigation.")

        if ahi >= 30:
            warnings.append("⚠️ AHI ≥ 30 indicates **severe** sleep apnea. Immediate treatment advised.")
        elif ahi >= 15:
            warnings.append("⚠️ AHI between 15–30 indicates **moderate** sleep apnea.")
        elif ahi >= 5:
            warnings.append("⚠️ AHI between 5–15 indicates **mild** sleep apnea.")
        else:
            warnings.append("✅ AHI within normal range.")

        if hr > 110:
            warnings.append("⚠️ Severe tachycardia detected (HR > 110). Cardiac evaluation recommended.")
        elif hr > 100:
            warnings.append("⚠️ Elevated heart rate (> 100 bpm). Possible sign of arousal or stress.")
        elif hr < 50:
            warnings.append("⚠️ Bradycardia detected (< 50 bpm). Monitor for arrhythmias.")

        if airflow < 0.2:
            warnings.append("⚠️ Critically low nasal airflow. May indicate nasal obstruction.")
        elif airflow < 0.3:
            warnings.append("⚠️ Reduced nasal airflow detected.")

        if chest < 0.2:
            warnings.append("⚠️ Minimal chest movement detected. Risk of central apnea.")
        elif chest < 0.3:
            warnings.append("⚠️ Shallow chest movement. Check for respiratory effort.")

        if snoring in ["yes", "true", "y"]:
            warnings.append("⚠️ Patient reports snoring. Common symptom of obstructive sleep apnea.")

        if sleep_stage == "REM":
            warnings.append("⚠️ Apnea events often worsen during REM sleep. Increased monitoring advised.")

        if sleep_stage == "NREM":
            warnings.append(" Deep sleep stage detected (NREM). Apnea may be underestimated in this stage.")

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
        return jsonify({"error": str(e)})


@app.route('/full_report', methods=['POST'])
def full_report():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        df["Gender"] = df["Gender"].str.strip().str.capitalize()
        df["Snoring"] = df["Snoring"].str.strip()
        df["EEG_Sleep_Stage"] = df["EEG_Sleep_Stage"].str.strip().str.upper()

        # التشخيص
        diagnosis_pred = diagnosis_model.predict(df)[0]
        diagnosis_label = diagnosis_label_encoder.inverse_transform([diagnosis_pred])[0]
        df["Diagnosis_of_SDB"] = diagnosis_label

        # العلاج
        treatment_pred = treatment_model.predict(df)[0]
        treatment_result = {}
        for i, col in enumerate(['Treatment_Required', 'CPAP', 'Surgery']):
            treatment_result[col] = treatment_label_encoders[col].inverse_transform([treatment_pred[i]])[0]

        # تحذيرات
        age = data.get("Age", 0)
        bmi = data.get("BMI", 0)
        o2 = data.get("Oxygen_Saturation", 100)
        ahi = data.get("AHI", 0)
        hr = data.get("ECG_Heart_Rate", 0)
        airflow = data.get("Nasal_Airflow", 1.0)
        chest = data.get("Chest_Movement", 1.0)

        warnings = []
        if age > 60:
            warnings.append("Patient age > 60. Risk increases with age.")
        if bmi >= 30:
            warnings.append("BMI >= 30 indicates obesity, a known risk factor.")
        if o2 < 90:
            warnings.append("Oxygen Saturation is below normal (< 90%)")
        if ahi >= 30:
            warnings.append("Severe sleep apnea detected. Immediate attention required.")
        elif ahi >= 15:
            warnings.append("Moderate sleep apnea detected. Consider further evaluation.")
        elif ahi >= 5:
            warnings.append("Mild sleep apnea detected.")
        else:
            warnings.append("AHI within normal range.")
        if hr > 100:
            warnings.append("Elevated heart rate detected.")
        if airflow < 0.3:
            warnings.append("Reduced nasal airflow.")
        if chest < 0.3:
            warnings.append("Shallow chest movement detected.")

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
        return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
