from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from EduManage React app

# Load saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_grade(marks):
    if marks >= 85:
        return 'High', 'Excellent Performance!', 'high'
    elif marks >= 55:
        return 'Medium', 'Average Performance', 'medium'
    else:
        return 'Low', 'Needs Improvement', 'low'

# Original web UI route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours     = float(request.form['study_hours'])
        attendance      = float(request.form['attendance'])
        previous_marks  = float(request.form['previous_marks'])
        sleep_hours     = float(request.form['sleep_hours'])
        extracurricular = int(request.form['extracurricular'])

        features = np.array([[study_hours, attendance, previous_marks, sleep_hours, extracurricular]])
        features_scaled = scaler.transform(features)
        predicted_marks = model.predict(features_scaled)[0]
        predicted_marks = round(float(predicted_marks), 1)
        predicted_marks = max(0, min(100, predicted_marks))

        grade, message, grade_class = get_grade(predicted_marks)

        return render_template('result.html',
                               marks=predicted_marks, grade=grade,
                               message=message, grade_class=grade_class,
                               study_hours=study_hours, attendance=attendance,
                               previous_marks=previous_marks, sleep_hours=sleep_hours,
                               extracurricular='Yes' if extracurricular else 'No')
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


# ── NEW JSON API for EduManage ─────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Single student prediction — called from EduManage"""
    try:
        data = request.get_json()
        study_hours     = float(data['study_hours'])
        attendance      = float(data['attendance'])
        previous_marks  = float(data['previous_marks'])
        sleep_hours     = float(data.get('sleep_hours', 7))
        extracurricular = int(data.get('extracurricular', 0))

        features = np.array([[study_hours, attendance, previous_marks, sleep_hours, extracurricular]])
        features_scaled = scaler.transform(features)
        predicted_marks = model.predict(features_scaled)[0]
        predicted_marks = round(float(predicted_marks), 1)
        predicted_marks = max(0, min(100, predicted_marks))

        grade, message, _ = get_grade(predicted_marks)

        return jsonify({
            'success': True,
            'predicted_marks': predicted_marks,
            'grade': grade,
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict-bulk', methods=['POST'])
def api_predict_bulk():
    """Bulk prediction for multiple students — called from EduManage"""
    try:
        data = request.get_json()
        students = data['students']  # array of student objects
        results = []

        for s in students:
            study_hours     = float(s.get('study_hours', 5))
            attendance      = float(s.get('attendance', 75))
            previous_marks  = float(s.get('previous_marks', 60))
            sleep_hours     = float(s.get('sleep_hours', 7))
            extracurricular = int(s.get('extracurricular', 0))

            features = np.array([[study_hours, attendance, previous_marks, sleep_hours, extracurricular]])
            features_scaled = scaler.transform(features)
            predicted_marks = model.predict(features_scaled)[0]
            predicted_marks = round(float(predicted_marks), 1)
            predicted_marks = max(0, min(100, predicted_marks))

            grade, message, _ = get_grade(predicted_marks)

            results.append({
                'student_id': s.get('id'),
                'student_name': s.get('name'),
                'predicted_marks': predicted_marks,
                'grade': grade,
                'message': message,
                'at_risk': grade == 'Low'
            })

        return jsonify({'success': True, 'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Random Forest', 'version': '1.0'})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # ✅ indented
    app.run(debug=False, host='0.0.0.0', port=port)