# Student Performance Prediction — ML Model

A Machine Learning project that predicts student academic performance using a **Random Forest Regression model**. Built with Python, scikit-learn, and Flask. Integrated with [EduManage SMS](https://github.com/Neha80915/edumanage-sms) as the AI prediction backend.

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Cross-Validation R² | 0.9944 |
| Test Set R² | 0.999 |
| RMSE | 0.64 |
| Training Data | student_data.csv |

## 📊 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| previous_marks | Previous exam score | 0 - 100 |
| attendance | Attendance percentage | 0 - 100 |
| study_hours | Daily study hours | 0 - 12 |
| sleep_hours | Daily sleep hours | 3 - 12 |
| extracurricular | Activity participation | 0 = No, 1 = Yes |

## 🎯 Output

- **predicted_marks** — Final score (0–100)
- **grade** — High (≥75) / Medium (55–74) / Low (<55)
- **at_risk** — True if grade is Low

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML Model | scikit-learn RandomForestRegressor |
| API Server | Flask 3.x |
| CORS | Flask-CORS |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Serialization | joblib |

## 📦 Installation

```bash
git clone https://github.com/nidishsharma/student_performance.git
cd student_performance

pip install -r requirements.txt
```

## 🚀 Usage

### Train the model
```bash
python model.py
```
This generates `model.pkl`, `scaler.pkl`, and all insight graphs in `static/`.

### Start the Flask API server
```bash
python app.py
```
Runs on **http://localhost:5000**

### Original web UI
Open browser at `http://localhost:5000` to use the standalone prediction form.

## 🔌 API Endpoints

### Single Prediction
```
POST /api/predict
Content-Type: application/json

{
  "study_hours": 5,
  "attendance": 80,
  "previous_marks": 65,
  "sleep_hours": 7,
  "extracurricular": 0
}
```
Response:
```json
{
  "success": true,
  "predicted_marks": 72.5,
  "grade": "Medium",
  "message": "Average Performance"
}
```

### Bulk Prediction
```
POST /api/predict-bulk
Content-Type: application/json

{
  "students": [
    { "id": "1", "name": "Rahul", "study_hours": 5, "attendance": 80, "previous_marks": 65, "sleep_hours": 7, "extracurricular": 0 }
  ]
}
```

### Health Check
```
GET /api/health
```

## 📁 Project Structure

```
student_performance/
├── app.py              — Flask API server
├── model.py            — Model training & graph generation
├── preprocess.py       — Data preprocessing pipeline
├── evaluate.py         — Model evaluation metrics
├── model.pkl           — Trained Random Forest model
├── scaler.pkl          — Fitted StandardScaler
├── student_data.csv    — Training dataset
├── requirements.txt    — Python dependencies
├── static/             — Generated insight graphs (PNG)
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   ├── cross_validation.png
│   ├── grade_distribution.png
│   ├── heatmap.png
│   ├── residuals.png
│   └── scatter.png
└── templates/          — Original web UI (HTML)
    ├── index.html
    └── result.html
```

## 🔗 Integration

This project is used as the ML backend for **EduManage SMS**:
- EduManage Repository: [github.com/Neha80915/edumanage-sms](https://github.com/Neha80915/edumanage-sms)
- Both servers must run simultaneously for full functionality
- EduManage runs on `localhost:5173`, this API runs on `localhost:5000`

## 🎓 About

**Project:** Minor Project — B.Tech Computer Science (AI/ML), Semester 4
**Developer:** nidishsharma
**Institution:** Rayat Bhara University, Mohali
**Collaboration:** Integrated into EduManage by [Neha80915](https://github.com/Neha80915)