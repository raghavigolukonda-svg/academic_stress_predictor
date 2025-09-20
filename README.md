# 📘 Academic Stress Predictor

An interactive web application that predicts a student's **academic stress level** based on various factors such as peer pressure, academic pressure from home, study environment, coping strategies, and bad habits. The app uses **machine learning models** (Logistic Regression, Random Forest, and SVM) to make predictions.

---

## 🚀 Features

- Predict academic stress level (1–5) for students.
- Interactive and **robot-themed frontend** with animated stars.
- Real-time prediction display.
- Handles imbalanced data using **SMOTE**.
- Scalable for more input features in the future.

---

## 📊 Dataset

The dataset contains survey responses from students:

| Column | Description |
|--------|-------------|
| `Timestamp` | Date & time of survey response |
| `Your Academic Stage` | Undergraduate, Postgraduate, PhD |
| `Peer pressure` | Scale 1–5 |
| `Academic pressure from your home` | Scale 1–5 |
| `Study Environment` | Peaceful / Noisy |
| `Coping Strategy` | Different strategies students use |
| `Bad Habits` | Yes / No (smoking, drinking) |
| `Academic Competition` | Scale 1–5 |
| `Rate your academic stress index` | Target variable, scale 1–5 |

> Missing values are handled and categorical features are **encoded** for the model.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Flask** (backend web framework)
- **Pandas & NumPy** (data processing)
- **Scikit-learn** (ML model building)
- **Imbalanced-learn (SMOTE)** (handle imbalanced classes)
- **Joblib** (save/load ML models)
- **HTML, CSS** (frontend with interactive design)

---

## 💻 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/academic-stress-predictor.git
cd academic-stress-predictor
2️⃣ Install dependencies

It is recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt


requirements.txt should include:

Flask
pandas
numpy
scikit-learn
imbalanced-learn
joblib

3️⃣ Prepare Dataset

Place your dataset academic_stress.csv in the project root.

Ensure the column names match exactly:

'Your Academic Stage', 'Peer pressure', 'Academic pressure from your home',
'Study Environment', 'What coping strategy you use as a student?',
'Do you have any bad habits like smoking, drinking on a daily basis?',
'What would you rate the academic competition in your student life',
'Rate your academic stress index'

4️⃣ Train Model (Optional)

If you want to retrain the model:

python train_model.py


This will generate:

academic_stress_model.pkl → trained ML model

scaler.pkl → standard scaler for inputs

feature_columns.pkl → columns for one-hot encoding

The model uses SMOTE to handle class imbalance and standard scaling for numeric features.

5️⃣ Run the Flask App
python app.py


Open browser: http://127.0.0.1:5000

Fill in the form and click Predict Stress Level.

See the predicted stress level displayed dynamically.

🎨 Frontend Design

Animated stars in background.

Robot-themed container with glowing effects.

Inputs highlight when focused.

Prediction result glows and appears below the form.

All frontend files are in templates/index.html.

🧠 How it Works

User inputs data in the form.

Flask backend receives POST request.

Data is preprocessed:

Encode categorical features (One-Hot / Label Encoding)

Scale numeric features

Data is passed to the trained ML model.

Predicted academic stress level (1–5) is returned and displayed.

📷 Screenshots


Input form with robot-style design


Prediction result displayed dynamically

⚠️ Notes

Ensure dataset and model files are present in project root.

If using new data, make sure to match feature columns for one-hot encoding.

Numeric inputs (Peer Pressure, Academic Pressure, Competition) must be between 1–5.
