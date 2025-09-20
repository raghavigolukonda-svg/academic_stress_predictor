from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import os

app = Flask(__name__)

MODEL_FILE = "academic_stress_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = r"academic Stress level - maintainance 1.csv"  # Update path if CSV is elsewhere

# Function to train model if not exists
def train_model():
    # Load dataset
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    
    target_col = "Rate your academic stress index"
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Timestamp']
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop([target_col, "Timestamp"], axis=1, errors='ignore')
    y = df_encoded[target_col]
    
    # Train-test split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    class_counts = y_train.value_counts()
    min_class_count = class_counts.min()
    k_neighbors = min(5, min_class_count - 1)
    if k_neighbors < 1:
        k_neighbors = 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    
    # Train model
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # Save model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Model and scaler trained and saved.")
    
    return model, scaler, X.columns.tolist()


# Load model and scaler, or train if missing
if os.path.isfile(MODEL_FILE) and os.path.isfile(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    df_temp = pd.read_csv(DATA_FILE)
    df_temp.columns = df_temp.columns.str.strip()
    categorical_cols = df_temp.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Timestamp']
    df_temp_encoded = pd.get_dummies(df_temp, columns=categorical_cols, drop_first=True)
    feature_columns = df_temp_encoded.drop(["Rate your academic stress index", "Timestamp"], axis=1, errors='ignore').columns.tolist()
    print("Model and scaler loaded successfully.")
else:
    model, scaler, feature_columns = train_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect inputs from form
    user_input = {
        "Peer pressure": int(request.form['peer_pressure']),
        "Academic pressure from your home": int(request.form['home_pressure']),
        "What would you rate the academic  competition in your student life": int(request.form['competition']),
        "Your Academic Stage": request.form['academic_stage'],
        "Study Environment": request.form['study_env'],
        "What coping strategy you use as a student?": request.form['coping'],
        "Do you have any bad habits like smoking, drinking on a daily basis?": request.form['bad_habits']
    }

    user_df = pd.DataFrame([user_input])

    # One-hot encode
    user_df_encoded = pd.get_dummies(user_df)

    # Add missing columns from training
    for col in feature_columns:
        if col not in user_df_encoded.columns:
            user_df_encoded[col] = 0

    # Ensure column order matches training
    user_df_encoded = user_df_encoded[feature_columns]

    # Scale
    user_scaled = scaler.transform(user_df_encoded)

    # Predict
    prediction = model.predict(user_scaled)[0]

    return render_template('index.html', prediction_text=f"Predicted Academic Stress Index: {prediction}")


if __name__ == "__main__":
    app.run(debug=True)
