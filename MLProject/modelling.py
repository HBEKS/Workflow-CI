import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Path dataset disesuaikan secara dinamis agar kebal terhadap perbedaan CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, 'loan_preprocessing', 'loan_processed.csv'),
        os.path.join(script_dir, 'loan_processed.csv'),
        '../loan_preprocessing/loan_processed.csv',
        'loan_preprocessing/loan_processed.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Data loaded from: {path}")
            break
    
    if df is None:
        raise FileNotFoundError("❌ Could not find loan_processed.csv")

    # Pisahkan fitur dan target
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Enable autolog
        mlflow.sklearn.autolog()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print("="*50)
        print("✅ MODEL BERHASIL DILATIH")
        print("="*50)
        print(f"Run ID     : {run_id}")
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1-Score   : {f1:.4f}")
        print("="*50)