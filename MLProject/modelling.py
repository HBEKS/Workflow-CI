# modelling.py yang sudah diperbaiki

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================
# SETUP
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("CI_Workflow")

# ============================================
# LOAD DATA
# ============================================
# Pastikan path file CSV ada
csv_path = 'loan_preprocessing/loan_processed.csv'
if not os.path.exists(csv_path):
    print(f"Error: File tidak ditemukan di {csv_path}")
    print("Mencoba path alternatif...")
    csv_path = '../loan_preprocessing/loan_processed.csv'
    
df = pd.read_csv(csv_path)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*60)
print("MLFLOW CI TRAINING WORKFLOW")
print(f"Python: 3.12.7 | MLflow: 2.19.0")
print("="*60)

with mlflow.start_run(run_name="CI_Training"):
    
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)
    
    # ============================================
    # BUAT FILE GAMBAR DULU, BARU LOG
    # ============================================
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - CI Training')
    plt.tight_layout()
    plt.savefig('confusion_matrix_ci.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix_ci.png')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - CI Training')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_ci.png')
    plt.close()
    mlflow.log_artifact('roc_curve_ci.png')
    
    # 3. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance - CI Training')
    plt.tight_layout()
    plt.savefig('feature_importance_ci.png')
    plt.close()
    mlflow.log_artifact('feature_importance_ci.png')
    
    mlflow.sklearn.log_model(model, "ci_model")
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nRun ID: {mlflow.active_run().info.run_id}")
    print("="*50)