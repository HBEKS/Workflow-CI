FROM python:3.12.7-slim

WORKDIR /app

# Install dependencies langsung (tanpa requirements.txt)
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy protobuf==3.20.3

# Copy semua file project
COPY . .

# Expose port
EXPOSE 5000

# Run MLflow model server
ENTRYPOINT ["mlflow", "models", "serve", "-m", "ci_model", "-h", "0.0.0.0", "-p", "5000"]