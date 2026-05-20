FROM python:3.12.7-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Expose port
EXPOSE 5000

# Run MLflow model server
ENTRYPOINT ["mlflow", "models", "serve", "-m", "ci_model", "-h", "0.0.0.0", "-p", "5000"]