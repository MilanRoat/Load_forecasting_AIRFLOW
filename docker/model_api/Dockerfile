FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib pandas dill prophet

WORKDIR /app
COPY src/ /app/src

# Point to Linear Regression meter model
ENV MODEL_PATH=/app/src/models/artifacts/models/linear_regression_model.pkl
# Optional: if you have an encoder
ENV ENCODER_PATH=/app/src/models/artifacts/models/encoder.pkl

# Run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "5501"]
