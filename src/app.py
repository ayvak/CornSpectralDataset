from fastapi import FastAPI
import uvicorn
import numpy as np
from pydantic import BaseModel
from data_preprocessing import load_and_preprocess_data
from model_training import build_regression_model

app = FastAPI()

# Load model, scaler, or re-initialize if needed
# In practice, you'd load a saved model (model.save('model.h5'))
X, y, scaler = load_and_preprocess_data("./data/MLE-Assignment.csv")
model = build_regression_model(input_dim=X.shape[1])
model.fit(X, y, epochs=5, verbose=0)  # For demonstration only

class PredictRequest(BaseModel):
    reflectances: list  # List of spectral values

@app.post("/predict")
def predict_don(data: PredictRequest):
    arr = np.array(data.reflectances).reshape(1, -1)
    arr_scaled = scaler.transform(arr)  # scale
    prediction = model.predict(arr_scaled)[0, 0]
    return {"predicted_DON": float(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
