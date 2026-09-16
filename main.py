from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An end-to-end ML API to predict customer churn.",
    version="1.0"
)

# 2. Load the model and the expected column structure into memory
# We do this outside the endpoint so it only loads once when the server starts
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 3. Define the data schema
# For this portfolio piece, we will accept a dictionary of raw customer features
class CustomerData(BaseModel):
    features: dict

@app.post("/predict")
def predict_churn(payload: CustomerData):
    # Convert the incoming JSON dictionary into a Pandas DataFrame (1 row)
    input_df = pd.DataFrame([payload.features])
    
    # Preprocess: One-Hot Encode any text columns just like we did in Phase 2
    categorical_cols = input_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        input_df = pd.get_dummies(input_df, columns=categorical_cols)
        
    # Crucial Step: Align the new data's columns with the model's expected columns
    # This fills any missing columns with 0s and drops any unexpected columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Make the prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": round(float(probability), 4)
    }