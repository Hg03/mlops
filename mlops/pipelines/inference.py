from fastapi import FastAPI
from pydantic import BaseModel
import comet_ml
from comet_ml import API
import xgboost as xgb
import numpy as np
from dotenv import load_dotenv
import uvicorn
import os

# Replace with your Comet credentials and model details
COMET_API_KEY = os.getenv("comet_ml_api_key")
WORKSPACE = "harish-workspace"
REGISTRY_NAME = "v1"
VERSION = "1.0.0"  # Optional

# Initialize FastAPI app
app = FastAPI()

# Define the input data schema using Pydantic
class InputFeatures(BaseModel):
    age: int
    gender: int
    annualincome: float
    numberofpurchases: int
    productcategory: int
    timespentonwebsite: float
    loyaltyprogram: int
    discountsavailed: int

# Load the model from Comet ML at startup
@app.on_event("startup")
async def load_model():
    global xgboost_model
    api = API(api_key=COMET_API_KEY)
    model = api.get_model(workspace=WORKSPACE, model_name=REGISTRY_NAME)
    
    # Download the XGBoost model from Comet ML
    model.download(version=VERSION, output_folder="./")
    
    # Load the model using XGBoost
    xgboost_model = xgb.Booster()
    xgboost_model.load_model("./model.pkl")
    print("XGBoost model loaded successfully.")

# Create a POST endpoint for predictions
@app.post("/predict")
async def predict(input_features: InputFeatures):
    # Extract features from the request body
    features = np.array([[input_features.feature1, input_features.feature2, input_features.feature3]]) 
    # Reshape the data based on the number of features (adjust accordingly)

    # Convert the features into a DMatrix (XGBoost's input format)
    dmatrix = xgb.DMatrix(features)

    # Make predictions using the loaded XGBoost model
    predictions = xgboost_model.predict(dmatrix)

    # Return the predictions as a response
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
