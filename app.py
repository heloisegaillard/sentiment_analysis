from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load your model
model = joblib.load('model.joblib')

# Initialize the FastAPI app

app = FastAPI()

# Define the input data model using Pydantic
class InputData(BaseModel):
    features: list

# Define a route for predictions
@app.post("/predict/")
async def predict(input_data: InputData):
    # Extract features from the request
    features = input_data.features

    if not features:
        raise HTTPException(status_code=400, detail="No features provided")

    try:
        # Convert features to the format the model expects (e.g., a 2D list)
        prediction = model.predict([features])

        # Return the prediction as JSON
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the FastAPI app, use the command below
# uvicorn main:app --reload
