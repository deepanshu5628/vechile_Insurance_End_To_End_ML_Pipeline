from fastapi import FastAPI,Request,HTTPException   
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses  import Response
from uvicorn import run as app_run

from src.exception import MyException
from src.logger import logging
import sys

from src.pipeline.training_pipeline import  TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.schema.prediction_schema import CustomData
from src.constants import APP_HOST,APP_PORT
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return {"message":"Veichle Insurance prediction app "}

@app.get("/train")
async def train():
    try:    
        logging.info("Training pipeline started")
        pipeline=TrainingPipeline()
        pipeline.run_pipeline()
        logging.info("Training pipeline completed")
        return {"status":"Training completed"}
    except Exception as e:
        raise MyException(e, sys)




@app.post("/predict")
async def predict(data:CustomData):
    try:
        logging.info("Prediction pipeline started")
        pipeline=PredictionPipeline()
        input_df=data.to_dataframe()
        output=pipeline.predict(input_df)
        logging.info("Prediction pipeline completed")
        return {"prediction is ":output}
    except MyException as e:
        raise HTTPException(status_code=500, detail=str(e))  # ← proper HTTP error


    
if __name__ =="__main__":
    app_run(app,host=APP_HOST,port=APP_PORT)

