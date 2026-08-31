import os
import sys
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import PredictionPipelineConfig
from src.entity.estimator import MyModel
from src.utils.main_utils import load_object

class PredictionPipeline:
    def __init__(self,prediction_pipeline_config:PredictionPipelineConfig=None):
        try:
            self.prediction_pipeline_config=prediction_pipeline_config or PredictionPipelineConfig()
            self.model=load_object(self.prediction_pipeline_config.model_path)
            self.preprocessor=load_object(self.prediction_pipeline_config.preprocessor_path)
        except Exception as e:
            raise MyException(e, sys)

    def predict(self,df:pd.DataFrame):
        try:
            # import through MyModel estimator 
            model=MyModel(self.preprocessor,self.model)
            transformed_input=model.transform(df)
            output=model.predict(transformed_input)
            return output.tolist()
        except Exception as e :
            raise MyException(e,sys)
