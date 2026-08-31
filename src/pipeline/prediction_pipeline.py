import os
import sys
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import PredictionPipelineConfig
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
            # import the preprocessor 
            # preprocessor=load_object(self.prediction_pipeline_config.preprocessor_path)
            # # import the model
            # model=load_object(self.prediction_pipeline_config.model_path)
            prcessed=self.preprocessor.transform(df)
            output=self.model.predict(prcessed)
            return output.tolist()
        except Exception as e :
            raise MyException(e,sys)


class CustomData:
    def __init__(self,Gender:str,Age:int,Driving_License:int,Region_Code:float,Previously_Insured:int,Vehicle_Age:str,Vehicle_Damage:str,Annual_Premium:float,Policy_Sales_Channel:float,Vintage:int):
        self.gender=Gender
        self.age=Age
        self.driving_license=Driving_License
        self.region_code=Region_Code
        self.previously_insured=Previously_Insured
        self.vehicle_age=Vehicle_Age
        self.vehicle_damage=Vehicle_Damage
        self.annual_premium=Annual_Premium
        self.policy_sales_channel=Policy_Sales_Channel
        self.vintage=Vintage

    def get_Data_as_Dataframe(self)->pd.DataFrame:
        try:
            input_df=pd.DataFrame({
                "Gender":[self.gender],
                "Age":[self.age],
                "Driving_License":[self.driving_license],
                "Region_Code":[self.region_code],
                "Previously_Insured":[self.previously_insured],
                "Vehicle_Age":[self.vehicle_age],
                "Vehicle_Damage":[self.vehicle_damage],
                "Annual_Premium":[self.annual_premium],
                "Policy_Sales_Channel":[self.policy_sales_channel],
                "Vintage":[self.vintage],
            })
            return input_df
        except Exception as e:
            raise MyException(e, sys)