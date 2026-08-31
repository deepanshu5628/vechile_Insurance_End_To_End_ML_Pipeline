import sys
from src.exception import MyException
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd
class MyModel:
    def __init__(self,pipeline_object:Pipeline,trained_model_object:object):
        try:
            self.pipeline_object=pipeline_object
            self.trained_model_object=trained_model_object
        except Exception as e:
            raise MyException(e,sys)

    def transform(self,df:pd.DataFrame):
        try:
            transformed_data=self.pipeline_object.transform(df)
            return transformed_data
        except Exception as e:
            raise MyException(e,sys)
        
    def predict(self,npy_arr:np.array):
        try:
            output=self.trained_model_object.predict(npy_arr)
            return output
        except Exception as e:
            raise MyException(e,sys)
