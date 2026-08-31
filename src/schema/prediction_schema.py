from pydantic import BaseModel, Field, field_validator
from typing import Literal
import pandas as pd
from src.exception import MyException
from src.logger import logging
import sys

class CustomData(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: int = Field(ge=18, le=100)
    Driving_License: Literal[0, 1]
    Region_Code: float = Field(ge=0, le=52)
    Previously_Insured: Literal[0, 1]
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(gt=0)
    Policy_Sales_Channel: float = Field(ge=0, le=163)
    Vintage: int = Field(ge=0, le=365)

    @field_validator("Gender", "Vehicle_Damage", mode="before")
    @classmethod
    def sanitize_strings(cls, v: str) -> str:
        return v.strip().title()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])
    
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