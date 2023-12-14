
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import pickle
import numpy as np
#from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import mlflow.spark

# Creating the Spark session
spark = SparkSession.builder \
    .appName("SparkFrontendApp") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

# importing the model
#rf_model_loaded = PipelineModel.load("/home/src/model/saved_model")
rf_model_loaded = mlflow.spark.load_model("/home/src/model/saved_model")

# creating app
api = FastAPI()

class Debtor(BaseModel):
    Limit_bal: float
    Sex: float
    Education: float
    Marriage: float
    Age: float
    Pay_1: float
    Pay_2: float 
    Pay_3: float
    Pay_4: float
    Pay_5: float
    Pay_6: float
    Bill_amt1: float
    Bill_amt2: float
    Bill_amt3: float
    Bill_amt4: float
    Bill_amt5: float
    Bill_amt6: float
    Pay_amt1: float
    Pay_amt2: float
    Pay_amt3: float 
    Pay_amt4: float 
    Pay_amt5: float
    Pay_amt6: float

@api.post('/predict_spark')
def predict_spark(debtor: Debtor):    
    
    # Reshape the input data to a 2D array
    input_data = [debtor.Limit_bal, debtor.Sex, debtor.Education, debtor.Marriage, debtor.Age, debtor.Pay_1, \
        debtor.Pay_2, debtor.Pay_3, debtor.Pay_4, debtor.Pay_5, debtor.Pay_6, debtor.Bill_amt1, debtor.Bill_amt2, debtor.Bill_amt3,
        debtor.Bill_amt4, debtor.Bill_amt5, debtor.Bill_amt6, debtor.Pay_amt1, debtor.Pay_amt2, debtor.Pay_amt3, debtor.Pay_amt4,
        debtor.Pay_amt5, debtor.Pay_amt6]
    input_data = np.array(input_data)
    
    rf_prediction = rf_model_loaded.predict_spark(input_data.reshape(1, -1))
    
    return {'rf_prediction': rf_prediction.tolist()[0]} # return a single value
    

spark.sparkContext.setLogLevel("WARN")

#Run the API with uvicorn

if __name__ == '__main__':
    # need to use 0.0.0.0 in docker as localhost to avoid error during startup
    uvicorn.run(api, host='0.0.0.0', port=8080) 
