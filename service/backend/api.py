
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
# if you use a PipelineModel to load the machine learning model 
# it will throws an error because it's not a pipelinemodel the stored one but an rf model
# I use the PipelineModel to load the object to do the preprocessing
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
# don't need to use mlflow because I saved the model directly using pyspark from the notebook in the backend directory
# take a look at the notebook in to see the operation done
#import mlflow.spark

# Creating the Spark session
spark = SparkSession.builder \
    .appName("SparkBackendApp") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

# importing the model
rf_model_loaded = RandomForestClassificationModel.load("/home/src/model/rf_model")
preproc_pipeline_loaded = PipelineModel.load("/home/src/model/preproc_pipeline")
#rf_model_loaded = mlflow.spark.load_model("/home/src/model/saved_model")

# the same used in the preprocessing object except for education_grouped
# because the pipeline saved need the correct column name
feature_columns = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

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

@api.post('/predict')
def predict(debtor: Debtor):    
    
    # take the input data from the request using the Debtor class parameters
    input_data = [debtor.Limit_bal, debtor.Sex, debtor.Education, debtor.Marriage, debtor.Age, debtor.Pay_1, \
        debtor.Pay_2, debtor.Pay_3, debtor.Pay_4, debtor.Pay_5, debtor.Pay_6, debtor.Bill_amt1, debtor.Bill_amt2, debtor.Bill_amt3, \
        debtor.Bill_amt4, debtor.Bill_amt5, debtor.Bill_amt6, debtor.Pay_amt1, debtor.Pay_amt2, debtor.Pay_amt3, debtor.Pay_amt4,
        debtor.Pay_amt5, debtor.Pay_amt6]
    
    # casting to tuple avoid the error in dataframe creation
    # because of float type schema inference thrown if i use the original input_data
    input_data = [tuple(input_data)] 
    input_data = spark.createDataFrame(input_data, feature_columns)
    
    # Create "EDUCATION_grouped" column
    input_data = input_data.withColumn(
        "EDUCATION_grouped",
        when((col("EDUCATION") == 4) | (col("EDUCATION") == 5) | (col("EDUCATION") == 6), 4)
        .otherwise(col("EDUCATION"))
    )

    
    # applying preprocessing step
    preprocessed_data = preproc_pipeline_loaded.transform(input_data)
    
    # if you do the prediction like this pyspark throws an exception
    # because the model need a Vector object to do the inference and
    # can't execute the prediction on a Dataset object
    # rf_prediction = rf_model_loaded.predict(preprocessed_data) 
    
    # prediction step
    features_col_name = rf_model_loaded.getFeaturesCol() # take the features col from the model to generalize the code
    dense_vector_input = preprocessed_data.head()[features_col_name] # get the dense vector from the preprocessed data
    rf_prediction = rf_model_loaded.predict(dense_vector_input) # executing the prediction method

    return {'rf_prediction': rf_prediction} # return a single value
    

spark.sparkContext.setLogLevel("WARN")

#Run the API with uvicorn

if __name__ == '__main__':
    # need to use 0.0.0.0 in docker as localhost to avoid error during startup
    uvicorn.run(api, host='0.0.0.0', port=8080) 
