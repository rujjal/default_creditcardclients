""" import uvicorn
from fastapi import FastAPI, Form
import requests
import os
#from flask import render_template

from fastapi import  Request
from fastapi.templating import Jinja2Templates

#Creating the app object
app = FastAPI()
templates = Jinja2Templates(directory="templates")

API_HOST = str(os.getenv("BACKEND_HOST"))
API_PORT = str(os.getenv("BACKEND_PORT"))
    

@app.get('/')
def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/predict')
def get_predict(request: Request):
    return templates.TemplateResponse("prediction_form.html", {"request": request})
    
@app.post('/predict')
def post_predict(request: Request,
                 Limit_bal: float = Form(...),
                 Sex: float = Form(...),
                 Education: float = Form(...),
                 Marriage: float = Form(...),
                 Age: float = Form(...),
                 Pay_1: float = Form(...),
                 Pay_2: float = Form(...),
                 Pay_3: float = Form(...),
                 Pay_4: float = Form(...),
                 Pay_5: float = Form(...),
                 Pay_6: float = Form(...),
                 Bill_amt1: float = Form(...),
                 Bill_amt2: float = Form(...),
                 Bill_amt3: float = Form(...),
                 Bill_amt4: float = Form(...),
                 Bill_amt5: float = Form(...),
                 Bill_amt6: float = Form(...),
                 Pay_amt1: float = Form(...),
                 Pay_amt2: float = Form(...),
                 Pay_amt3: float = Form(...),
                 Pay_amt4: float = Form(...),
                 Pay_amt5: float = Form(...),
                 Pay_amt6: float = Form(...)
                 ):
    
    # json input creation: I follow the standard used by the backend service
    # for the col_values I take only the first element because I implemented a single call prediction 
    col_names = ["Limt_bal", "Sex", "Education", "Marriage", "Age", "Pay_1", " Pay_2", "Pay_3", "Pay_4", "Pay_5","Pay_6","Bill_amt1",
                 "Bill_amt2","Bill_amt3","Bill_amt4","Bill_amt5","Pay_amt1","Pay_amt2","Pay_amt3","Pay_amt4","Pay_amt5","Pay_amt6"]
    col_values = [Limt_bal, Sex, Education, Marriage, Age, Pay_1, Pay_2, Pay_3, Pay_4, Pay_5,Pay_6,Bill_amt1,
                 Bill_amt2,Bill_amt3,Bill_amt4,Bill_amt5,Pay_amt1,Pay_amt2,Pay_amt3,Pay_amt4,Pay_amt5,Pay_amt6]
    json_input = dict(zip(col_names, col_values))
    
    # api call to obtain the result
    api_url = f"http://{API_HOST}:{API_PORT}/predict"
    
    response = requests.post(api_url, json=json_input)
    response = response.json()
    
    return templates.TemplateResponse("prediction_form.html", {"request": request, "rf_prediction": response["rf_prediction"]} )

#Run the API with uvicorn

if __name__ == '__main__':
    # need to use 0.0.0.0 in docker as localhost to avoid error during startup
    uvicorn.run(app, host='0.0.0.0', port=8080) """


import uvicorn
from fastapi import FastAPI, Form
import requests
import os
from pyspark.sql import SparkSession
from fastapi import Request
from fastapi.templating import Jinja2Templates

# Creating the Spark session
spark = SparkSession.builder.appName("SparkFrontendApp").getOrCreate()

# Your existing code for the FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

API_HOST = str(os.getenv("BACKEND_HOST"))
API_PORT = str(os.getenv("BACKEND_PORT"))

# ... (Your existing code for app routes)

# This route is the one that communicates with the Spark-backed FastAPI
@app.post('/predict_spark')
def post_predict_spark(request: Request,
                       Limit_bal: float = Form(...),
                       Sex: float = Form(...),
                       Education: float = Form(...),
                       Marriage: float = Form(...),
                       Age: float = Form(...),
                       Pay_1: float = Form(...),
                       Pay_2: float = Form(...),
                       Pay_3: float = Form(...),
                       Pay_4: float = Form(...),
                       Pay_5: float = Form(...),
                       Pay_6: float = Form(...),
                       Bill_amt1: float = Form(...),
                       Bill_amt2: float = Form(...),
                       Bill_amt3: float = Form(...),
                       Bill_amt4: float = Form(...),
                       Bill_amt5: float = Form(...),
                       Pay_amt1: float = Form(...),
                       Pay_amt2: float = Form(...),
                       Pay_amt3: float = Form(...),
                       Pay_amt4: float = Form(...),
                       Pay_amt5: float = Form(...),
                       Pay_amt6: float = Form(...)
                       ):

    # Your existing code for creating the JSON input
    col_names = ["Limt_bal", "Sex", "Education", "Marriage", "Age", "Pay_1", " Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6",
                 "Bill_amt1", "Bill_amt2", "Bill_amt3", "Bill_amt4", "Bill_amt5", "Pay_amt1", "Pay_amt2", "Pay_amt3",
                 "Pay_amt4", "Pay_amt5", "Pay_amt6"]
    col_values = [Limit_bal, Sex, Education, Marriage, Age, Pay_1, Pay_2, Pay_3, Pay_4, Pay_5, Pay_6, Bill_amt1,
                   Bill_amt2, Bill_amt3, Bill_amt4, Bill_amt5, Pay_amt1, Pay_amt2, Pay_amt3, Pay_amt4, Pay_amt5, Pay_amt6]
    json_input = dict(zip(col_names, col_values))

    # API call to obtain the result from the Spark-backed FastAPI
    api_url_spark = f"http://{API_HOST}:{API_PORT}/predict_spark"
    response_spark = requests.post(api_url_spark, json=json_input)
    response_spark = response_spark.json()

    return templates.TemplateResponse("prediction_form.html", {"request": request, "rf_prediction_spark": response_spark["rf_prediction"]})

