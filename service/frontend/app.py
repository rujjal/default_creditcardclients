
import uvicorn
from fastapi import FastAPI, Form
import requests
import os
from pyspark.sql import SparkSession
from fastapi import Request
from fastapi.templating import Jinja2Templates

# Creating the Spark session
spark = SparkSession.builder.appName("SparkFrontendApp").getOrCreate()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

API_HOST = str(os.getenv("BACKEND_HOST"))
API_PORT = str(os.getenv("BACKEND_PORT"))


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

    
    col_names = ["Limt_bal", "Sex", "Education", "Marriage", "Age", "Pay_1", " Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6",
                 "Bill_amt1", "Bill_amt2", "Bill_amt3", "Bill_amt4", "Bill_amt5", "Pay_amt1", "Pay_amt2", "Pay_amt3",
                 "Pay_amt4", "Pay_amt5", "Pay_amt6"]
    col_values = [Limit_bal, Sex, Education, Marriage, Age, Pay_1, Pay_2, Pay_3, Pay_4, Pay_5, Pay_6, Bill_amt1,
                   Bill_amt2, Bill_amt3, Bill_amt4, Bill_amt5, Pay_amt1, Pay_amt2, Pay_amt3, Pay_amt4, Pay_amt5, Pay_amt6]
    json_input = dict(zip(col_names, col_values))

    
    api_url_spark = f"http://{API_HOST}:{API_PORT}/predict_spark"
    response_spark = requests.post(api_url_spark, json=json_input)
    response_spark = response_spark.json()

    return templates.TemplateResponse("prediction_form.html", {"request": request, "rf_prediction_spark": response_spark["rf_prediction"]})

