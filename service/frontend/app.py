
import uvicorn
from fastapi import FastAPI, Form
import requests
import os
from fastapi import Request
from fastapi.templating import Jinja2Templates


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

    # in the previous version here you missed the "Bill_amt6" and wrote uncorrectly "Pay_2" so when the request was sent
    # to the backend it cant unmarshal the input in the Debtor class and because of this the return is the http error 422 
    col_names = ["Limit_bal", "Sex", "Education", "Marriage", "Age", "Pay_1", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6",
                 "Bill_amt1", "Bill_amt2", "Bill_amt3", "Bill_amt4", "Bill_amt5", "Bill_amt6", "Pay_amt1", "Pay_amt2", "Pay_amt3",
                 "Pay_amt4", "Pay_amt5", "Pay_amt6"]
    col_values = [Limit_bal, Sex, Education, Marriage, Age, Pay_1, Pay_2, Pay_3, Pay_4, Pay_5, Pay_6, Bill_amt1,
                   Bill_amt2, Bill_amt3, Bill_amt4, Bill_amt5, Bill_amt6, Pay_amt1, Pay_amt2, Pay_amt3, Pay_amt4, Pay_amt5, Pay_amt6]
    json_input = dict(zip(col_names, col_values))
    
    api_url_spark = f"http://{API_HOST}:{API_PORT}/predict"
    response_spark = requests.post(api_url_spark, json=json_input)
    response_spark = response_spark.json()
    
    return templates.TemplateResponse("prediction_form.html", {"request": request, "rf_prediction": response_spark["rf_prediction"]})

if __name__ == '__main__':
    # need to use 0.0.0.0 in docker as localhost to avoid error during startup
    uvicorn.run(app, host='0.0.0.0', port=8080)
