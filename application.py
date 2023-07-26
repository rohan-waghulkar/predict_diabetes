from flask import Flask,request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app=application
scaler=pickle.load(open('/config/workspace/Model/standerdScaler.pkl','rb'))
model=pickle.load(open('/config/workspace/Model/modelforprediction.pkl','rb'))
## Route for Home page 
@app.route("/")
def index():
    return render_template('index.html')
## Route for Single data point prediction
@ app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=''
    if request.method=='POST':
        pregnancies=int(request.form.get('pregnancies'))
        Glucose=float(request.form.get('glucose'))
        Bloodpressure=float(request.form.get('bloodpressure'))
        SkinThikness=float(request.form.get("skinthickness"))
        Insulin=float(request.form.get("insulin"))
        BMI=float(request.form.get('bmi'))
        DiabetesPedigreeFunction=float(request.form.get('dpf'))
        Age=float(request.form.get('age'))
        new_data=scaler.transform([[pregnancies ,Glucose,Bloodpressure,SkinThikness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
        if predict[0]==1:
            result='Diabetic'
        else:
            result='Non-Diabetic'
        return render_template("single_prdiction.html",result=result)
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
