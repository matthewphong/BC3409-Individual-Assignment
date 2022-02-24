#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
app = Flask(__name__)

from flask import request, render_template
import joblib
from scipy import stats
import numpy as np 

@app.route("/", methods =["GET","POST"])
def index():
    if request.method == "POST":
        income= request.form.get("income")
        age= request.form.get("age")
        loan= request.form.get("loan")
        print(income,age,loan)
        lr_model=joblib.load("LR")
        lr_pred= lr_model.predict([[float(income),float(age),float(loan)]])
        lr_s = "The Logistic Regression predicted default score is: " + str(lr_pred)
        dt_model=joblib.load("DT")
        dt_pred= dt_model.predict([[float(income),float(age),float(loan)]])
        dt_s = "The Decision Tree predicted default score is: " + str(dt_pred)
        rf_model=joblib.load("RF")
        rf_pred= rf_model.predict([[float(income),float(age),float(loan)]])
        rf_s = "The Random Forest predicted default score is: " + str(rf_pred)
        xg_model=joblib.load("XG")
        xg_pred= xg_model.predict([[float(income),float(age),float(loan)]])
        xg_s = "The XGBoost predicted default score is: " + str(xg_pred)
        nn_model=joblib.load("NN")
        nn_pred= nn_model.predict([[float(income),float(age),float(loan)]])
        nn_s = "The Neural Network predicted default score is: " + str(nn_pred)
        return(render_template("index.html", lr_result=lr_s, dt_result=dt_s, rf_result=rf_s, xg_result=xg_s, nn_result=nn_s,))
    else:
        return(render_template("index.html", result="2"))

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=int("8473"))


# In[ ]:




