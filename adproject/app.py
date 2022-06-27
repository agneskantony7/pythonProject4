from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import pickle
from flask import redirect, url_for
app = Flask(__name__)

model = pickle.load(open("modelg.pkl", "rb"))
@app.route('/')
    #return render_template('login.html')
def home():
    return render_template('ad.html')

#Ventricle = request.form['Ventricle']
#def create_app():
 #   app = Flask(__name__)
  #  app.config['SECRET_KEY'] = 'fytgmfdargvjjhhgf'
   #from .auth import auth
    #return app
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    #print(request.form)
    #int_features = [int(x) for x in request.form.values()]
    #final = [np.array(int_features)]
    #print(int_features)
    #print(final)
    #prediction = model.predict(final)
    #output = '{0:.{1}f}'.format(prediction[0][1], 2)

    Ventricle = request.form['Ventricle']
    Hippocampus = request.form['Hippocampus']
    WholeBrain = request.form['Whole-brain']
    Fusiform = request.form['Fusiform']
    Entorhinal = request.form['Entorhinal']
    MidTemp = request.form['Mid-temp']
    RAVLT_immediate = request.form['RAVLT immediate']
    RAVLT_learning = request.form['RAVLT Learning']
    RAVLT_forgetting = request.form['RAVLT Forgetting']
    RAVLT_perc_forgetting = request.form['RAVLT perc_forgetting']
    FDG = request.form['FDG']
    AGE = request.form['AGE']
    PTGENDER_male = request.form['PTGENDER']
    PTGENDER_female= request.form['PTGENDER']
    PTEDUCAT = request.form['PTEDUCATION']
    APOE4 = request.form['APOE4']
    form_array = np.array([[Ventricle, Hippocampus, WholeBrain, Fusiform, Entorhinal, MidTemp, RAVLT_immediate, RAVLT_learning, RAVLT_forgetting, RAVLT_perc_forgetting, FDG, AGE, PTEDUCAT, APOE4,PTGENDER_male,PTGENDER_female]])
    form_array = np.array(form_array, dtype=float)
    model = pickle.load(open("modelg.pkl", "rb"))
    #prediction = -(model.predict(form_array)[0]) * 100
    prediction = model.predict(form_array)
   # prediction = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(prediction)
    #print(prediction )#result=prediction
    if prediction >= 27 and prediction <= 30:
        result = "normal"
    elif prediction >= 21 and prediction <= 26:
        result = "mild"
    elif prediction >= 15 and prediction <= 20:
        result = "moderate"
    elif prediction >=5 and prediction <=14:
        result = "ad"
    else:
        result = "error"
    return render_template("result.html", result=result)
    #return prediction
if __name__=='__main__':
    app.run(debug=True)