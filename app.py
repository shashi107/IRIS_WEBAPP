from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import pandas as pd
import pickle as plk

app = Flask(__name__)
@app.route("/")
def index():
 
    return render_template("index.html")                
 
@app.route("/predict",methods = ["POST"])
def predictApp():
 
    with open('iris.pkl','rb') as file:
        gbc=plk.load(file)


 
    sepalLenght = request.form.get("sepalLenght")
    sepalWidth = request.form.get("sepalWidth")
    petalLenght = request.form.get("petalLenght")
    petalWidth = request.form.get("petalWidth")
 
    setosa = np.array([0])
    versicolor = np.array([1])
    virginica = np.array([2])
    real_values = np.array([sepalLenght,sepalWidth,petalLenght,petalWidth]).reshape(1, -1)
 
    predict_GBC = gbc.predict(real_values)
 
    return render_template("index.html", setosa = setosa, versicolor = versicolor, virginica = virginica, predict_GBC = predict_GBC )                
    
 
if __name__ == "__main__":
    app.run()