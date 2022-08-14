from email.policy import default
from pickle import FALSE, TRUE
from flask import Flask, render_template, request
import pickle
from datetime import datetime
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))



@app.route("/")
def Homee():
    return render_template('indexself.html')
    

@app.route("/predict", methods=['POST'])
def submit():
    if request.method=="POST":
        age=request.form['Age']
        chest=request.form['chest']
        RBP=request.form['RBP']
        sex=request.form['sex']
        CHLS=request.form['CHLS']
        FBS=request.form['FBS']
        RECG=request.form['RECG']
        MHR=request.form['MHR']
        EXA=request.form['EXA']
        oldpeak=request.form['oldpeak']
        ST=request.form['ST']

        input=(age,sex,chest,RBP,CHLS,FBS,RECG,MHR,EXA,oldpeak,ST)
        #input=(41,1,2,120,157,0,0,182,0,0,1)
        features=[np.asarray(input)]
        prediction=model.predict(features)


    return render_template('indexself.html',prediction=prediction)
if __name__=='__main__':
    app.run(debug=True,port=8000)