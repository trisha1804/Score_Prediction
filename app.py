import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__,template_folder='templates')

model=pickle.load(open('regressor.pkl','rb'))

@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    output=round(prediction[0],2)
    
    return render_template('pred.html',prediction_text=f'If the student studies for {int_features[0]} hours/day, the score is {output}%.')

if __name__=="__main__":
    app.run(debug=True)