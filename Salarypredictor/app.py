import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if(request.method=='POST'):
        int_features=request.form['experience']
        int_features=[float(i) for i in int_features]
        final_features = [int_features]
        print(final_features)
        prediction = model.predict(final_features)
        return render_template('index.html', prediction_text='The predicted salary is {}'.format(prediction))


if __name__=="__main__":
    app.run(debug=True)