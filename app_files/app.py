from flask import Flask, request, jsonify, render_template, url_for

import pandas as pd
import pickle
import logging
import os

with open('ml-model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

FEATURES_MASK = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',\
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

app = Flask(__name__)

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Prediction {}".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__=='__main__':
    app.run( debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)) )
