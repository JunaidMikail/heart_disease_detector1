import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__,template_folder='templates',static_folder='static',)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['age', 'sex', 'restingbp', 'chestpaintype', 'cholesterol', 'fastingbs', 'restingecg',
                     'maxhr', 'exerciseangina', 'oldpeak', 'st_slope']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "** Heart Disease **"
    else:
        res_val = "No Heart Disease "

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
