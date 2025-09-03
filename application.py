import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
## import ridge regressor and standardscaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))



@app.route("/")
def index():
  return render_template('index.html')

@app.route('/predictdata', methods =['GET','POST'])
def predict_datapoint():
  if request.method == "POST":
    try:
      # Expected feature order should match model training
      # Using common FWI predictors from your dataset
      features = [
        float(request.form.get('Temperature')),
        float(request.form.get('RH')),
        float(request.form.get('Ws')),
        float(request.form.get('Rain')),
        float(request.form.get('FFMC')),
        float(request.form.get('DMC')),
        float(request.form.get('DC')),
        float(request.form.get('ISI')),
        float(request.form.get('BUI')),
      ]

      input_array = np.array(features).reshape(1, -1)
      scaled_input = standard_scaler.transform(input_array)
      result = ridge_model.predict(scaled_input)[0]

      return render_template('home.html', prediction=round(float(result), 3))
    except Exception as e:
      return render_template('home.html', error=str(e))
  else:
    return render_template('home.html')

if __name__ == "__main__":
  app.run(host="0.0.0.0")

