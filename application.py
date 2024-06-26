from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

app = Flask(__name__)


## Route for homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        name = (request.form.get('name'))
        company = (request.form.get('company'))
        year = float(request.form.get('year'))
        kms_driven = (request.form.get('kms_driven'))
        fuel_type = (request.form.get('fuel_type'))

        result = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([[name, company, year, kms_driven, fuel_type]]).reshape(1,
                                                                                                                  5)))
        print(result)

        return render_template('index.html', result=result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
