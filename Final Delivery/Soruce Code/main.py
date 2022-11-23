from flask import Flask,render_template,request,redirect
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')


@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


def cross_origin(**kwargs):
    _options = kwargs


def decorator(f):
    LOG.debug("Enabling %s for cross_origin using options:%s", f, _options)
    if _options.get('automatic_options', True):
        f.required_methods = getattr(f, 'required_methods', set())
        f.required_methods.add('OPTIONS')
        f.provide_automatic_options = False

    def wrapped_function(*args, **kwargs):
        # Handle setting of Flask-Cors parameters
        options = get_cors_options(current_app, _options)

        if options.get('automatic_options') and request.method == 'OPTIONS':
            resp = current_app.make_default_options_response()
        else:
            resp = make_response(f(*args, **kwargs))

        set_cors_headers(resp, options)
        setattr(resp, FLASK_CORS_EVALUATED, True)
        return resp

    return update_wrapper(wrapped_function, f)
    return decorator


@app.route('/predict',methods=['POST'])
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')
    columns = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
    data = np.array([car_model, company, year, driven, fuel_type])
    prediction=model.predict(pd.DataFrame(columns,data.reshape(1,5)))
    print(prediction)

    return str(np.round(prediction[0],2))


if __name__=='__main__':
    app.run(debug=True)