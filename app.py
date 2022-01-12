import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    x1 = request.form["z1"]
    data = pd.read_csv('C:\\Users\\DELL\\Desktop\\covid_data_prediction\\active cases in india.csv')
    data = data[['id','cases']]
    x =  np.array(data['id']).reshape(-1,1)
    y =  np.array(data['cases']).reshape(-1,1)
    polyFeat = PolynomialFeatures(degree=2)
    x = polyFeat.fit_transform(x)
    model = linear_model.LinearRegression()
    model.fit(x,y)
    accuracy = model.score(x,y)
    print(f'Accuracy: {round(accuracy*100,3)}%')
    y0 = model.predict(x)
    days = int(x1)
    pred = round(int(model.predict(polyFeat.fit_transform([[126 + days]])))/1000000,2),'Million'
    print(pred)
    x1 = np.array(list(range(1,126 + days))).reshape(-1,1)
    y1 = model.predict(polyFeat.fit_transform(x1)) 
    return render_template('index.html', predict = 'Approx {}  covid cases will be new in upcoming {} days'.format(pred,days))


if __name__ == "__main__":
    app.run(debug=True)