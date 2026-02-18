from flask import Flask, render_template, request
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

def train_model():
    api_url = "http://api.worldweatheronline.com/premium/v1/weather.ashx"
    params = {
        'key': '679f5acc04e94339b0314215243108', 
        'q': 'London',
        'format': 'json',
        'num_of_days': '5'
    }

    response = requests.get(api_url, params=params)
    weather_data = response.json()

    current_condition = weather_data['data']['current_condition'][0]
    forecast = weather_data['data']['weather'][0]['hourly']

    df = pd.DataFrame(forecast)

    columns_of_interest = ['tempC', 'windspeedKmph', 'humidity', 'visibility']
    cleaned_data = df[columns_of_interest].dropna()

    X = cleaned_data.drop('tempC', axis=1)
    y = cleaned_data['tempC']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    windspeed = float(request.form['windspeed'])
    humidity = float(request.form['humidity'])
    visibility = float(request.form['visibility'])

    input_data = pd.DataFrame([[windspeed, humidity, visibility]],
                              columns=['windspeedKmph', 'humidity', 'visibility'])

    prediction = model.predict(input_data)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
