import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#pip install yfinance
import yfinance as yf
from datetime import datetime, timedelta

from flask import Flask
from flask import render_template, request
app = Flask(__name__)



############  functions  ##################

# Getting the last stock values
def Get_Last_Stock_Values(Tickers):
    # Get current date and time
    current_time = datetime.now()
    # Get closing time of the stock market (Assuming 16:00 as closing time)
    closing_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    # Check if current time is after closing time
    if current_time > closing_time:
        # If after closing time, set start_date as today
        end_date = current_time.strftime("%Y-%m-%d")
    else:
        # If before closing time, set start_date as yesterday
        end_date = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
    # Set end_date as N days ago from the start_date
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    for Ticker in Tickers:
        stock_data = yf.download(Ticker, start=start_date, end=end_date)
        if len(stock_data)>0: 
            stock_data.to_csv(f'Data/Stock_Values/{Ticker}.csv')
        else:
            print('Warning: No data returned from API')


# Function to load data for a given Ticker symbol
def load_data(look_back, Ticker, step_lenght):
    Data_url= f'Data/Stock_Values/{Ticker}.csv' 
    dataset= pd.DataFrame(pd.read_csv(Data_url)['Close']).values.astype('float32') 
    dates_df= pd.read_csv(Data_url)['Date']
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    dataset= dataset[::-step_lenght][::-1]
    dates_df= dates_df[::-step_lenght][::-1]

    testX= dataset[-look_back:, 0].reshape(1, 1, -1)
    dates= dates_df[-look_back:]
    return testX, scaler, dates.values


# Function to generate future dates
def generate_dates(dates, No_steps, step_lenght):
    next_dates = []
    last_date= datetime.strptime(dates[-1], "%Y-%m-%d")
    for _ in range(No_steps):
        next_date = (last_date + timedelta(days= step_lenght)).strftime("%Y-%m-%d")
        last_date= datetime.strptime(next_date, "%Y-%m-%d")
        next_dates.append(next_date)
    Dates= np.concatenate((dates, next_dates), axis=-1)
    return Dates



############  Pre-defined values  ##################
look_back= 6
No_steps= 5

Model_url_daily= 'Saved_Models/stock_prediction_model-close-value-Daily_keras_format'
Model_url_weekly= 'Saved_Models/stock_prediction_model-close-value-Daily_keras_format'

Tickers = ['AZEK', 'BCC', 'COCO', 'FTAI', 'MDB', 'PLAB']  # Example Ticker symbol 

# Getting the last stock values
Get_Last_Stock_Values(Tickers)

# Load the saved model
model_daily = tf.keras.models.load_model(Model_url_daily)
model_weekly = tf.keras.models.load_model(Model_url_weekly)


############ Flask - routes ##################

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict-daily', methods=['GET', 'POST']) 
def run_model_daily():
    if request.method == 'POST':
        Ticker = request.form.get('symbol')
        step_lenght= 1
        testX, scaler, dates = load_data(look_back, Ticker, step_lenght) 
        Dates = generate_dates(dates, No_steps, step_lenght) 

        Data= testX
        for _ in range(1, No_steps+1):
            testPredict = model_daily.predict(Data[0][0][-look_back:].reshape(1,1,-1))    
            Data = np.concatenate((Data, np.reshape(testPredict, (1, 1, 1))), axis=-1)
        testPredict = scaler.inverse_transform(Data[0]).reshape(1,1,-1)

        Xtest= testPredict[0][0][0:look_back]
        Xpred= testPredict[0][0][look_back-1:]

        # plot baseline and predictions  
        plt.figure(figsize=(8, 5)) 
        plt.plot(Dates[:look_back], Xtest, 'bo-')
        plt.plot(Dates[look_back-1: look_back+No_steps+1], Xpred, 'ro--')
        plt.xticks(rotation= 45, fontsize=7) 
        plt.title(f'Symbol: {Ticker}', loc= 'center')
        plt.savefig('static/plot_daily.png') 
        plt.close()
    return render_template('predict-daily.html', symbols=Tickers)



@app.route('/predict-weekly', methods=['GET', 'POST']) 
def run_model_weekly():
    if request.method == 'POST':
        Ticker = request.form.get('symbol')
        step_lenght= 5
        testX, scaler, dates = load_data(look_back, Ticker, step_lenght) 
        Dates = generate_dates(dates, No_steps, step_lenght) 

        Data= testX
        for _ in range(1, No_steps+1):
            testPredict = model_weekly.predict(Data[0][0][-look_back:].reshape(1,1,-1))    
            Data = np.concatenate((Data, np.reshape(testPredict, (1, 1, 1))), axis=-1)
        testPredict = scaler.inverse_transform(Data[0]).reshape(1,1,-1)

        Xtest= testPredict[0][0][0:look_back]
        Xpred= testPredict[0][0][look_back-1:]

        # plot baseline and predictions  
        plt.figure(figsize=(8, 5)) 
        plt.plot(Dates[:look_back], Xtest, 'bo-')
        plt.plot(Dates[look_back-1: look_back+No_steps+1], Xpred, 'ro--')
        plt.xticks(rotation= 45, fontsize=7) 
        plt.title(f'Symbol: {Ticker}', loc= 'center')
        plt.savefig('static/plot_weekly.png') 
        plt.close()
    return render_template('predict-weekly.html', symbols=Tickers)



############  Running model  ##################

if __name__ == "__main__":
    app.run(host="0.0.0.0")
