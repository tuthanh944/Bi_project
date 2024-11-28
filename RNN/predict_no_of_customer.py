import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt

def prepare_data_RNN(data, sequence_length=28, test_size=28):
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
    data = data.dropna(subset=['order_date'])
    
    daily_customers = data.groupby('order_date')['cust_id'].nunique().reset_index()
    daily_customers.columns = ['date', 'unique_customers']
    
    data_values = daily_customers['unique_customers'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_values)

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, sequence_length)
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    return X_train, X_test, y_train, y_test, scaler, data_scaled, daily_customers
def train_rnn_model(X_train, y_train, X_test, y_test, sequence_length=28, epochs=20, batch_size=8):
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    # model.save('model/rnn_model.h5') 
    model.save('model/rnn_model.keras')

    return model
def evaluate_and_forecast_rnn(model, X_test, y_test, scaler, data_scaled, daily_customers, sequence_length=28, forecast_days=7):
    y_pred_scaled = model.predict(X_test)
    
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred_scaled)
    
    last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    forecasts = []

    for _ in range(forecast_days):
        next_pred_scaled = model.predict(last_sequence)[0][0]
        forecasts.append(next_pred_scaled)
        
        next_sequence = np.append(last_sequence[0][1:], [[next_pred_scaled]], axis=0)
        last_sequence = next_sequence.reshape(1, sequence_length, 1)

    forecasts_actual = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    
    test_dates = daily_customers['date'][-len(y_test_actual):].reset_index(drop=True)
    comparison_df = pd.DataFrame({
        'date': test_dates,
        'actual': y_test_actual.flatten(),
        'predicted': y_pred_actual.flatten()
    })
    
    forecast_dates = pd.date_range(start=daily_customers['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecasted_unique_customers': forecasts_actual
    })
    return comparison_df, forecast_df


# # Ví dụ sử dụng
# data_path = 'F:\Bussiness Inteligence\Bi_project\data\sales_06_FY2020-21.csv'
# data = pd.read_csv(data_path)
# # model, data_scaled, scaler, daily_customers, X_test, y_test = predict_with_rnn(data)
# X_train, X_test, y_train, y_test, scaler, data_scaled, daily_customers=prepare_data_RNN(data)
# train_rnn_model(X_train, y_train, X_test, y_test)
# # Đánh giá và dự đoán 7 ngày tiếp theo
# test_results, forecast_results = evaluate_and_forecast(model, X_test, y_test, scaler, data_scaled, daily_customers)

# # Hiển thị kết quả
# print("Test Results:")
# print(test_results)
# print("\nForecast Results (Next 7 Days):")
# print(forecast_results)
