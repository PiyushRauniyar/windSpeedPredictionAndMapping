import numpy as np
import time
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error
import serial  # pySerial for reading from USB

# Load the trained model and scaler (fix the file path)
model = load_model('model/lstm_wind_speed_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Flag to indicate whether to simulate the windspeed data
simulate_device = False

try:
    # Try to initialize serial connection to the anemometer
    ser = serial.Serial('COM3', 9600)  # Adjust for Windows, '/dev/ttyUSB0' for Linux/macOS
    print("Anemometer connected on COM3.")
except serial.SerialException:
    simulate_device = True  # If the device is not connected, switch to simulation mode
    print("Anemometer not connected, running in simulation mode.")

# Function to read wind speed data from the anemometer via USB or simulate data
def get_windspeed_from_anemometer():
    if simulate_device:
        # Simulate wind speed data if device is not connected
        wind_speed = np.random.uniform(5, 15)  # Generate random windspeed between 5 and 15
        print(f"Simulated wind speed: {wind_speed}")
        return wind_speed
    else:
        try:
            if ser.in_waiting > 0:  # Check if there's data in the buffer
                wind_speed_data = ser.readline().decode('utf-8').strip()
                wind_speed = float(wind_speed_data)
                return wind_speed
            else:
                print("No data from the anemometer.")
                return None
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            return None
        except ValueError as e:
            print(f"Data conversion error: {e}")
            return None

# Function to normalize incoming wind speed data
def normalize_data(data, scaler):
    return scaler.transform(np.array(data).reshape(-1, 1))

# Function to predict the next 10 seconds
def predict_next_10_seconds(model, data, scaler, time_step=10):
    input_data = np.array(data).reshape(1, time_step, 1)
    prediction = model.predict(input_data)
    prediction_actual = scaler.inverse_transform(prediction)
    return prediction_actual[0][0]

# Function to monitor model performance using MAE
def monitor_performance(y_true, y_pred, threshold=5):
    mae = mean_absolute_error(y_true, y_pred)
    if mae > threshold:
        print(f"Model MAE {mae} exceeds threshold {threshold}. Retraining required.")
        return True
    return False

# Function to detect sudden changes in wind speed
def detect_sudden_change(recent_data, change_threshold=10):
    if len(recent_data) >= 2:
        change = abs(recent_data[-1] - recent_data[-2])
        if change > change_threshold:
            print(f"Sudden change detected: {change}. Retraining required.")
            return True
    return False

# Function to retrain the LSTM model
def retrain_model(recent_training_data, recent_training_labels):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    # Define the LSTM model structure
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile and retrain the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(recent_training_data, recent_training_labels, epochs=5, batch_size=64)
    
    # Save the retrained model (overwrite the old model)
    model.save('backend/model/lstm_wind_speed_model.h5')
    print("Model retrained and saved.")

# Initialize recent data and training data lists
recent_data = []  # To store the last 60 wind speed points for prediction
recent_training_data = []  # To store input data for retraining
recent_training_labels = []  # To store labels (actual wind speed) for retraining

# Simulating real-time data reception and prediction
while True:
    # Get new wind speed data from the anemometer or simulate data
    new_wind_speed = get_windspeed_from_anemometer()
    
    if new_wind_speed is not None:
        # Append the new data to recent_data list
        recent_data.append(new_wind_speed)
    
        # Append the data to the training dataset for potential retraining
        if len(recent_training_data) == 60:
            recent_training_data.append(normalize_data(recent_data[:-1], scaler).tolist())
            recent_training_labels.append(new_wind_speed)

        # If we have more than 60 points, keep only the last 60
        if len(recent_data) > 60:
            recent_data = recent_data[-60:]
        
        # Make predictions when we have 60 data points
        if len(recent_data) == 60:
            recent_data_normalized = normalize_data(recent_data, scaler)
            predicted_wind_speed = predict_next_10_seconds(model, recent_data_normalized[-10:], scaler)
            print(f'Predicted wind speed for the next 10 seconds: {predicted_wind_speed}')

            # Monitor performance by comparing actual wind speed to predicted
            y_true = [new_wind_speed]  # In real-time, this would be the actual label
            if monitor_performance(y_true, [predicted_wind_speed], threshold=5):
                print("Performance degraded. Retraining model...")
                retrain_model(np.array(recent_training_data), np.array(recent_training_labels))
            
            # Detect sudden changes in wind speed
            if detect_sudden_change(recent_data, change_threshold=10):
                print("Sudden wind speed change detected. Retraining model...")
                retrain_model(np.array(recent_training_data), np.array(recent_training_labels))
    
    # Sleep for 10 seconds to simulate real-time data reception
    time.sleep(10)
