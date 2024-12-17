import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocessing(generator_file_path, anemometer_file_path):
    generator_data = pd.read_csv(generator_file_path)
    wind_speed_data = pd.read_csv(anemometer_file_path)
    
    generator_data.drop(columns=['name', 'TURBINEID', 'DEVICETYPE', 'DEVICEID'], inplace=True)
    wind_speed_data.drop(columns=['name', 'windspeed_MW'], inplace=True)
    
    generator_data = generator_data.drop_duplicates(subset='time')
    wind_speed_data = wind_speed_data.drop_duplicates(subset='time')
    
    # Merging both DataFrames based on 'time' column, keeping only the intersection
    merged_df = pd.merge(wind_speed_data, generator_data, on='time', how='inner')
    
    # Filtering MPPT Data
    merged_df['IREG_SPEED'] = pd.to_numeric(merged_df['IREG_SPEED'], errors='coerce')
    filtered_data = merged_df[(merged_df['IREG_SPEED'] >= 400) & (merged_df['IREG_SPEED'] <= 750)]
    
    filtered_data.reset_index(drop=True, inplace=True)
    data=filtered_data.drop(columns=['time'])
    
    data['IREG_UBUS'] = pd.to_numeric(data['IREG_UBUS'], errors='coerce')
    data['IREG_IBUS'] = pd.to_numeric(data['IREG_IBUS'], errors='coerce')
    data['IREG_PWM'] = pd.to_numeric(data['IREG_PWM'], errors='coerce')
    data['IREG_CHOPPER_PWM'] = pd.to_numeric(data['IREG_CHOPPER_PWM'], errors='coerce')
    data['IREG_IGBT_TEMP'] = pd.to_numeric(data['IREG_IGBT_TEMP'], errors='coerce')
    data['IREG_MOTOR_TEMP'] = pd.to_numeric(data['IREG_MOTOR_TEMP'], errors='coerce')
    data['windspeed'] = pd.to_numeric(data['windspeed'], errors='coerce')
    
    X = data.drop(columns=['IREG_CHOPPER_PWM', 'IREG_IGBT_TEMP', 'IREG_MOTOR_TEMP'])
    
    y = X['IREG_SPEED']
    y = pd.DataFrame(y)
    X = X.drop(columns=['IREG_SPEED'])
    
    return X, y

def standardScaling(X, y):
    X_scaler = StandardScaler()
    X_scaler.fit(X)
    X_scaled = X_scaler.transform(X)

    y_scaler = StandardScaler()
    y_scaler.fit(y)
    y_scaled = y_scaler.transform(y)
    
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    y_scaled = pd.DataFrame(y_scaled, columns=y.columns)
    
    return X_scaled, y_scaled, X_scaler, y_scaler

def splitting_dataset(X, y, test_size):
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size, random_state=42)
    return train_X, val_X, train_y, val_y