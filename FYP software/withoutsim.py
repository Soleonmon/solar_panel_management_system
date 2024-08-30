import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import os
import time

# Function to save data to Excel
def save_to_excel(file_path, cycle_info):
    df = pd.DataFrame(cycle_info)
    df.to_excel(file_path, index=False)

# Load data from Excel file
data = pd.read_excel("hourly datanew.xlsx")

# Drop rows where the target variable has NaN values
data = data.dropna(subset=['solarradiation'])

# Initialize list to store cycle information
cycle_info = []

# Main loop to make predictions and save to Excel
num_cycles = 336  # Number of simulation cycles

for cycle in range(num_cycles):
    # Train a decision tree regressor for each cycle
    X = data[['temp', 'humidity', 'windspeed', 'cloudcover', 'uvindex', 'solarenergy']]  # Features
    y = data['solarradiation']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=cycle)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Define the condition based on multiple factors, using dynamic values from the Excel file
    humidity1, humidity2 = data.at[cycle, 'humidity1'], data.at[cycle, 'humidity2']
    windspeed1, windspeed2 = data.at[cycle, 'windspeed1'], data.at[cycle, 'windspeed2']
    temp1, temp2 = data.at[cycle, 'temp1'], data.at[cycle, 'temp2']
    cloudcover1, cloudcover2 = data.at[cycle, 'cloudcover1'], data.at[cycle, 'cloudcover2']
    uvindex1, uvindex2 = data.at[cycle, 'uvindex1'], data.at[cycle, 'uvindex2']
    solarenergy1, solarenergy2 = data.at[cycle, 'solarenergy1'], data.at[cycle, 'solarenergy2']

    condition = (
        (X_test['humidity'] > humidity1) & (X_test['humidity'] <= humidity2) &
        (X_test['windspeed'] > windspeed1) & (X_test['windspeed'] <= windspeed2) &
        (X_test['temp'] > temp1) & (X_test['temp'] <= temp2) &
        (X_test['cloudcover'] > cloudcover1) & (X_test['cloudcover'] <= cloudcover2) &
        (X_test['uvindex'] > uvindex1) & (X_test['uvindex'] <= uvindex2) &
        (X_test['solarenergy'] > solarenergy1) & (X_test['solarenergy'] <= solarenergy2)
    )

    # Filter test data based on the condition
    filtered_X_test = X_test[condition]

    print(f'Cycle {cycle + 1}: Conditions -')
    print(f'Humidity: >{humidity1} & <={humidity2}')
    print(f'Windspeed: >{windspeed1} & <={windspeed2}')
    print(f'Temp: >{temp1} & <={temp2}')
    print(f'Cloudcover: >{cloudcover1} & <={cloudcover2}')
    print(f'UV Index: >{uvindex1} & <={uvindex2}')
    print(f'Solar Energy: >{solarenergy1} & <={solarenergy2}')
    print(f'Filtered Data Points: {len(filtered_X_test)}')
    print('\n')

    if not filtered_X_test.empty:
        # Make predictions on the filtered test data
        filtered_y_pred = model.predict(filtered_X_test)

        # Calculate and print the average of the predicted values
        average_prediction = round(np.mean(filtered_y_pred), 6)
        print("Average of Predicted Values:", average_prediction)

        # Save the cycle details and predictions to the list
        cycle_info.append({
            'Cycle': cycle + 1,
            'Mean Squared Error': mse,
            'Filtered Data Points': len(filtered_X_test),
            'Average Prediction': average_prediction
        })
    else:
        print(f'Cycle {cycle + 1}: No data points satisfy the condition.')
        # Save the cycle details with no prediction
        cycle_info.append({
            'Cycle': cycle + 1,
            'Mean Squared Error': mse,
            'Filtered Data Points': 0,
            'Average Prediction': None
        })

    # Save data to Excel every cycle
    save_to_excel('finaldatasheet.xlsx', cycle_info)

    # Pause for a moment before next cycle (optional)
    time.sleep(0.1)  # Adjust as needed

# Clear the Excel file for the next simulation run
#if os.path.exists('finaldatasheet.xlsx'):
 #   os.remove('finaldatasheet.xlsx')


