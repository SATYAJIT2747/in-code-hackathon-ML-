#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from twilio.rest import Client

# accelerometer and gyroscope data
accelerometer_data = pd.read_csv('Accelerometer.csv', delimiter=',')
gyroscope_data = pd.read_csv('Gyroscope.csv', delimiter=',')

# Merge based on timestamp and milliseconds
merged_data = pd.merge(accelerometer_data, gyroscope_data, on=['Timestamp', 'Milliseconds'])

# Magnitude of acceleration vector
merged_data['accel_magnitude'] = (merged_data['X_accel']**2 + merged_data['Y_accel']**2 + merged_data['Z_accel']**2)**0.5

# Threshold values for each behavior (set according to your requirements)
aggressive_threshold = 10.0
risky_threshold = 15.0

# Labels based on the magnitude
merged_data.loc[merged_data['accel_magnitude'] < aggressive_threshold, 'label'] = 'Normal'
merged_data.loc[(merged_data['accel_magnitude'] >= aggressive_threshold) & (merged_data['accel_magnitude'] < risky_threshold), 'label'] = 'Aggressive'
merged_data.loc[merged_data['accel_magnitude'] >= risky_threshold, 'label'] = 'Risky'

# Split into training and testing
X = merged_data[['X_accel', 'Y_accel', 'Z_accel', 'X_gyro', 'Y_gyro', 'Z_gyro']]
y = merged_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train
rf_classifier.fit(X_train, y_train)

# Predict
y_pred = rf_classifier.predict(X_test)

# Performance
print(classification_report(y_test, y_pred))

# Manual data input for the new data point
new_X_accel = float(input("Enter the X_accel value: "))
new_Y_accel = float(input("Enter the Y_accel value: "))
new_Z_accel = float(input("Enter the Z_accel value: "))
new_X_gyro = float(input("Enter the X_gyro value: "))
new_Y_gyro = float(input("Enter the Y_gyro value: "))
new_Z_gyro = float(input("Enter the Z_gyro value: "))

new_data = pd.DataFrame({'X_accel': [new_X_accel], 'Y_accel': [new_Y_accel], 'Z_accel': [new_Z_accel], 'X_gyro': [new_X_gyro], 'Y_gyro': [new_Y_gyro], 'Z_gyro': [new_Z_gyro]})

# The trained model now predicts the category
predicted_category = rf_classifier.predict(new_data)

print("Predicted Driver Behavior Category:", predicted_category)

# Twilio account credentials
account_sid = 'AC67448eb3b0b9100b11e198108ed3975d'
auth_token = 'c5416ff1425c46e93b86c4227c7ec84a'

# Twilio phone number and recipient's phone number
twilio_number = '+16186346543'
recipient_number = '+917848015623'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Send a message if the predicted category is risky or aggressive
if predicted_category in ['Risky', 'Aggressive']:
    message_body = 'Driver behavior is risky or aggressive!'
    message = client.messages.create(
        body=message_body,
        from_=twilio_number,
        to=recipient_number
    )
    print("Message sent successfully. SID:", message.sid)

    

