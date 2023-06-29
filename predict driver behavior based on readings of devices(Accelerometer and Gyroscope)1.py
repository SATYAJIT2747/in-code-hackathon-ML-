#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[4]:




# accelerometer and gyroscope data 
accelerometer_data = pd.read_csv('Accelerometer.csv', delimiter=',')
gyroscope_data = pd.read_csv('Gyroscope.csv', delimiter=',')

# Merge based on timestamp and milliseconds
merged_data = pd.merge(accelerometer_data, gyroscope_data, on=['Timestamp', 'Milliseconds'])

# magnitude of acceleration vector
merged_data['accel_magnitude'] = (merged_data['X_accel']**2 + merged_data['Y_accel']**2 + merged_data['Z_accel']**2)**0.5

# threshold values for each behavior (u take on ur own acoording to u)
aggressive_threshold = 10.0
risky_threshold = 15.0

# labels based on the magnitude 
merged_data.loc[merged_data['accel_magnitude'] < aggressive_threshold, 'label'] = 'Normal'
merged_data.loc[(merged_data['accel_magnitude'] >= aggressive_threshold) & (merged_data['accel_magnitude'] < risky_threshold), 'label'] = 'Aggressive'
merged_data.loc[merged_data['accel_magnitude'] >= risky_threshold, 'label'] = 'Risky'

# Split  into training and testing 
X = merged_data[['X_accel', 'Y_accel', 'Z_accel', 'X_gyro', 'Y_gyro', 'Z_gyro']]
y = merged_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train 
rf_classifier.fit(X_train, y_train)

# Predict 
y_pred = rf_classifier.predict(X_test)

#  performance
print(classification_report(y_test, y_pred))


# In[5]:


import pandas as pd

# ur new  data point
new_data = pd.DataFrame({'X_accel': [0.5], 'Y_accel': [-0.3], 'Z_accel': [0.2], 'X_gyro': [0.1], 'Y_gyro': [-0.1], 'Z_gyro': [0.3]})

#the trained model now  predict the category
predicted_category = rf_classifier.predict(new_data)

print(predicted_category)


# In[ ]:




