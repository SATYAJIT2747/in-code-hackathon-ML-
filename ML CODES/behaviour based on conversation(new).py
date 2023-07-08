#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from twilio.rest import Client

data = pd.DataFrame({
    'Conversation': [
        "Customer: How's your day going? Driver: It's been great so far. How about yours?",
        "Customer: Can you please drive a bit more carefully? Driver: Sure, I'll be more cautious.",
        "Customer: Thank you for the ride. Driver: You're welcome! Have a nice day!",
        "Customer: Do you always drive at this speed? Driver: I apologize. I'll slow down.",
        "Customer: How long have you been driving? Driver: I've been driving for 5 years now.",
        "Customer: That was a reckless maneuver! Driver: I'm sorry, it won't happen again.",
        "Customer: Please stop talking on the phone while driving. Driver: My apologies, I'll hang up.",
        "Customer: You're driving too fast! Driver: I'll try to maintain a safer speed.",
        "Customer: I appreciate your safe and calm driving. Driver: Thank you, I prioritize safety.",
        "Customer: Watch out! You almost hit that car. Driver: I apologize, I didn't see it."
    ],
    'Label': ['Normal', 'Risky', 'Normal', 'Aggressive', 'Normal', 'Aggressive', 'Risky', 'Aggressive', 'Normal', 'Risky']
})

vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(data['Conversation'])
rf_classifier = RandomForestClassifier()
# Training
rf_classifier.fit(X_vectorized, data['Label'])

# Manually enter the conversation
new_conversation = input("Enter the conversation: ")
new_conversation_vectorized = vectorizer.transform([new_conversation])

# Predict category for the new conversation
predicted_category = rf_classifier.predict(new_conversation_vectorized)[0]

# Print the predicted category
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


# In[ ]:




