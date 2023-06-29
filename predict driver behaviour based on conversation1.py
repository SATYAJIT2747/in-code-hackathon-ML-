#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import speech_recognition as sr


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

# speech recognition
recognizer = sr.Recognizer()

# Record audio from the microphone
with sr.Microphone() as source:
    print("Speak now...")
    audio = recognizer.listen(source)

try:
    # Convert speech to text
    new_conversation = recognizer.recognize_google(audio)
    print("You said:", new_conversation)
    new_conversation_vectorized = vectorizer.transform([new_conversation])

    # Predict category for the new conversation
    predicted_category = rf_classifier.predict(new_conversation_vectorized)[0]

    # Print the predicted category
    print("Predicted Driver Behavior Category:", predicted_category)

except sr.UnknownValueError:
    print("Unable to recognize speech")
except sr.RequestError:
    print("Speech recognition request error")


# In[ ]:




