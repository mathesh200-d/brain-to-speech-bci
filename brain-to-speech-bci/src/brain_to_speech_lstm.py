import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pyttsx3
import os

# Load dataset
df = pd.read_csv('eeg_yes_no_dataset.csv')

# Separate label and features
labels = df['label']
features = df.drop('label', axis=1)

# Convert to numeric
X = features.apply(pd.to_numeric, errors='coerce').fillna(0).values

# Trim extra columns to fit channels
channels = 60
total_features = X.shape[1]
usable_features = total_features - (total_features % channels)
X = X[:, :usable_features]  # Trim columns

# Calculate timesteps
timesteps = usable_features // channels
samples = X.shape[0]
X_reshaped = X.reshape((samples, timesteps, channels))

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, channels)))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Predict
sample = X_test[0].reshape(1, timesteps, channels)
prediction = model.predict(sample)
predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]

# Speak the result
try:
    engine = pyttsx3.init()
    engine.say(f"The predicted thought is {predicted_label}")
    engine.runAndWait()
except:
    print("Text-to-speech error")

# Output
print("Predicted:", predicted_label)