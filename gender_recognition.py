# Author: Eelis Peltola
# id:240286
# Last modified: 12.12.2017
# Trains a simple NN classifier for voice.csv, a file with voice audio frequency data and gender tags

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np


# Use data from voice.csv file
parent_dir = "/mnt/01D1942B38462240/Users/Eelis/Documents/Dokumentit/urban_informatics/codes"
voice_file = os.path.join(parent_dir, 'voice.csv')


# Load data from voice.csv file
def load_data():
    df = pd.read_csv(voice_file)
    data = np.array(df.iloc[:])
    formatted = dict()
    formatted['data'] = data[:, :-1]
    formatted['target'] = data[:, -1]
    formatted['target'][formatted['target'] == 'male'] = 0
    formatted['target'][formatted['target'] == 'female'] = 1
    return formatted


# Preprocess data into train and test sets
voice_data = load_data()
X = voice_data['data']
Y = voice_data['target']
X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Build model
model = Sequential([
    Dense(128, input_dim=20, activation='sigmoid'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Early stop if no improvement in model
early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

# Train model, still fast with 1000 epochs but could be lowered
model.fit(X_train, Y_train, epochs=1000, batch_size=64, callbacks=[early_stop])
# Save model to root
model.save('voicemodel.h5')
print('\n')

# Test model
acc = model.evaluate(X_test, Y_test)[1]*100
print('\nAccuracy = {} %'.format(acc))
