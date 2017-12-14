# Author: Eelis Peltola
# id:240286
# Last modified: 12.12.2017
# Trains a CNN for urban sound source classification for the UrbanSounds8K dataset.
# CNN structure is from Salamon and Bello, proposed in https://arxiv.org/pdf/1608.04363.pdf.

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.regularizers import l2
# from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.initializers import RandomNormal
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

# Set TF (graph-level) and Numpy random seeds for predictable randomness
tf.set_random_seed(0)
np.random.seed(0)

# Set parameters for model
frames = 41
bands = 60
feature_size = bands*frames
num_channels = 2
input_shape = (bands, frames, num_channels)
num_labels = 10


# Build model with the structure from Salamon & Bello
def build_model():
    # Paper describes 5x5 kernel, but only 3x3 fits with data
    k_size = 3

    model = Sequential()

    # First layer
    model.add(Conv2D(24, kernel_size=(k_size, k_size), kernel_initializer="normal", padding="same",
                     input_shape=(bands, frames, num_channels)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Second layer
    model.add(Conv2D(48, kernel_size=(k_size, k_size), kernel_initializer="normal", padding="same"))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Third layer
    model.add(Conv2D(48, kernel_size=(k_size, k_size), padding="valid"))
    model.add(Activation('relu'))

    # Flatten to let Keras do shape inference
    model.add(Flatten())

    # Fourth layer
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fifth layer
    model.add(Dense(num_labels, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    # Compile with categorical crossentropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer="adam")

    return model


# Calculate model accuracy, ROC and f-score
def evaluate(model, test_x, test_y):
    y_proba = model.predict_proba(test_x, verbose=0)
    y_predi = y_proba.argmax(axis=-1)
    ytrue = np.argmax(test_y, 1)

    roc_score = roc_auc_score(test_y, y_proba)
    _, accuracy = model.evaluate(test_x, test_y, batch_size=32)
    print("\nROC: {:.3f}".format(roc_score))
    print("Accuracy = {:.2f}".format(accuracy))
    _, _, f, _ = precision_recall_fscore_support(ytrue, y_predi, average='micro')
    print("F-Score: {:.2f}".format(f))

    return roc_score, accuracy


# Load training, test, and validation folds from numpy arrays.
# Uses folds 1...8 for training, fold 9 for validation and fold 10 for testing
def load_train_test_valid(data_dir):
    train_feature = None
    train_label = None
    for k in range(1, 9):
        fold = 'fold'+str(k)
        print("\nAdding " + fold + "...")
        feature_fn = os.path.join(data_dir, fold+'_x.npy')
        label_fn = os.path.join(data_dir, fold+'_y.npy')
        loaded_f = np.load(feature_fn)
        loaded_l = np.load(label_fn)
        if k == 1:
            train_feature = loaded_f
            train_label = loaded_l
        else:
            train_feature = np.concatenate((train_feature, loaded_f), axis=0)
            train_label = np.concatenate((train_label, loaded_l), axis=0)

    # Validation data from fold 9
    valid_fold = 'fold'+str(9)
    valid_feature_fn = os.path.join(data_dir, valid_fold+'_x.npy')
    valid_label_fn = os.path.join(data_dir, valid_fold+'_y.npy')
    valid_feature = np.load(valid_feature_fn)
    valid_label = np.load(valid_label_fn)

    # Test data from fold 10
    test_fold = 'fold'+str(10)
    test_feature_fn = os.path.join(data_dir, test_fold+'_x.npy')
    test_label_fn = os.path.join(data_dir, test_fold+'_y.npy')
    test_feature = np.load(test_feature_fn)
    test_label = np.load(test_label_fn)

    print("Features loaded from ", data_dir)
    return train_feature, train_label, valid_feature, valid_label, test_feature, test_label


# Early stop for if there is no improvement in model
early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
callback_list = [early_stop]

# Load training, validation, and test data
fold_dir = os.path.join(os.path.dirname(__file__), 'US8K_folds')
train_x, train_y, valid_x, valid_y, test_x, test_y = load_train_test_valid(fold_dir)

print("Building model...")
cnn_model = build_model()
print("Training model...")
cnn_model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), callbacks=callback_list, batch_size=128,
              epochs=50)

# Save model
# filepath = os.path.join(os.path.dirname(__file__), "salamon-cnn.h5")
# cnn_model.save(filepath)

# Evaluate model and calculate average ROC and accuracy
print("Evaluating model...")
roc, acc = evaluate(cnn_model, test_x, test_y)
# Calculate and plot confusion matrix
y_prob = cnn_model.predict_proba(test_x, verbose=0)
y_pred = y_prob.argmax(axis=-1)
y_true = np.argmax(test_y, 1)
labels = ["air-con", "horn", "children", "dog", "drill", "engine", "gun",
          "hammer", "siren", "music"]
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, labels, labels)

print("Showing Confusion Matrix")
plt.figure(figsize=(12,6))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 11}, fmt='g', linewidths=0.1)
plt.show()
# plot_fn = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
# plt.savefig(plot_fn)

