# Original author: Aqib Saeed
# Modified by: Eelis Peltola
# id:240286
# Last modified: 12.12.2017
# Log-scaled mel spectrogram feature and label extraction from US8K dataset, slightly
# modified from http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/

import glob
import os
import librosa
import numpy as np


# Window size for cutting sound into small clips
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


# Extract Mel spectrogram features and labels from all .wav files in sub_dirs
def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60,
                     frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    exceptions = 0
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                sound_clip, s = librosa.load(fn)
            # Some wav files in the dataset will perhaps not open with Librosa, skip them
            except Exception:
                print("Exception with sound clip ", fn)
                exceptions += 1
                continue
            # Labels are encoded in sound file names, see US8K Readme
            label = fn.split('fold')[1].split('-')[1]
            for (start, end) in windows(sound_clip, window_size):
                start = int(start)
                end = int(end)
                if len(sound_clip[start:end]) == int(window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.logamplitude(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    print("Exceptions in total: ", exceptions)
    return np.array(features), np.array(labels, dtype=np.int)


# Perform one hot encoding for labels to ensure labels do not affect training
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot = np.zeros((n_labels, n_unique_labels))
    one_hot[np.arange(n_labels), labels] = 1
    return one_hot


# Make directory if it doesn't exist already
def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)


# Extract fold data with extract_features(...) and save into Numpy arrays
def save_folds(save_directory, data_dir):
    for k in range(1, 11):
        fold_name = 'fold' + str(k)
        print("\nSaving " + fold_name)
        features, labels = extract_features(data_dir, [fold_name])
        labels = one_hot_encode(labels)

        feature_file = os.path.join(save_directory, fold_name + '_x.npy')
        labels_file = os.path.join(save_directory, fold_name + '_y.npy')
        np.save(feature_file, features)
        print("Saved " + feature_file)
        np.save(labels_file, labels)
        print("Saved " + labels_file)


# Extract features from dataset_dir and save into save_dir
dataset_dir = os.path.join(os.pardir, 'UrbanSound8K/audio')
save_dir = os.path.join(os.path.dirname(__file__), 'US8K_folds')
assure_path_exists(save_dir)
save_folds(save_dir, dataset_dir)
