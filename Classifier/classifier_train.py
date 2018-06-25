import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold

from keras.callbacks import *
import tensorflow as tf
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('tf')

import GAN

seed = 7
np.random.seed(seed)

def classifier_model():
    num_classes = 3

    model = Sequential()
    model.add(Conv2D(16, kernel_size = (3,3), input_shape = (5, 72, 72), padding = 'valid', data_format = 'channels_first'))
    model.add(ELU(alpha = 0.7))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, kernel_size = (3,3), padding = 'same'))
    model.add(ELU(alpha = 0.7))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, kernel_size = (3,3), padding = 'same'))
    model.add(ELU(alpha = 0.7))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(ELU(alpha = 0.7))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation = 'softmax'))
    #model.summary()
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
    return model


def prepare_data():
    X_patient = []
    Y_patient = []

    for patient in os.listdir('/data_dir/cancer/'):
        arr = np.load('/data_dir/cancer/' + patient)
        X_patient.append(arr)
        Y_patient.append(1)

    for patient in os.listdir('/data_dir/non_cancer/'):
        arr = np.load('/data_dir/non_cancer/' + patient)
        X_patient.append(arr)
        Y_patient.append(0)

    X_patient = np.array(X_patient).astype(np.float32)
    X_patient = X_patient / 255
    
    gobj = GAN.Generator()
    gen_list = []
    for i in range(300):
        gen_list.append(gobj.generate_samples())
        Y_patient.append(2)
    
    gen_list = np.array(gen_list)
    X_patient = np.concatenate((X_patient, gen_list))
    Y_patient = np.array(Y_patient)
    print(X_patient.shape)
    idx = np.random.permutation(len(X_patient))
    X_patient, Y_patient = X_patient[idx], Y_patient[idx]
    
    return X_patient, Y_patient


def train():
    epochs = 100
    batch_size = 20
    n_classes = 3

    #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    X, Y = prepare_data()
    k = 10
    model = classifier_model()
    hist = []
    for i in range(k):
        idx = np.random.permutation(len(X))
        train = idx[:-100]
        test = idx[-100: ]

        y_train = keras.utils.to_categorical(Y[train])
        y_test = keras.utils.to_categorical(Y[test])
        
        print(i + 'th fold...')
        filepath="/output/{}th-fold-weights-improvement.hdf5".format(i)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # Fit the model
        m_t = model.fit(X[train], y_train, epochs=20, batch_size=20, verbose=1, validation_data = (X[test], y_test), callbacks = [checkpoint])
        hist.append(m_t)
        print("Testting on random sample- Predicted:",model.predict(X[test][0].reshape(1,5,72,72)), "Actual:",y_test[0])
        # evaluate the model
        scores = model.evaluate(X[test], y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    return model, hist

model, hist = train()

