from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import Dropout
from keras.layers import concatenate

from keras.callbacks import *
import tensorflow as tf
from keras import initializers
from keras.layers import BatchNormalization
K.set_image_dim_ordering('tf')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def dice_coef(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 0.
    intersection = K.sum(y_true*y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)



def get_unet_small():
    inputs = Input((512, 512, 1))
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='elu',padding='same', name='conv_1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='elu',padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='elu',padding='same', name='conv_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='elu',padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='elu',padding='same', name='conv_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='elu',padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='elu',padding='same', name='conv_4')(conv4)
    conv4 = BatchNormalization()(conv4)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis = 3)
    conv7 = Conv2D(128, (3, 3), activation='elu',padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='elu',padding='same', name='conv_7')(conv7)
    conv7 = BatchNormalization()(conv7)

    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2],axis = 3)
    conv8 = Conv2D(64, (3, 3), activation='elu',padding='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='elu',padding='same', name='conv_8')(conv8)
    conv8 = BatchNormalization()(conv8)

    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1],axis = 3)
    conv9 = Conv2D(32, (3, 3), activation='elu',padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='elu',padding='same', name='conv_9')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=0.001, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train():
    epochs = 500
    batch_size = 10

    imgs_train = np.load("/ouput/normalized_trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load("/ouput/trainMasks.npy").astype(np.float32)
    imgs_train = imgs_train.transpose(0,2,3,1)
    imgs_mask_train = imgs_mask_train.transpose(0,2,3,1)

    
    # Renormalizing the masks
    imgs_mask_train[imgs_mask_train > 0.] = 1.0
    
    # Now the Test Data
    imgs_test = np.load("/ouput/normalized_testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load("/output/testMasks.npy").astype(np.float32)
    imgs_test = imgs_test.transpose(0,2,3,1)
    imgs_mask_test_true = imgs_mask_test_true.transpose(0,2,3,1)
    
    # Renormalizing the test masks
    imgs_mask_test_true[imgs_mask_test_true > 0] = 1.0

    print('Creating and compiling model...')
    model = get_unet_small()

    #accuracy = Accuracy(copy.deepcopy(imgs_test),copy.deepcopy(imgs_mask_test_true))
    filepath="/saved_model/weights-improvement.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max', epochs = 1)
    print('Fitting model...')
    model.fit(x=imgs_train, y=imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True
            ,validation_data = (imgs_test, imgs_mask_test_true), callbacks = [checkpoint])
    return model

def plot_stats(unet):
	fig = plt.figure()
	plt.plot(unet.history.history['dice_coef'])
	plt.plot(unet.history.history['val_dice_coef'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')
	fig.save('/output/accuracy_stats.png')
	
	fig = plt.figure()
	plt.plot(unet.history.history['loss'])
	plt.plot(unet.history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')
	fig.save('/output/loss_stats.png')

	
if __name__ == '__main__':
	unet = train()
	print('Successfully trained..')
	plot_stats(unet)



