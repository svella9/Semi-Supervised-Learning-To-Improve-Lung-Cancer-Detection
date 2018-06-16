from keras.models import Sequential, Model
from keras.layers.merge import _Merge
from keras.optimizers import RMSprop
from functools import partial
from keras.models import load_model
import keras.backend as K
from glob import glob
import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        weights = K.random_uniform((32, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class Generator:
    def __init__(self):
        self.generator_model = load_model('1000_generator_epoch.hdf5', custom_objects = {'wasserstein_loss' : wasserstein_loss})
    
    def generate_samples(self):
        noise = np.random.normal(0, 1, (5, 100))
        gen_imgs = self.generator_model.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 1
        min_max = [np.amin(gen_imgs), np.amax(gen_imgs)]
        gen_imgs = (gen_imgs - min_max[0]) / (min_max[1]  - min_max[0])
        gen_imgs = gen_imgs.reshape(5,72,72)
        #return gen_imgs
        #print(np.unique(gen_imgs))
        for i in range(gen_imgs.shape[0]):
            img = gen_imgs[i].reshape((72,72))
            print(gen_imgs.shape)
            plt.imshow(img, cmap = plt.cm.gray)
            plt.show()


obj = Generator()
obj.generate_samples()