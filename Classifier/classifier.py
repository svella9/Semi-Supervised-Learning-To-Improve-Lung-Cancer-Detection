import os
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K
from keras import backend as K
K.set_image_dim_ordering('tf')

seed = 7
np.random.seed(seed)

def classifier(data_path):
	classifier_model = load_model('./saved_model/6th-fold-weights-improvement.hdf5')
	print('Classifier Model loaded!!')

	patients = 	os.listdir(data_path)
	for patient_file in patients: 
		print('Processing ', patient_file)
		nodules_imgs = np.load(os.path.join(data_path,patient_file))
		nodules_imgs = nodules_imgs.reshape((1,) + nodules_imgs.shape)
		prediction = classifier_model.predict(nodules_imgs)
		return prediction