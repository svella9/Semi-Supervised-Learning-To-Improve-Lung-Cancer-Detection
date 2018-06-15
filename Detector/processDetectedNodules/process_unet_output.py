import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras import backend as K
from keras.callbacks import *
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
K.set_image_dim_ordering('tf')

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


def generate_masks_from_unet(scans):
	global unet_model
	# Make sure all the slices are of shape 512 * 512. Else pad them with 170.
	n_rows_pad = ( (512 - scans.shape[1]) // 2, (512 - scans.shape[1]) // 2 if (512 - scans.shape[1]) % 2 == 0 else (512 - scans.shape[1]) // 2 + 1)
	n_cols_pad = ( (512 - scans.shape[2]) // 2, (512 - scans.shape[2]) // 2 if (512 - scans.shape[2]) % 2 == 0 else (512 - scans.shape[2]) // 2 + 1)
	scans = np.pad(scans, pad_width = ((0,0), n_rows_pad, n_cols_pad), mode='constant', constant_values=170)
	
	#Reshape the scans to include n_channels in their shape.
	scans = scans.reshape((scans.shape[0], scans.shape[1], scans.shape[2], 1))	
	scans = scans.astype(np.float32)
	predicted_masks = unet_model.predict(scans)
	
	scans = scans[0].reshape(512,512)
	predicted_masks = predicted_masks[0].reshape(512,512)

	res = np.where(predicted_masks == 1.0, scans, 0.0)
	return res

def crop_nodules(masked_scan):
	if np.all(masked_scan == 0):
		return []
	
	structure = np.ones((3, 3), dtype=np.int)
	labeled, ncomponents = label(masked_scan, structure)
	props = list(regionprops(labeled))
	labels = list(np.unique(labeled))
	labels.remove(0)
	result = []
	for i, l in enumerate(labels):
		try:
			indices = np.indices(masked_scan.shape).T[:,:,[1, 0]]
			indices = indices[labeled == l]

			crop_img = masked_scan[indices[0][0]:indices[-1][0] + 1, indices[0][1]:indices[-1][1] + 1]
			n_rows_pad = ( (72 - crop_img.shape[0]) // 2, (72 - crop_img.shape[0]) // 2 if (72 - crop_img.shape[0]) % 2 == 0 else (72 - crop_img.shape[0]) // 2 + 1)
			n_cols_pad = ( (72 - crop_img.shape[1]) // 2, (72 - crop_img.shape[1]) // 2 if (72 - crop_img.shape[1]) % 2 == 0 else (72 - crop_img.shape[1]) // 2 + 1)
			crop_img = np.pad(crop_img, pad_width = (n_rows_pad, n_cols_pad), mode='constant', constant_values=0)
			if np.all(crop_img == 0) == False:
				result.append((crop_img, props[i]))

		except Exception as e:
			print(e)
	return result


unet_model = load_model('/saved_model/weights-improvement.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})
print('Model loaded!!')
patients_dir = "../Data-Preprocessing/output"
cancer_nodule_dir = "/output/"

os.makedirs(cancer_nodule_dir)
patients = os.listdir(patients_dir)

print('patients count:', len(patients))

for patient_file in patients: 
	print('Processing ', patient_file)
	try:
		patient_scans = np.load(patients_dir + patient_file)[0]
		top_five_nodules = []

		for scan in patient_scans:
			scan = scan.reshape((1, scan.shape[0], scan.shape[1]))
			result = generate_masks_from_unet(scan)
			crop_images_n_area = crop_nodules(result)
			top_five_nodules.extend(crop_images_n_area)
			top_five_nodules.sort(key = lambda x : x[1].equivalent_diameter, reverse = True)
			top_five_nodules = top_five_nodules[:5] if len(top_five_nodules) > 5 else top_five_nodules
		
		np.save(cancer_nodule_dir + patient_file.replace('_clean.npy','_nodules.npy'), np.array([x[0] for x in top_five_nodules]))
		print('Done..')

	except Exception as e:
		print(e)