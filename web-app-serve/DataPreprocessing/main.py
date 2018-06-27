import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import preprocessing

if __name__ == '__main__':
	INPUT_FOLDER = 'C:/Users/KNatarajan/Desktop/Project/stage1'
	OUTPUT_FOLDER = 'C:/Users/KNatarajan/Desktop/Project/output'
	preprocessing.full_prep(INPUT_FOLDER, OUTPUT_FOLDER)