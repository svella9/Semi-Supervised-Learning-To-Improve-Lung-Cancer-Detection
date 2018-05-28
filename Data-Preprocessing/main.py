import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import preprocessing

if __name__ == '__main__':
	INPUT_FOLDER = os.path.join(os.getcwd(), 'stage1')
	OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
	preprocessing.full_prep(INPUT_FOLDER, OUTPUT_FOLDER)