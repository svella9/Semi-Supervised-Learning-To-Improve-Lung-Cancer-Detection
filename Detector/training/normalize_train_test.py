import numpy as np

def normalize():
	trainImages = np.load('/output/trainImages.npy')
	train_lungwin = np.array([np.amin(trainImages), np.amax(trainImages)])
	print(train_lungwin)
	normalized_train = (trainImages - train_lungwin[0]) / (train_lungwin[1] - train_lungwin[0])
	normalized_train[normalized_train < 0] = 0
	normalized_train[normalized_train > 1] = 1
	normalized_train = (normalized_train * 255).astype('uint8')
	np.save('/output/normalized_trainImages.npy', normalized_train)
	print('Successfully normalized train set')
	del trainImages
	del normalized_train
	
	testImages = np.load('/output/testImages.npy')
	test_lungwin = np.array([np.amin(testImages), np.amax(testImages)])
	print(test_lungwin)
	normalized_test = (testImages - test_lungwin[0]) / (test_lungwin[1] - test_lungwin[0])
	normalized_test[normalized_test < 0] = 0
	normalized_test[normalized_test > 1] = 1
	normalized_test = (normalized_test * 255).astype('uint8')
	np.save('/output/normalized_testImages.npy', normalized_test)
	print('Successfully normalized test set')
	del testImages
	del normalized_test

if __name__ == '__main__':
	normalize()