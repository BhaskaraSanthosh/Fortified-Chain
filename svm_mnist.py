# Disable TensorFlow Debug Info
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import MNIST from Keras
from keras.datasets import mnist

# Import SVM
from sklearn import *

# -------------FUNCTIONS---------------------------


# Main Function
def main():

	# Get the MNIST Train and Test datasets
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	# Reshape dataset from 28x28 to 784
	train_images = train_images.reshape((60000, 784))
	test_images = test_images.reshape((10000, 784))

	# Normalize pixels between 0 to 1
	train_images = train_images.astype('float32') / 255
	test_images = test_images.astype('float32') / 255

	# Define steps to perform on dataset
	steps = [
		('SVM', svm.SVC(kernel='poly'))
	]

	# Create a pipeline for these steps
	pipeline1 = pipeline.Pipeline(steps) # define Pipeline object

	# Parameters for SVM
	#parameters = {'SVM__C':[0.001, 0.1, 100, 10e5], 'SVM__gamma':[10, 1, 0.1, 0.01]}
	parameters = {'SVM__C':[100], 'SVM__gamma':[0.1]}

	# Apply steps on to create SVM model
	grid = model_selection.GridSearchCV(pipeline1, param_grid=parameters, cv=5, verbose=2)

	print('Training SVM for 5 epochs...')
	grid.fit(train_images, train_labels)
	print('Done.')

	print('Testing SVM on Test Dataset with parameters:', parameters)
	print ("Test Accuracy = %3.4f" %(grid.score(test_images, test_labels)))


# -------------------------------------------------

if __name__ == "__main__":
	main()

# ----------------OUTPUT---------------------------

'''
Using TensorFlow backend.
Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling
5 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ...................... SVM__C=0.001, SVM__gamma=10, total= 5.4min
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  5.4min remaining:    0.0s
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ...................... SVM__C=0.001, SVM__gamma=10, total= 5.6min
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ...................... SVM__C=0.001, SVM__gamma=10, total= 7.2min
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ...................... SVM__C=0.001, SVM__gamma=10, total= 7.4min
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ...................... SVM__C=0.001, SVM__gamma=10, total= 6.5min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 32.1min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [10]}
Test Accuracy = 0.98


Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV] SVM__C=0.001, SVM__gamma=1 ......................................
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] ....................... SVM__C=0.001, SVM__gamma=1, total= 6.5min
[CV] SVM__C=0.001, SVM__gamma=1 ......................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  6.5min remaining:    0.0s
[CV] ....................... SVM__C=0.001, SVM__gamma=1, total= 6.4min
[CV] SVM__C=0.001, SVM__gamma=1 ......................................
[CV] ....................... SVM__C=0.001, SVM__gamma=1, total= 6.4min
[CV] SVM__C=0.001, SVM__gamma=1 ......................................
[CV] ....................... SVM__C=0.001, SVM__gamma=1, total= 6.5min
[CV] SVM__C=0.001, SVM__gamma=1 ......................................
[CV] ....................... SVM__C=0.001, SVM__gamma=1, total= 6.4min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 32.2min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [1]}
Test Accuracy = 0.98


Using TensorFlow backend.
Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling
5 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=0.1 ....................................
[CV] ..................... SVM__C=0.001, SVM__gamma=0.1, total= 9.9min
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  9.9min remaining:    0.0s
[CV] SVM__C=0.001, SVM__gamma=0.1 ....................................
[CV] ..................... SVM__C=0.001, SVM__gamma=0.1, total= 9.6min
[CV] SVM__C=0.001, SVM__gamma=0.1 ....................................
[CV] ..................... SVM__C=0.001, SVM__gamma=0.1, total= 9.6min
[CV] SVM__C=0.001, SVM__gamma=0.1 ....................................
[CV] ..................... SVM__C=0.001, SVM__gamma=0.1, total=10.3min
[CV] SVM__C=0.001, SVM__gamma=0.1 ....................................
[CV] ..................... SVM__C=0.001, SVM__gamma=0.1, total=11.2min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 50.5min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [0.1]}
Test Accuracy = 0.97


Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV] SVM__C=0.1, SVM__gamma=10 .......................................
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] ........................ SVM__C=0.1, SVM__gamma=10, total= 6.5min
[CV] SVM__C=0.1, SVM__gamma=10 .......................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  6.5min remaining:    0.0s
[CV] ........................ SVM__C=0.1, SVM__gamma=10, total= 6.4min
[CV] SVM__C=0.1, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=0.1, SVM__gamma=10, total= 6.4min
[CV] SVM__C=0.1, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=0.1, SVM__gamma=10, total= 6.4min
[CV] SVM__C=0.1, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=0.1, SVM__gamma=10, total= 6.5min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 32.2min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.1], 'SVM__gamma': [10]}
Test Accuracy = 0.9787


Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling
5 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.01, SVM__gamma=1 .......................................
[CV] ........................ SVM__C=0.01, SVM__gamma=1, total= 6.1min
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  6.1min remaining:    0.0s
[CV] SVM__C=0.01, SVM__gamma=1 .......................................
[CV] ........................ SVM__C=0.01, SVM__gamma=1, total= 5.3min
[CV] SVM__C=0.01, SVM__gamma=1 .......................................
[CV] ........................ SVM__C=0.01, SVM__gamma=1, total= 6.4min
[CV] SVM__C=0.01, SVM__gamma=1 .......................................
[CV] ........................ SVM__C=0.01, SVM__gamma=1, total= 6.7min
[CV] SVM__C=0.01, SVM__gamma=1 .......................................
[CV] ........................ SVM__C=0.01, SVM__gamma=1, total= 5.6min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 30.2min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.01], 'SVM__gamma': [1]}
Test Accuracy = 0.9787


Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV] SVM__C=0.1, SVM__gamma=0.1 ......................................
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] ....................... SVM__C=0.1, SVM__gamma=0.1, total= 6.6min
[CV] SVM__C=0.1, SVM__gamma=0.1 ......................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  6.6min remaining:    0.0s
[CV] ....................... SVM__C=0.1, SVM__gamma=0.1, total= 6.6min
[CV] SVM__C=0.1, SVM__gamma=0.1 ......................................
[CV] ....................... SVM__C=0.1, SVM__gamma=0.1, total= 6.6min
[CV] SVM__C=0.1, SVM__gamma=0.1 ......................................
[CV] ....................... SVM__C=0.1, SVM__gamma=0.1, total= 6.6min
[CV] SVM__C=0.1, SVM__gamma=0.1 ......................................
[CV] ....................... SVM__C=0.1, SVM__gamma=0.1, total= 6.8min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 33.2min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.1], 'SVM__gamma': [0.1]}
Test Accuracy = 0.9786


Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling
5 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=100, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=100, SVM__gamma=10, total= 5.3min
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  5.3min remaining:    0.0s
[CV] SVM__C=100, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=100, SVM__gamma=10, total= 5.3min
[CV] SVM__C=100, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=100, SVM__gamma=10, total= 5.3min
[CV] SVM__C=100, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=100, SVM__gamma=10, total= 5.2min
[CV] SVM__C=100, SVM__gamma=10 .......................................
[CV] ........................ SVM__C=100, SVM__gamma=10, total= 5.3min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 26.4min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [100], 'SVM__gamma': [10]}
Test Accuracy = 0.9787


Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV] SVM__C=100, SVM__gamma=1 ........................................
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] ......................... SVM__C=100, SVM__gamma=1, total= 6.7min
[CV] SVM__C=100, SVM__gamma=1 ........................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  6.7min remaining:    0.0s
[CV] ......................... SVM__C=100, SVM__gamma=1, total= 6.7min
[CV] SVM__C=100, SVM__gamma=1 ........................................
[CV] ......................... SVM__C=100, SVM__gamma=1, total= 6.9min
[CV] SVM__C=100, SVM__gamma=1 ........................................
[CV] ......................... SVM__C=100, SVM__gamma=1, total= 6.5min
[CV] SVM__C=100, SVM__gamma=1 ........................................
[CV] ......................... SVM__C=100, SVM__gamma=1, total= 6.6min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 33.4min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [100], 'SVM__gamma': [1]}
Test Accuracy = 0.9787

Training SVM for 5 epochs...
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV] SVM__C=100, SVM__gamma=0.1 ......................................
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] ....................... SVM__C=100, SVM__gamma=0.1, total= 6.7min
[CV] SVM__C=100, SVM__gamma=0.1 ......................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  6.7min remaining:    0.0s
[CV] ....................... SVM__C=100, SVM__gamma=0.1, total= 7.0min
[CV] SVM__C=100, SVM__gamma=0.1 ......................................
[CV] ....................... SVM__C=100, SVM__gamma=0.1, total= 6.8min
[CV] SVM__C=100, SVM__gamma=0.1 ......................................
[CV] ....................... SVM__C=100, SVM__gamma=0.1, total= 6.6min
[CV] SVM__C=100, SVM__gamma=0.1 ......................................
[CV] ....................... SVM__C=100, SVM__gamma=0.1, total= 6.8min
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed: 34.0min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [100], 'SVM__gamma': [0.1]}
Test Accuracy = 0.9787
'''