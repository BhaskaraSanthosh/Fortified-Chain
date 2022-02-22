
# Import SVM
from sklearn import model_selection, svm, pipeline, preprocessing

# Import Pandas
from pandas import read_csv, concat, DataFrame

# -------------FUNCTIONS---------------------------


# Function to read and split input dataset
def read_dataset(filename, k, fold):

	print('Current fold:', fold)

	if fold>=k:
		exit('Please enter valid fold number.')

	# Read input file
	dataset = read_csv(filename)

	# Find split position
	n1 = int(len(dataset)/k) * fold
	n2 = int(len(dataset)/k) * (fold+1)

	print('Split locations for test dataset:', n1, n2)

	# Split into training and testing
	train1 = dataset.iloc[0:n1, :]
	train2 = dataset.iloc[n2:, :]

	train = concat([train1, train2])
	test = dataset.iloc[n1:n2, :]

	# Split into features and labels
	X_train = train.iloc[:, 1:]
	Y_train = train.iloc[:, 0]

	X_test = test.iloc[:, 1:]
	Y_test = test.iloc[:, 0]

	print('Train on', len(X_train), 'samples. Test on', len(X_test))
	return X_train, Y_train, X_test, Y_test


# Function to classify using svm
def classify_svm(filename, k, C, gamma):

	print('\n\n')
	print('Now For C\t:', C)
	print('For Gamma\t:', gamma)
	print('For K    \t:', k)

	# List of test accuracies for various folds
	overall_test_acc = []

	# Run for k folds
	for i in range(0, k):
		print()

		# Get the Train and Test datasets
		train_images, train_labels, test_images, test_labels = read_dataset(filename, k, fold=i)

		# Define steps to perform on dataset
		steps = [
			('SVM', svm.SVC(kernel='poly'))
		]

		# Normalise input data
		train_images = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(train_images)
		test_images = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(test_images)

		# Create a pipeline for these steps
		pipeline1 = pipeline.Pipeline(steps) # define Pipeline object

		# Parameters for SVM
		#parameters = {'SVM__C':[0.001, 0.1, 100, 10e5], 'SVM__gamma':[10, 1, 0.1, 0.01]}    #[(slice(None), slice(None))]
		parameters = {'SVM__C':C, 'SVM__gamma':gamma}

		# Apply steps on to create SVM model
		grid = model_selection.GridSearchCV(pipeline1, param_grid=parameters, cv=[(slice(None), slice(None))], verbose=0, n_jobs=1)

		print('Training SVM...')
		grid.fit(train_images, train_labels)
		print('Done.')

		# Test it on given dataset for current fold
		print('Testing SVM...')
		curr_test_acc = grid.score(test_images, test_labels)
		print('Done.')
		print ("Current Test Accuracy = ", curr_test_acc)

		# Append score to overall results
		overall_test_acc.append(curr_test_acc)

	final_acc = sum(overall_test_acc)/len(overall_test_acc)
	print('\nAvg. of all test accuracies:', final_acc)


	return final_acc


# Main Function
def main():
	print('-')

	# Input filename
	filename = './dataset/mnist.csv'

	# Number of folds
	k = 5

	# SVM parameters to test on
	C = [0.001, 0.1, 100]
	gamma = [10, 1, 0.1]

	# Final Result Matrix
	res_matrix = []
	for pos, c in enumerate(C):
		res_matrix.append([])
		for g in gamma:
			res_matrix[pos].append(classify_svm(filename, k, [c], [g]))

	print('\n\nFinal Result Matrix for all parameters:\n', DataFrame(res_matrix))

	print('-')


# -------------------------------------------------

if __name__ == "__main__":
	main()

# ----------------OUTPUT---------------------------

''' PROGRAM OUTPUT
-
Current fold: 0
Split locations for test dataset: 0 12000
Train on 48000 Test on 12000
Training SVM...
Fitting 1 folds for each of 1 candidates, totalling 1 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ......... SVM__C=0.001, SVM__gamma=10,
score=1.000, total=10.1min
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed: 10.1min remaining:    0.0s
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed: 10.1min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [10]}
Current Test Accuracy =  0.97725


Current fold: 1
Split locations for test dataset: 12000 24000
Train on 48000 Test on 12000
Training SVM...
Fitting 1 folds for each of 1 candidates, totalling 1 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ......... SVM__C=0.001, SVM__gamma=10,
score=1.000, total= 9.9min
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed:  9.9min remaining:    0.0s
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed:  9.9min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [10]}
Current Test Accuracy =  0.9759166666666667


Current fold: 2
Split locations for test dataset: 24000 36000
Train on 48000 Test on 12000
Training SVM...
Fitting 1 folds for each of 1 candidates, totalling 1 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ......... SVM__C=0.001, SVM__gamma=10,
score=1.000, total= 9.9min
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed:  9.9min remaining:    0.0s
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed:  9.9min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [10]}
Current Test Accuracy =  0.9749166666666667


Current fold: 3
Split locations for test dataset: 36000 48000
Train on 48000 Test on 12000
Training SVM...
Fitting 1 folds for each of 1 candidates, totalling 1 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ......... SVM__C=0.001, SVM__gamma=10,
score=1.000, total=10.2min
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed: 10.2min remaining:    0.0s
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed: 10.2min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [10]}
Current Test Accuracy =  0.97325


Current fold: 4
Split locations for test dataset: 48000 60000
Train on 48000 Test on 12000
Training SVM...
Fitting 1 folds for each of 1 candidates, totalling 1 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[CV] SVM__C=0.001, SVM__gamma=10 .....................................
[CV] ......... SVM__C=0.001, SVM__gamma=10,
score=1.000, total= 9.6min
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed:  9.6min remaining:    0.0s
[Parallel(n_jobs=1)]: Done   1 out of   1 |
elapsed:  9.6min finished
Done.
Testing SVM on Test Dataset with parameters: {'SVM__C': [0.001], 'SVM__gamma': [10]}
Current Test Accuracy =  0.97775

Avg. Test Accuracy: 0.9758166666666668
-
'''