
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
	dataset = read_csv(filename, sep=',', skiprows=11)

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
	X_train = train.iloc[:, :-1]
	Y_train = train.iloc[:, -1]

	X_test = test.iloc[:, :-1]
	Y_test = test.iloc[:, -1]

	#print('Train on', len(X_train), 'samples. Test on', len(X_test))
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
			('SVM', svm.SVC(kernel='linear'))
		]

		# Normalise input data
		train_images = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(train_images)
		test_images = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(test_images)

		#print(train_images)

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
	filename = './dataset/bupa.dat'

	# Number of folds
	k = 10

	# SVM parameters to test on 0.001, 0.1, 100, 10e5
	C = [1000]
	gamma = [10, 1, 0.1, 0.01]

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

Now For C       : [0.1]
For Gamma       : [10]
For K           : 10

Current fold: 0
Split locations for test dataset: 0 34
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5882352941176471

Current fold: 1
Split locations for test dataset: 34 68
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7058823529411765

Current fold: 2
Split locations for test dataset: 68 102
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6176470588235294

Current fold: 3
Split locations for test dataset: 102 136
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7941176470588235

Current fold: 4
Split locations for test dataset: 136 170
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6176470588235294

Current fold: 5
Split locations for test dataset: 170 204
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6470588235294118

Current fold: 6
Split locations for test dataset: 204 238
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.8235294117647058

Current fold: 7
Split locations for test dataset: 238 272
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7352941176470589

Current fold: 8
Split locations for test dataset: 272 306
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5588235294117647

Current fold: 9
Split locations for test dataset: 306 340
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.4117647058823529

Avg. of all test accuracies: 0.65



Now For C       : [0.1]
For Gamma       : [1]
For K           : 10

Current fold: 0
Split locations for test dataset: 0 34
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.38235294117647056
Current fold: 1
Split locations for test dataset: 34 68
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6764705882352942

Current fold: 2
Split locations for test dataset: 68 102
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5294117647058824

Current fold: 3
Split locations for test dataset: 102 136
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7352941176470589

Current fold: 4
Split locations for test dataset: 136 170
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5882352941176471

Current fold: 5
Split locations for test dataset: 170 204
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5294117647058824

Current fold: 6
Split locations for test dataset: 204 238
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7058823529411765

Current fold: 7
Split locations for test dataset: 238 272
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6470588235294118

Current fold: 8
Split locations for test dataset: 272 306
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5588235294117647

Current fold: 9
Split locations for test dataset: 306 340
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5

Avg. of all test accuracies: 0.5852941176470589



Now For C       : [0.1]
For Gamma       : [0.1]
For K           : 10

Current fold: 0
Split locations for test dataset: 0 34
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.29411764705882354
Current fold: 1
Split locations for test dataset: 34 68
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7647058823529411

Current fold: 2
Split locations for test dataset: 68 102
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6176470588235294

Current fold: 3
Split locations for test dataset: 102 136
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7058823529411765

Current fold: 4
Split locations for test dataset: 136 170
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5882352941176471

Current fold: 5
Split locations for test dataset: 170 204
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.38235294117647056
Current fold: 6
Split locations for test dataset: 204 238
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5882352941176471

Current fold: 7
Split locations for test dataset: 238 272
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.9117647058823529

Current fold: 8
Split locations for test dataset: 272 306
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.4411764705882353

Current fold: 9
Split locations for test dataset: 306 340
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.47058823529411764
Avg. of all test accuracies: 0.5764705882352942



Now For C       : [0.1]
For Gamma       : [0.01]
For K           : 10

Current fold: 0
Split locations for test dataset: 0 34
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.29411764705882354
Current fold: 1
Split locations for test dataset: 34 68
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7647058823529411

Current fold: 2
Split locations for test dataset: 68 102
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.6176470588235294

Current fold: 3
Split locations for test dataset: 102 136
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.7058823529411765

Current fold: 4
Split locations for test dataset: 136 170
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5882352941176471

Current fold: 5
Split locations for test dataset: 170 204
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.38235294117647056
Current fold: 6
Split locations for test dataset: 204 238
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.5882352941176471

Current fold: 7
Split locations for test dataset: 238 272
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.9117647058823529

Current fold: 8
Split locations for test dataset: 272 306
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.4411764705882353

Current fold: 9
Split locations for test dataset: 306 340
Training SVM...
Done.
Testing SVM...
Done.
Current Test Accuracy =  0.47058823529411764
Avg. of all test accuracies: 0.5764705882352942


Final Result Matrix for all parameters:
       0         1         2         3
0  0.65  0.585294  0.576471  0.576471
-

-
'''