
# Import SVM
from sklearn import model_selection, svm, pipeline, preprocessing

# Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Import Pandas
from pandas import read_csv, concat, DataFrame

# Import numpy
from numpy import array

# -------------FUNCTIONS---------------------------


# Function to read input file
def read_file(filename):

	# Read input file
	dataset = read_csv(filename, sep=',')

	# Shuffle the dataset
	dataset = dataset.sample(frac = 1).reset_index(drop=True)

	# Split into samples and labels
	dataset_X = dataset.iloc[:, :-1]
	dataset_Y = dataset.iloc[:, -1]

	# Normalize the inputs
	dataset_X = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset_X)

	# Return shuffled dataset
	print('Dataset read.')

	return DataFrame(dataset_X), DataFrame(dataset_Y)


# Function to read and split input dataset
def read_dataset(dataset_X, dataset_Y, k, fold):

	print('Current fold: %d', fold)

	if fold>=k:
		exit('Please enter valid fold number.')

	# Find split position
	n1 = int(len(dataset_X)/k) * fold
	n2 = int(len(dataset_X)/k) * (fold+1)

	print('Split locations for test dataset: %d & %d', n1, n2)

	# Split into training and testing
	train1 = dataset_X.iloc[0:n1, :]
	train2 = dataset_X.iloc[n2:, :]
	X_train = concat([train1, train2])
	X_test = dataset_X.iloc[n1:n2, :]

	train1 = dataset_Y.iloc[0:n1, :]
	train2 = dataset_Y.iloc[n2:, :]
	Y_train = concat([train1, train2])
	Y_test = dataset_Y.iloc[n1:n2, :]

	return X_train, array(Y_train).reshape(len(Y_train)), X_test, array(Y_test).reshape(len(Y_test))


# Function to classify using svm
def classify_svm(dataset_X, dataset_Y, k, C, gamma, kernel1):

	print('\nGiven C: %s', str(C))
	print('Given Gamma: %s', str(gamma))
	print('Given K: %d', k)
	print('Given Kernel: %s', kernel1)

	# List of test accuracies for various folds
	overall_test_acc = []

	# Run for k folds
	for i in range(0, k):

		# Get the Train and Test datasets
		train_images, train_labels, test_images, test_labels = read_dataset(dataset_X, dataset_Y, k, fold=i)

		# Doing this or Defining as step in pipeline produces same result
		# Instantiate Random Forest Classifier
		classifier = RandomForestClassifier(n_estimators=200, max_features = 'auto', random_state=0)

		# Train on the Random Forest
		classifier.fit(train_images, train_labels)

		# Extract important features from Random Forest
		sfm = SelectFromModel(classifier, threshold=0.1)
		sfm.fit(train_images, train_labels)

		train_images = sfm.transform(train_images)
		test_images = sfm.transform(test_images)

		# Define steps to perform on dataset
		steps = [
			#('sel', SelectFromModel(RandomForestClassifier(n_estimators=300, random_state=0), threshold=0.1)),
			('SVM', svm.SVC(kernel=kernel1))
		]

		# Create a pipeline for these steps
		pipeline1 = pipeline.Pipeline(steps) # define Pipeline object

		# Parameters for SVM
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
		print("Current fold's Test Accuracy: %s", str(curr_test_acc))

		# Append score to overall results
		overall_test_acc.append(curr_test_acc)

	final_acc = sum(overall_test_acc)/len(overall_test_acc)
	print('Avg. of all folds\' test accuracies: %s', str(final_acc))

	return final_acc * 100


# Main Function
def main():
	print('-')
	print('Start.')

	# Input filename
	filename = './dataset/cleveland297.csv'

	# Read the input file
	dataset_X, dataset_Y = read_file(filename)

	# Number of folds
	k = 5

	# SVM parameters to test on
	C = [0.01, 0.1, 5, 10, 100, 1000, 10000]
	gamma = [1, 0.1, 0.01,0.001,0.0001]
	kernel = 'rbf'

	print('C:\n', C)
	print('Gamma:\n', gamma)
	print('No. of folds:', k)

	# Final Result Matrix
	res_matrix = []
	for pos, c in enumerate(C):
		res_matrix.append([])
		for g in gamma:
			res_matrix[pos].append(classify_svm(dataset_X, dataset_Y, k, [c], [g], kernel))

	print('\n\nFinal Result Matrix for all parameters:\n', DataFrame(res_matrix))

	print('End.')
	print('-')


# -------------------------------------------------

if __name__ == "__main__":
	main()
