
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
	dataset = read_csv(filename, sep=',', skiprows=18)

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
	filename = './dataset/cleveland297.csv'

	# Number of folds
	k = 10

	# SVM parameters to test on
	C = [0.001, 0.1, 10, 1000, 10000]
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